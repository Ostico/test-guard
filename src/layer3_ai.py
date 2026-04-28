"""Layer 3: Per-file evaluator (deterministic shortcuts + AI fallthrough).

Evaluates each source file through an 8-gate shortcut truth table using
coverage data from L1 and test-match data from L2. Files that cannot be
resolved deterministically fall through to the GitHub Models AI for a
structured JSON verdict with confidence-based downgrade.

Flow:
  1. Shortcut phase: Gates 1-8 resolve most files without AI.
  2. AI phase: Remaining files are batched and sent to the model.
  3. Result assembly: Shortcut + AI verdicts merged; confidence-based
     downgrade converts low-confidence FAIL → WARNING.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import TypedDict, cast

from openai import APIStatusError, OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema

from src.models import FileVerdict, LayerResult, Verdict


class Relevance(Enum):
    """Tri-state test relevance for a source file.

    YES: a changed test file is demonstrably linked to this source file.
    NO: no test files were changed in this PR at all.
    UNKNOWN: test files changed but none could be linked to this source file.
    """

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


def compute_test_relevance(
    source_file: str,
    changed_test_files: list[str],
    l2_matched_test: str | None,
    test_diffs: dict[str, str],
) -> Relevance:
    """Determine whether any changed test file is relevant to this source file.

    Uses three detection strategies in order of confidence:
      1. L2 exact match: Layer 2 already identified a paired test file.
      2. Filename stem match: test file name contains the source file's stem.
      3. Diff content match: source file stem appears in the test diff text.

    Returns UNKNOWN (not NO) when test files exist but none match — the
    distinction matters for Gate 6/8: UNKNOWN sends the file to AI, while
    NO triggers an immediate FAIL (no tests at all in the PR).

    Args:
        source_file: Path of the source file being evaluated.
        changed_test_files: All test files modified in this PR.
        l2_matched_test: The test file L2 paired with this source, or None.
        test_diffs: Raw diff text keyed by test file path.

    Returns:
        Relevance.YES if a link is found, NO if no tests changed, UNKNOWN otherwise.
    """
    if not changed_test_files:
        # No test files touched in this PR — relevance is definitively NO.
        return Relevance.NO

    source_stem = PurePosixPath(source_file).stem.lower()

    for test_file in changed_test_files:
        if l2_matched_test is not None and test_file == l2_matched_test:
            # Strategy 1: L2 already did the hard work of matching.
            return Relevance.YES
        if source_stem in PurePosixPath(test_file).stem.lower():
            # Strategy 2: e.g. "billing" in "test_billing.py".
            return Relevance.YES
        diff_text = test_diffs.get(test_file, "")
        if source_stem in diff_text.lower():
            # Strategy 3: source name referenced inside the test diff body.
            return Relevance.YES

    # Tests changed but none linked — ambiguous, not absent.
    return Relevance.UNKNOWN


_IMPORT_RE = re.compile(
    r"(?:"
    r"^import\s"
    r"|^from\s+\S+\s+import\s"
    r"|require\s*\("
    r"|^include\s"
    r"|^#include\b"
    r"|^use\s"
    r")",
    re.IGNORECASE,
)

_COMMENT_PREFIXES = ("//", "/*", "*", "--", "#")


def is_trivial_diff(diff: str) -> bool:
    """Return True if the diff contains only whitespace or comment changes.

    Import changes are treated as non-trivial even though they look like
    single-line additions — they alter module dependencies and can introduce
    side effects that warrant test coverage.

    A diff is trivial only when every changed line (+ or -) is either blank,
    a comment, or pure whitespace after stripping the diff prefix.
    """
    for line in diff.splitlines():
        if not (line.startswith("+") or line.startswith("-")):
            continue
        content = line[1:].strip()
        if not content:
            continue
        if _IMPORT_RE.search(content):
            # Import changes affect module dependencies — not trivial.
            return False
        if content.startswith(_COMMENT_PREFIXES):
            continue
        return False
    return True


def evaluate_file_shortcut(
    source_file: str,
    diff: str,
    is_deleted: bool,
    coverage_details: dict[str, float] | None,
    coverage_threshold: float,
    test_relevance: Relevance,
) -> Verdict | None:
    """Apply the 8-gate deterministic truth table to a single source file.

    Gates are evaluated in order; the first match returns a Verdict.
    Returns None when no gate fires — the file must go to the AI.

    Args:
        source_file: Path of the source file.
        diff: Raw unified diff for this file.
        is_deleted: True if the file was removed in this PR.
        coverage_details: Per-file coverage percentages from L1, or None.
        coverage_threshold: Minimum coverage % to auto-pass.
        test_relevance: Tri-state relevance computed by compute_test_relevance().

    Returns:
        A Verdict if a shortcut applies, or None to fall through to AI.
    """
    # Gate 1: Deleted files have no remaining code to test.
    if is_deleted:
        return Verdict.SKIP

    # Gate 2: Whitespace/comment-only diffs don't need tests.
    if is_trivial_diff(diff):
        return Verdict.SKIP

    has_coverage = coverage_details is not None and source_file in coverage_details
    coverage_ok = (
        has_coverage
        and coverage_details is not None
        and coverage_details[source_file] >= coverage_threshold
    )

    # Gate 3: Coverage meets threshold — existing tests already cover the changes.
    if coverage_ok:
        return Verdict.PASS

    # Gate 4: No tests changed in this PR and coverage is absent/low — clear FAIL.
    if test_relevance == Relevance.NO:
        return Verdict.FAIL

    # Gate 5: Coverage data exists, relevant tests exist, but coverage is below
    # threshold — tests are present but insufficient.
    if has_coverage and test_relevance == Relevance.YES:
        return Verdict.FAIL

    # Gates 6-8: Ambiguous cases (UNKNOWN relevance, or no coverage data with
    # YES relevance) — fall through to AI for judgment.
    return None


@dataclass
class Layer3Result:
    """Intermediate result container used internally by run_layer3().

    Holds per-file verdicts and an execution status string so the verdict
    property can distinguish between "no files evaluated" (PASS) and
    "AI failed with no fallback" (SKIP).
    """

    per_file_verdicts: dict[str, Verdict]
    execution_status: str  # "OK" or "ERROR"

    @property
    def verdict(self) -> Verdict:
        if self.execution_status == "ERROR" and not self.per_file_verdicts:
            return Verdict.SKIP

        verdicts = list(self.per_file_verdicts.values())
        if not verdicts:
            return Verdict.PASS

        non_skip = [v for v in verdicts if v != Verdict.SKIP]
        if not non_skip:
            return Verdict.PASS

        if Verdict.FAIL in non_skip:
            return Verdict.FAIL
        if Verdict.WARNING in non_skip:
            return Verdict.WARNING
        return Verdict.PASS


def _build_ai_prompt(
    files_for_ai: list[str],
    source_diffs: dict[str, str],
    test_diffs: dict[str, str],
    coverage_details: dict[str, float] | None,
    coverage_threshold: float,
    matched_tests: dict[str, str | None],
    max_diff_chars: int = 10_000,
) -> str:
    """Build the structured Markdown user prompt for the AI batch.

    The prompt has three sections:
      1. Coverage Summary — per-file coverage percentages.
      2. Source File Changes — diffs for files needing AI review.
      3. Test File Changes — relevant test diffs, each annotated as either
         "matched to <source>" or "candidate" (unmatched to any source).

    The matched/candidate annotation tells the AI which test files are
    definitively linked to a source file vs. which are circumstantial.
    """
    if not files_for_ai:
        return ""

    # Build a reverse map: test file → list of source files it is matched to.
    # Used to annotate each test diff in the prompt so the AI knows which
    # source files a given test is authoritative for.
    matched_test_to_sources: dict[str, list[str]] = {}
    for src, test in matched_tests.items():
        if test is not None:
            matched_test_to_sources.setdefault(test, []).append(src)

    parts: list[str] = ["## Coverage Summary", "Per-file changed-line coverage:"]

    for src in files_for_ai:
        if coverage_details is not None and src in coverage_details:
            pct = coverage_details[src]
            parts.append(
                f"- {src}: {pct:.0f}% of changed lines covered"
                f" (threshold: {coverage_threshold:.0f}%)"
            )
        else:
            parts.append(f"- {src}: no coverage data available")
    parts.append("")

    parts.append("## Source File Changes (files needing AI review)")
    parts.append("")
    for src in files_for_ai:
        parts.append(f"### {src}")
        diff = source_diffs.get(src, "")
        parts.append(f"```diff\n{_sanitize_diff(diff, max_chars=max_diff_chars)}\n```")
        parts.append("")

    if test_diffs:
        parts.append("## Test File Changes (relevant to files above)")
        parts.append("")
        for test_file, diff in test_diffs.items():
            sources = matched_test_to_sources.get(test_file)
            annotation = f"matched to {', '.join(sources)}" if sources else "candidate"
            parts.append(f"### {test_file} (modified, {annotation})")
            parts.append(f"```diff\n{_sanitize_diff(diff, max_chars=max_diff_chars)}\n```")
            parts.append("")

    return "\n".join(parts)


_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "test_adequacy.txt"


class _AiFileResponse(TypedDict):
    file: str
    verdict: str
    reason: str


class _AiResponse(TypedDict):
    verdict: str
    confidence: float
    files: list[_AiFileResponse]


_VERDICT_SCHEMA: dict[str, object] = {
    # strict mode + additionalProperties: False forces the model to return
    # exactly the fields we expect — no extra keys, no missing keys.
    # This prevents schema drift from silently corrupting _parse_ai_response().
    "type": "object",
    "required": ["verdict", "confidence", "files"],
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["pass", "fail", "warning"],
        },
        "confidence": {"type": "number"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["file", "verdict", "reason"],
                "additionalProperties": False,
                "properties": {
                    "file": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["pass", "fail", "warning"],
                    },
                    "reason": {"type": "string"},
                },
            },
        },
    },
}


_INJECTION_LINE_RE = re.compile(
    # Matches common prompt-injection patterns that could appear in diff lines.
    # Diffs include arbitrary user-authored code, so we must redact lines that
    # look like system instructions before embedding them in the AI prompt.
    r"^(?:[+\-\s]*)?(?:SYSTEM:|INSTRUCTION:|IGNORE PREVIOUS|You are\b)",
    re.IGNORECASE,
)


def _sanitize_diff(diff: str, max_chars: int = 10_000) -> str:
    """Truncate a diff and redact prompt-injection attempts.

    Diffs contain arbitrary user code that could include lines crafted to
    hijack the AI's instructions (e.g. "SYSTEM: ignore previous instructions").
    Matching lines are replaced with "[REDACTED]" before the diff is embedded
    in the prompt.
    """
    sanitized_lines = [
        "[REDACTED]" if _INJECTION_LINE_RE.match(line) else line
        for line in diff.splitlines()
    ]
    sanitized = "\n".join(sanitized_lines)
    if len(sanitized) > max_chars:
        return sanitized[:max_chars] + "...[truncated]"
    return sanitized


# ---------------------------------------------------------------------------
# Smart batching & model fallback
# ---------------------------------------------------------------------------

# Default model chain: try gpt-4.1-mini first, fall back to gpt-4.1-nano on 403.
# When the user explicitly sets ai-model to something other than the chain head,
# a single attempt is made with that model (no fallback).
_DEFAULT_FALLBACK_CHAIN: tuple[str, ...] = (
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
)

_CHARS_PER_TOKEN = 3
_INPUT_TOKEN_LIMIT = 8000
_SYSTEM_OVERHEAD_TOKENS = 800   # system prompt + JSON schema overhead
_SAFETY_FACTOR = 0.85
_USER_PROMPT_TOKEN_BUDGET = int(
    (_INPUT_TOKEN_LIMIT - _SYSTEM_OVERHEAD_TOKENS) * _SAFETY_FACTOR
)  # = 6120 tokens
_FILE_ENTRY_OVERHEAD_TOKENS = 25  # markdown headers, code fences per file
_BATCH_OVERHEAD_TOKENS = 30       # section headers per batch prompt
_RETRY_MAX_DIFF_CHARS = 3000      # tighter truncation on 413 retry


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~3 chars per token, minimum 1.

    Code diffs tokenize at ~3-3.5 characters per token with GPT-family
    tokenizers (symbols and punctuation inflate token count vs. prose).
    The _SAFETY_FACTOR on the budget provides additional headroom.
    """
    return len(text) // _CHARS_PER_TOKEN + 1


def _estimate_file_cost(
    src: str,
    source_diffs: dict[str, str],
    test_diffs: dict[str, str],
    matched_tests: dict[str, str | None],
    max_diff_chars: int = 10_000,
) -> int:
    """Estimate token cost of including one source file in a batch.

    Accounts for: coverage summary line, source diff, matched test diff.

    Candidate test diffs (not matched to any source) are NOT included here
    because they appear in every batch regardless — their cost is accounted
    for once in _batch_files() as base_overhead.
    """
    cost = _FILE_ENTRY_OVERHEAD_TOKENS
    diff = source_diffs.get(src, "")
    cost += _estimate_tokens(_sanitize_diff(diff, max_chars=max_diff_chars))
    matched = matched_tests.get(src)
    if matched and matched in test_diffs:
        cost += _FILE_ENTRY_OVERHEAD_TOKENS
        cost += _estimate_tokens(
            _sanitize_diff(test_diffs[matched], max_chars=max_diff_chars)
        )
    return cost


def _filter_test_diffs_for_batch(
    batch_files: list[str],
    test_diffs: dict[str, str],
    matched_tests: dict[str, str | None],
) -> dict[str, str]:
    """Filter test diffs to those relevant to a batch.

    Three categories of test files:
      - Batch-matched: paired with a source file in this batch → included.
      - Candidate: not matched to any source file at all → included in every
        batch because the AI needs them to judge ambiguous files.
      - Out-of-batch: matched to a source file NOT in this batch → excluded
        to avoid confusing the AI with irrelevant test context.

    Includes:
      - Test files matched to source files *in this batch*.
      - Candidate test files (not matched to any source file at all).
    Excludes:
      - Test files matched to source files *outside* this batch.
    """
    relevant: set[str] = set()
    for src in batch_files:
        matched = matched_tests.get(src)
        if matched and matched in test_diffs:
            relevant.add(matched)
    # Collect all test files that are matched to any source (across all batches).
    batch_matched = {t for src in batch_files
                     for t in [matched_tests.get(src)] if t is not None}
    for test_file in test_diffs:
        if test_file not in batch_matched:
            # Not matched to any source in this batch — treat as candidate.
            relevant.add(test_file)
    return {t: test_diffs[t] for t in sorted(relevant)}


def _batch_files(
    files_for_ai: list[str],
    source_diffs: dict[str, str],
    test_diffs: dict[str, str],
    matched_tests: dict[str, str | None],
    token_budget: int = _USER_PROMPT_TOKEN_BUDGET,
) -> list[list[str]]:
    """Greedily pack files into batches that fit within the token budget.

    Each batch becomes one API call.  Files are appended to the current batch
    until the next file would exceed the budget, then a new batch starts.
    A single file that exceeds the budget gets its own batch — the 413 retry
    path will handle it with tighter truncation.

    base_overhead accounts for candidate test diffs (unmatched to any source)
    that appear in every batch, so they are charged once per batch rather than
    per file.  Greedy packing is used because optimal bin-packing is NP-hard
    and the number of files per PR is small in practice.
    """
    if not files_for_ai:
        return []

    # Candidate tests go in every batch — charge their token cost as a fixed
    # overhead so it doesn't get double-counted in per-file estimates.
    all_matched = {t for t in matched_tests.values() if t is not None}
    candidate_tokens = sum(
        _estimate_tokens(_sanitize_diff(diff))
        for t, diff in test_diffs.items()
        if t not in all_matched
    )
    base_overhead = _BATCH_OVERHEAD_TOKENS + candidate_tokens

    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_tokens = base_overhead

    for src in files_for_ai:
        cost = _estimate_file_cost(src, source_diffs, test_diffs, matched_tests)
        if current_batch and (current_tokens + cost) > token_budget:
            batches.append(current_batch)
            current_batch = []
            current_tokens = base_overhead
        current_batch.append(src)
        current_tokens += cost

    if current_batch:
        batches.append(current_batch)
    return batches


_SIZE_ERROR_PATTERNS = ("too large", "context length")


def _is_retryable_size_error(exc: Exception) -> bool:
    """True when the API rejected the request for being too large (413/400)."""
    if isinstance(exc, APIStatusError) and exc.status_code in (413, 400):
        msg = str(exc).lower()
        return any(p in msg for p in _SIZE_ERROR_PATTERNS)
    return False


def _validate_batch_verdicts(
    verdicts: list[FileVerdict],
    batch_files: list[str],
) -> list[FileVerdict] | None:
    """Filter AI verdicts against the batch whitelist.

    Returns filtered list if every batch file has a verdict.
    Returns None if any batch file is missing (incomplete response).
    """
    batch_set = set(batch_files)
    kept = [fv for fv in verdicts if fv.file in batch_set]
    covered = {fv.file for fv in kept}
    if not batch_set.issubset(covered):
        return None
    return kept


def _is_model_forbidden(exc: Exception) -> bool:
    """True when the model returned 403 (not enabled / permission denied)."""
    if isinstance(exc, APIStatusError):
        return exc.status_code == 403
    return False


def _resolve_models(configured_model: str) -> list[str]:
    """Return the model fallback chain.

    If the configured model is the head of the default chain, the full chain
    is used.  Otherwise a single-element list is returned (no fallback).
    """
    if configured_model == _DEFAULT_FALLBACK_CHAIN[0]:
        return list(_DEFAULT_FALLBACK_CHAIN)
    return [configured_model]


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text().strip()


def _call_ai_for_batch(
    batch_files: list[str],
    source_diffs: dict[str, str],
    test_diffs: dict[str, str],
    coverage_details: dict[str, float] | None,
    coverage_threshold: float,
    matched_tests: dict[str, str | None],
    model: str,
    system_prompt: str,
    token: str,
) -> tuple[str | None, Exception | None]:
    """Call the AI for a single batch with one model.

    On 413 (body too large), retries once with tighter diff truncation.
    Returns ``(raw_response, None)`` on success or ``(None, exception)``
    on failure.
    """
    batch_test_diffs = _filter_test_diffs_for_batch(
        batch_files, test_diffs, matched_tests,
    )
    user_prompt = _build_ai_prompt(
        batch_files, source_diffs, batch_test_diffs,
        coverage_details, coverage_threshold, matched_tests,
    )
    try:
        raw = _call_github_models(model, system_prompt, user_prompt, token)
        return raw, None
    except Exception as exc:
        if _is_retryable_size_error(exc):
            user_prompt = _build_ai_prompt(
                batch_files, source_diffs, batch_test_diffs,
                coverage_details, coverage_threshold, matched_tests,
                max_diff_chars=_RETRY_MAX_DIFF_CHARS,
            )
            try:
                raw = _call_github_models(
                    model, system_prompt, user_prompt, token,
                )
                return raw, None
            except Exception as retry_exc:
                return None, retry_exc
        return None, exc


def _call_github_models(
    model: str,
    system_prompt: str,
    user_prompt: str,
    token: str,
) -> str:
    """Call GitHub Models API and return the raw response text."""
    client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=token,
    )
    messages = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=user_prompt),
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=2048,
        response_format=ResponseFormatJSONSchema(
            type="json_schema",
            json_schema=JSONSchema(
                name="test_verdict",
                strict=True,
                schema=_VERDICT_SCHEMA,
            ),
        ),
    )
    if not response.choices:
        return ""
    return response.choices[0].message.content or ""


def _parse_ai_response(
    raw: str,
) -> tuple[Verdict, float, list[FileVerdict]]:
    """Parse the AI's JSON response into structured data.

    Returns (verdict, confidence, file_verdicts).
    On parse failure, returns (SKIP, 0.0, []).
    """
    try:
        data = cast(_AiResponse, json.loads(raw))
    except json.JSONDecodeError:
        return Verdict.SKIP, 0.0, []

    try:
        verdict_map = {"pass": Verdict.PASS, "fail": Verdict.FAIL, "warning": Verdict.WARNING}
        verdict = verdict_map.get(data["verdict"], Verdict.SKIP)
        confidence = float(data["confidence"])

        file_verdicts: list[FileVerdict] = []
        for f in data["files"]:
            fv_verdict = verdict_map.get(f["verdict"], Verdict.SKIP)
            file_verdicts.append(
                FileVerdict(
                    file=f["file"],
                    verdict=fv_verdict,
                    reason=f["reason"],
                    layer="layer3",
                )
            )

        return verdict, confidence, file_verdicts
    except (KeyError, TypeError, ValueError, IndexError):
        return Verdict.SKIP, 0.0, []


def run_layer3(
    source_diffs: dict[str, str],
    deleted_files: set[str],
    test_diffs: dict[str, str],
    l2_matched_tests: dict[str, str | None],
    coverage_details: dict[str, float] | None,
    coverage_threshold: float,
    model: str,
    token: str,
    confidence_threshold: float,
) -> LayerResult:
    """Run the full Layer 3 evaluation pipeline.

    Three phases:
      1. Shortcut phase: each source file is evaluated against Gates 1-8.
         Files resolved here never reach the AI.
      2. AI phase: remaining files are batched and sent to the model.
         current_model_idx persists across batches so a 403 on batch N
         advances the model for all subsequent batches (not just that one).
         ai_failed_files collects files whose batch failed; they receive
         SKIP verdicts and fall back to L1+L2.
      3. Result assembly: shortcut verdicts and AI verdicts are merged.
         Low-confidence FAIL verdicts are downgraded to WARNING.
         The details message varies by outcome (all shortcuts / mixed / error).
    """
    if not source_diffs:
        return LayerResult(
            layer="layer3",
            verdict=Verdict.PASS,
            details="No source files to evaluate.",
            file_verdicts=[],
            short_circuit=False,
        )

    changed_test_files = list(test_diffs.keys())
    per_file_verdicts: dict[str, Verdict] = {}
    files_for_ai: list[str] = []
    shortcut_reasons: dict[str, str] = {}

    for source_file, diff in source_diffs.items():
        is_deleted = source_file in deleted_files
        relevance = compute_test_relevance(
            source_file,
            changed_test_files,
            l2_matched_tests.get(source_file),
            test_diffs,
        )
        verdict = evaluate_file_shortcut(
            source_file, diff, is_deleted,
            coverage_details, coverage_threshold, relevance,
        )
        if verdict is not None:
            per_file_verdicts[source_file] = verdict
            if is_deleted:
                reason = "shortcut → deleted file"
            elif is_trivial_diff(diff):
                reason = "shortcut → trivial change (whitespace/comments only)"
            elif verdict == Verdict.PASS:
                cov = coverage_details[source_file] if coverage_details else 0.0
                reason = f"shortcut → coverage {cov:.0f}% ≥ {coverage_threshold:.0f}%"
            elif verdict == Verdict.FAIL and relevance == Relevance.NO:
                reason = "shortcut → no relevant tests in PR and no/low coverage"
            elif verdict == Verdict.FAIL:
                cov = coverage_details[source_file] if coverage_details and source_file in coverage_details else 0.0
                reason = f"shortcut → coverage {cov:.0f}% < {coverage_threshold:.0f}%, relevant tests exist but insufficient"
            else:
                reason = f"shortcut → {verdict.value}"
            shortcut_reasons[source_file] = reason
        else:
            files_for_ai.append(source_file)

    execution_status = "OK"
    ai_file_verdicts: list[FileVerdict] = []
    ai_failed_files: list[str] = []
    error_message = ""
    batch_count = 0

    if files_for_ai:
        try:
            system_prompt = _load_system_prompt()
        except (FileNotFoundError, OSError):
            system_prompt = None

        if system_prompt is None:
            ai_failed_files.extend(files_for_ai)
            execution_status = "ERROR"
            error_message = "System prompt file missing or unreadable"
        else:
            models = _resolve_models(model)
            batches = _batch_files(
                files_for_ai, source_diffs, test_diffs, l2_matched_tests,
            )
            batch_count = len(batches)

            # Persists across batches: a 403 on one batch advances the model
            # for all subsequent batches, not just the failing one.
            current_model_idx = 0

            for batch in batches:
                if current_model_idx >= len(models):
                    # All models exhausted — skip remaining batches.
                    ai_failed_files.extend(batch)
                    execution_status = "ERROR"
                    continue

                success = False
                exc: Exception | None = None
                while current_model_idx < len(models) and not success:
                    raw, exc = _call_ai_for_batch(
                        batch, source_diffs, test_diffs,
                        coverage_details, coverage_threshold, l2_matched_tests,
                        models[current_model_idx], system_prompt, token,
                    )
                    if raw is not None:
                        _, ai_confidence, batch_verdicts = _parse_ai_response(raw)
                        validated = _validate_batch_verdicts(batch_verdicts, batch)
                        if validated is None:
                            # None means the AI omitted files — treat as failure
                            # rather than silently accepting an incomplete response.
                            exc = RuntimeError(
                                f"AI response missing files for batch: {batch}"
                            )
                            break
                        ai_file_verdicts.extend(validated)
                        for fv in validated:
                            fv_verdict = fv.verdict
                            if (
                                ai_confidence < confidence_threshold
                                and fv_verdict == Verdict.FAIL
                            ):
                                # Low-confidence FAIL → WARNING (non-blocking).
                                fv_verdict = Verdict.WARNING
                            per_file_verdicts[fv.file] = fv_verdict
                        success = True
                    elif exc and _is_model_forbidden(exc):
                        # 403: this model is not enabled — try the next one.
                        current_model_idx += 1
                    else:
                        break

                if not success:
                    execution_status = "ERROR"
                    error_message = str(exc) if exc else "Unknown AI error"
                    print(f"::warning::Layer 3 AI call failed: {error_message}")
                    ai_failed_files.extend(
                        f for f in batch if f not in per_file_verdicts
                    )

    l3 = Layer3Result(per_file_verdicts, execution_status)

    all_file_verdicts: list[FileVerdict] = []
    for src, v in per_file_verdicts.items():
        reason = shortcut_reasons.get(src, "")
        if not reason:
            matched_fv = [fv for fv in ai_file_verdicts if fv.file == src]
            reason = matched_fv[0].reason if matched_fv else "AI judgment"
        all_file_verdicts.append(
            FileVerdict(file=src, verdict=v, reason=reason, layer="layer3")
        )

    for f in ai_failed_files:
        all_file_verdicts.append(
            FileVerdict(
                file=f,
                verdict=Verdict.SKIP,
                reason="AI analysis unavailable — deferred to Layer 2",
                layer="layer3",
            )
        )

    if execution_status == "ERROR":
        error_suffix = f" ({error_message})" if error_message else ""
        if per_file_verdicts:
            details = (
                f"AI analysis failed{error_suffix} — shortcuts resolved;"
                " remaining files deferred to L1+L2."
            )
        else:
            details = f"AI analysis failed{error_suffix} — falling back to L1+L2."
    elif not files_for_ai:
        details = "All files resolved by deterministic shortcuts."
    else:
        shortcut_count = len(source_diffs) - len(files_for_ai)
        batch_note = f" ({batch_count} batch)" if batch_count == 1 else f" ({batch_count} batches)"
        details = (
            f"Evaluated {len(source_diffs)} files:"
            f" {len(files_for_ai)} via AI{batch_note},"
            f" {shortcut_count} via shortcuts."
        )

    return LayerResult(
        layer="layer3",
        verdict=l3.verdict,
        details=details,
        file_verdicts=all_file_verdicts,
        short_circuit=False,
    )
