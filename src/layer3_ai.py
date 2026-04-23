"""Layer 3: Per-file evaluator (deterministic shortcuts + AI fallthrough).

Evaluates each source file through an 8-gate shortcut truth table using
coverage data from L1 and test-match data from L2. Files that cannot be
resolved deterministically fall through to the GitHub Models AI for a
structured JSON verdict with confidence-based downgrade.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import TypedDict, cast

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema

from src.models import FileVerdict, LayerResult, Verdict


class Relevance(Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


def compute_test_relevance(
    source_file: str,
    changed_test_files: list[str],
    l2_matched_test: str | None,
    test_diffs: dict[str, str],
) -> Relevance:
    if not changed_test_files:
        return Relevance.NO

    source_stem = PurePosixPath(source_file).stem.lower()

    for test_file in changed_test_files:
        if l2_matched_test is not None and test_file == l2_matched_test:
            return Relevance.YES
        if source_stem in PurePosixPath(test_file).stem.lower():
            return Relevance.YES
        diff_text = test_diffs.get(test_file, "")
        if source_stem in diff_text.lower():
            return Relevance.YES

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
    for line in diff.splitlines():
        if not (line.startswith("+") or line.startswith("-")):
            continue
        content = line[1:].strip()
        if not content:
            continue
        if _IMPORT_RE.search(content):
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
    if is_deleted:
        return Verdict.SKIP

    if is_trivial_diff(diff):
        return Verdict.SKIP

    has_coverage = coverage_details is not None and source_file in coverage_details
    coverage_ok = (
        has_coverage
        and coverage_details is not None
        and coverage_details[source_file] >= coverage_threshold
    )

    if coverage_ok:
        return Verdict.PASS

    if test_relevance == Relevance.NO:
        return Verdict.FAIL

    if has_coverage and test_relevance == Relevance.YES:
        return Verdict.FAIL

    return None


@dataclass
class Layer3Result:
    per_file_verdicts: dict[str, Verdict]
    execution_status: str

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
) -> str:
    if not files_for_ai:
        return ""

    matched_test_to_sources: dict[str, list[str]] = {}
    for src, test in matched_tests.items():
        if test is not None:
            matched_test_to_sources.setdefault(test, []).append(src)

    parts: list[str] = []

    parts.append("## Coverage Summary")
    parts.append("Per-file changed-line coverage:")
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
        parts.append(f"```diff\n{_sanitize_diff(diff)}\n```")
        parts.append("")

    if test_diffs:
        parts.append("## Test File Changes (relevant to files above)")
        parts.append("")
        for test_file, diff in test_diffs.items():
            sources = matched_test_to_sources.get(test_file)
            annotation = f"matched to {', '.join(sources)}" if sources else "candidate"
            parts.append(f"### {test_file} (modified, {annotation})")
            parts.append(f"```diff\n{_sanitize_diff(diff)}\n```")
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
    r"^(?:[+\-\s]*)?(?:SYSTEM:|INSTRUCTION:|IGNORE PREVIOUS|You are\b)",
    re.IGNORECASE,
)


def _sanitize_diff(diff: str, max_chars: int = 10_000) -> str:
    sanitized_lines = [
        "[REDACTED]" if _INJECTION_LINE_RE.match(line) else line
        for line in diff.splitlines()
    ]
    sanitized = "\n".join(sanitized_lines)
    if len(sanitized) > max_chars:
        return sanitized[:max_chars] + "...[truncated]"
    return sanitized


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text().strip()



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
            shortcut_reasons[source_file] = f"shortcut → {verdict.value}"
        else:
            files_for_ai.append(source_file)

    execution_status = "OK"
    ai_file_verdicts: list[FileVerdict] = []
    ai_failed_files: list[str] = []
    error_message = ""

    if files_for_ai:
        try:
            system_prompt = _load_system_prompt()
            user_prompt = _build_ai_prompt(
                files_for_ai, source_diffs, test_diffs,
                coverage_details, coverage_threshold, l2_matched_tests,
            )
            raw_response = _call_github_models(model, system_prompt, user_prompt, token)
            _, ai_confidence, ai_file_verdicts = _parse_ai_response(raw_response)

            for fv in ai_file_verdicts:
                fv_verdict = fv.verdict
                if ai_confidence < confidence_threshold and fv_verdict == Verdict.FAIL:
                    fv_verdict = Verdict.WARNING
                per_file_verdicts[fv.file] = fv_verdict

        except Exception as exc:
            execution_status = "ERROR"
            error_message = str(exc)
            print(f"::warning::Layer 3 AI call failed: {exc}")
            ai_failed_files = [f for f in files_for_ai if f not in per_file_verdicts]

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
        details = (
            f"Evaluated {len(source_diffs)} files:"
            f" {len(files_for_ai)} via AI, {shortcut_count} via shortcuts."
        )

    return LayerResult(
        layer="layer3",
        verdict=l3.verdict,
        details=details,
        file_verdicts=all_file_verdicts,
        short_circuit=False,
    )
