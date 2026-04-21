"""Layer 3: GPT-5-mini AI judgment via GitHub Models API.

Only invoked for files that Layer 2 flagged as FAIL or WARNING.
Sends focused diffs + test content and gets structured JSON verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

from src.models import FileVerdict, LayerResult, Verdict

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "test_adequacy.txt"
_GITHUB_MODELS_URL = "https://models.github.ai/inference/chat/completions"


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text().strip()


def _build_prompt(
    file_diffs: dict[str, str],
    test_contents: dict[str, str],
) -> str:
    """Build the user prompt with diff context and test files."""
    parts: list[str] = []

    for filepath, diff in file_diffs.items():
        parts.append(f"## Source file: {filepath}")
        parts.append(f"```diff\n{diff}\n```")

        # Find matching test content
        matching_tests = [
            (tf, tc) for tf, tc in test_contents.items()
            if filepath.split("/")[-1].replace(".py", "") in tf
            or filepath.split("/")[-1].replace(".php", "") in tf
            or filepath.split("/")[-1].replace(".ts", "") in tf
            or filepath.split("/")[-1].replace(".js", "") in tf
            or filepath.split("/")[-1].replace(".go", "") in tf
            or filepath.split("/")[-1].replace(".java", "") in tf
        ]

        if matching_tests:
            for test_file, test_code in matching_tests:
                parts.append(f"### Test file: {test_file}")
                parts.append(f"```\n{test_code}\n```")
        else:
            parts.append("### No test file found for this source file.")

        parts.append("")

    return "\n".join(parts)


def _call_github_models(
    model: str,
    system_prompt: str,
    user_prompt: str,
    token: str,
) -> str:
    """Call GitHub Models API and return the raw response text."""
    response = requests.post(
        _GITHUB_MODELS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 2048,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _parse_ai_response(
    raw: str,
) -> tuple[Verdict, float, list[FileVerdict]]:
    """Parse the AI's JSON response into structured data.

    Returns (verdict, confidence, file_verdicts).
    On parse failure, returns (SKIP, 0.0, []).
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Verdict.SKIP, 0.0, []

    if not all(k in data for k in ("verdict", "confidence", "files")):
        return Verdict.SKIP, 0.0, []

    verdict_map = {"pass": Verdict.PASS, "fail": Verdict.FAIL, "warning": Verdict.WARNING}
    verdict = verdict_map.get(data["verdict"], Verdict.SKIP)
    confidence = float(data.get("confidence", 0.0))

    file_verdicts = []
    for f in data.get("files", []):
        fv_verdict = verdict_map.get(f.get("verdict", ""), Verdict.SKIP)
        file_verdicts.append(FileVerdict(
            file=f.get("file", "unknown"),
            verdict=fv_verdict,
            reason=f.get("reason", "No reason provided"),
            layer="layer3",
        ))

    return verdict, confidence, file_verdicts


def run_layer3(
    file_diffs: dict[str, str],
    test_contents: dict[str, str],
    model: str,
    token: str,
    confidence_threshold: float,
) -> LayerResult:
    """Execute Layer 3 AI analysis.

    Args:
        file_diffs: {filepath: diff_text} for files needing AI review.
        test_contents: {test_filepath: file_content} for related test files.
        model: Model identifier (e.g., "openai/gpt-5-mini").
        token: GitHub token for Models API authentication.
        confidence_threshold: Minimum confidence to enforce verdict (0.0-1.0).

    Returns:
        LayerResult with AI verdict and per-file analysis.
    """
    if not file_diffs:
        return LayerResult(
            layer="layer3",
            verdict=Verdict.PASS,
            details="No files required AI review.",
            file_verdicts=[],
            short_circuit=False,
        )

    try:
        system_prompt = _load_system_prompt()
        user_prompt = _build_prompt(file_diffs, test_contents)
        raw_response = _call_github_models(model, system_prompt, user_prompt, token)
        verdict, confidence, file_verdicts = _parse_ai_response(raw_response)
    except Exception as exc:
        return LayerResult(
            layer="layer3",
            verdict=Verdict.SKIP,
            details=f"AI analysis failed: {exc}",
            file_verdicts=[],
            short_circuit=False,
        )

    if verdict == Verdict.SKIP:
        return LayerResult(
            layer="layer3",
            verdict=Verdict.SKIP,
            details="AI returned unparseable response — skipping.",
            file_verdicts=file_verdicts,
            short_circuit=False,
        )

    # If confidence is below threshold, downgrade FAIL to WARNING
    if confidence < confidence_threshold and verdict == Verdict.FAIL:
        verdict = Verdict.WARNING
        details = (
            f"AI verdict: fail → warning (confidence {confidence:.0%} "
            f"below threshold {confidence_threshold:.0%})"
        )
    else:
        details = f"AI verdict: {verdict.value} (confidence: {confidence:.0%})"

    return LayerResult(
        layer="layer3",
        verdict=verdict,
        details=details,
        file_verdicts=file_verdicts,
        short_circuit=False,
    )
