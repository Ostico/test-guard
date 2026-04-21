"""GitHub API client for PR comments and commit status checks."""

from __future__ import annotations

import re

import requests

from src.github_api import GITHUB_API_URL, create_session, post_json
from src.models import Report, Verdict

_VERDICT_EMOJI = {
    Verdict.PASS: "✅",
    Verdict.FAIL: "❌",
    Verdict.WARNING: "⚠️",
    Verdict.SKIP: "⏭️",
}

_VERDICT_CONCLUSION = {
    Verdict.PASS: "success",
    Verdict.FAIL: "failure",
    Verdict.WARNING: "neutral",  # Non-blocking in status checks
    Verdict.SKIP: "skipped",
}

_TOKEN_PATTERNS = (
    r"ghp_\w+",
    r"gho_\w+",
    r"github_pat_\w+",
    r"Bearer\s+\S+",
)


def format_report(report: Report) -> str:
    """Format a Report as a Markdown PR comment."""
    emoji = _VERDICT_EMOJI[report.overall_verdict]
    lines = [
        "## 🧪 Test-Guard Report",
        "",
    ]

    for lr in report.layers:
        layer_emoji = _VERDICT_EMOJI[lr.verdict]
        layer_name = lr.layer.replace("layer", "Layer ")
        lines.append(f"### {layer_name}: {layer_emoji} {lr.verdict.value.upper()}")
        lines.append(lr.details)
        lines.append("")

        if lr.file_verdicts:
            lines.append("| File | Verdict | Reason |")
            lines.append("|---|---|---|")
            for fv in lr.file_verdicts:
                fv_emoji = _VERDICT_EMOJI[fv.verdict]
                lines.append(f"| `{fv.file}` | {fv_emoji} {fv.verdict.value} | {fv.reason} |")
            lines.append("")

    lines.append(f"**Result: {emoji} {report.overall_verdict.value.upper()}**")
    return "\n".join(lines)


def _redact_response_text(text: str, max_len: int = 200) -> str:
    """Redact sensitive tokens and truncate response text for logging."""
    redacted = text[:max_len]
    for pattern in _TOKEN_PATTERNS:
        redacted = re.sub(pattern, "[REDACTED]", redacted)
    return redacted


def post_comment(
    session: requests.Session,
    repo: str,
    pr_number: int,
    body: str,
) -> None:
    """Post or update a comment on a PR."""
    url = f"{GITHUB_API_URL}/repos/{repo}/issues/{pr_number}/comments"
    resp = post_json(session, url, {"body": body})
    if not resp.ok:
        print(
            "::warning::Failed to post PR comment "
            f"({resp.status_code}): {_redact_response_text(resp.text)}"
        )


_CHECK_RUN_NAME = "Test-Guard"


def post_check_run(
    session: requests.Session,
    repo: str,
    sha: str,
    conclusion: str,
    title: str,
    summary: str,
) -> None:
    """Create a completed check run via the Checks API."""
    url = f"{GITHUB_API_URL}/repos/{repo}/check-runs"
    resp = post_json(
        session,
        url,
        {
            "name": _CHECK_RUN_NAME,
            "head_sha": sha,
            "status": "completed",
            "conclusion": conclusion,
            "output": {
                "title": title,
                "summary": summary,
            },
        },
    )
    if not resp.ok:
        print(
            "::warning::Failed to create check run "
            f"({resp.status_code}): {_redact_response_text(resp.text)}"
        )


def report_to_github(
    report: Report,
    token: str,
    repo: str,
    pr_number: int | None,
    sha: str,
) -> None:
    """Post a check run and optionally a PR comment."""
    try:
        session = create_session(token)

        desc_map = {
            Verdict.PASS: "All test adequacy checks passed",
            Verdict.FAIL: "Test adequacy issues found",
            Verdict.WARNING: "Test adequacy warnings (non-blocking)",
            Verdict.SKIP: "Analysis skipped",
        }
        conclusion = _VERDICT_CONCLUSION[report.overall_verdict]
        summary = format_report(report)
        post_check_run(
            session, repo, sha, conclusion, desc_map[report.overall_verdict], summary,
        )

        if pr_number is not None:
            post_comment(session, repo, pr_number, summary)
    except Exception as exc:
        print(f"::warning::GitHub reporting failed: {_redact_response_text(str(exc), max_len=500)}")
