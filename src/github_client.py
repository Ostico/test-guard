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

_VERDICT_STATE = {
    Verdict.PASS: "success",
    Verdict.FAIL: "failure",
    Verdict.WARNING: "success",  # Warnings don't block
    Verdict.SKIP: "success",
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
        "## 🧪 Test Guard Report",
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


def post_status(
    session: requests.Session,
    repo: str,
    sha: str,
    state: str,
    description: str,
) -> None:
    """Post a commit status check."""
    url = f"{GITHUB_API_URL}/repos/{repo}/statuses/{sha}"
    resp = post_json(
        session,
        url,
        {
            "state": state,
            "description": description[:140],
            "context": "test-guard",
        },
    )
    if not resp.ok:
        print(
            "::warning::Failed to post commit status "
            f"({resp.status_code}): {_redact_response_text(resp.text)}"
        )


def report_to_github(
    report: Report,
    token: str,
    repo: str,
    pr_number: int | None,
    sha: str,
) -> None:
    """Post both the PR comment and commit status."""
    try:
        session = create_session(token)

        # Always post status
        state = _VERDICT_STATE[report.overall_verdict]
        desc_map = {
            Verdict.PASS: "All test adequacy checks passed",
            Verdict.FAIL: "Test adequacy issues found",
            Verdict.WARNING: "Test adequacy warnings (non-blocking)",
            Verdict.SKIP: "Analysis skipped",
        }
        post_status(session, repo, sha, state, desc_map[report.overall_verdict])

        # Post comment only on PRs
        if pr_number is not None:
            body = format_report(report)
            post_comment(session, repo, pr_number, body)
    except Exception as exc:
        print(f"::warning::GitHub reporting failed: {_redact_response_text(str(exc), max_len=500)}")
