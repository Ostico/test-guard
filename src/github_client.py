"""GitHub API client for PR comments and commit status checks."""

from __future__ import annotations

import requests

from src.models import Report, Verdict

_GITHUB_API = "https://api.github.com"

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


def post_comment(
    token: str,
    repo: str,
    pr_number: int,
    body: str,
) -> None:
    """Post or update a comment on a PR."""
    url = f"{_GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"body": body},
        timeout=30,
    )


def post_status(
    token: str,
    repo: str,
    sha: str,
    state: str,
    description: str,
) -> None:
    """Post a commit status check."""
    url = f"{_GITHUB_API}/repos/{repo}/statuses/{sha}"
    requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={
            "state": state,
            "description": description[:140],
            "context": "test-guard",
        },
        timeout=30,
    )


def report_to_github(
    report: Report,
    token: str,
    repo: str,
    pr_number: int | None,
    sha: str,
) -> None:
    """Post both the PR comment and commit status."""
    # Always post status
    state = _VERDICT_STATE[report.overall_verdict]
    desc_map = {
        Verdict.PASS: "All test adequacy checks passed",
        Verdict.FAIL: "Test adequacy issues found",
        Verdict.WARNING: "Test adequacy warnings (non-blocking)",
        Verdict.SKIP: "Analysis skipped",
    }
    post_status(token, repo, sha, state, desc_map[report.overall_verdict])

    # Post comment only on PRs
    if pr_number:
        body = format_report(report)
        post_comment(token, repo, pr_number, body)
