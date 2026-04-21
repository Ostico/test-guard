"""Main orchestrator — runs the 3-layer pipeline."""
from __future__ import annotations

# pyright: reportAny=false

import re
import sys

import requests as http_requests

from src.config import Config, parse_config
from src.github_client import report_to_github
from src.layer1_coverage import run_layer1
from src.layer2_heuristic import run_layer2
from src.layer3_ai import run_layer3
from src.models import Report, Verdict

_GITHUB_API = "https://api.github.com"


def _get_pr_context(
    config: Config,
) -> tuple[list[str], list[str], str, dict[str, str]]:
    """Fetch PR context from GitHub API.

    Returns:
        (changed_files, all_repo_files, head_sha, file_diffs)
    """
    headers = {
        "Authorization": f"Bearer {config.github_token}",
        "Accept": "application/vnd.github+json",
    }

    # Get PR files
    pr_url = f"{_GITHUB_API}/repos/{config.repo}/pulls/{config.pr_number}/files"
    response = http_requests.get(pr_url, headers=headers, params={"per_page": 100}, timeout=30)
    response.raise_for_status()
    pr_files = response.json()

    changed_files = [f["filename"] for f in pr_files]
    file_diffs = {f["filename"]: f.get("patch", "") for f in pr_files if f.get("patch")}

    # Get head SHA
    pr_detail_url = f"{_GITHUB_API}/repos/{config.repo}/pulls/{config.pr_number}"
    pr_detail = http_requests.get(pr_detail_url, headers=headers, timeout=30).json()
    head_sha = pr_detail["head"]["sha"]

    # Get repo file tree (for test-file lookup)
    tree_url = f"{_GITHUB_API}/repos/{config.repo}/git/trees/{head_sha}?recursive=1"
    tree_resp = http_requests.get(tree_url, headers=headers, timeout=30).json()
    all_repo_files = [item["path"] for item in tree_resp.get("tree", []) if item["type"] == "blob"]

    return changed_files, all_repo_files, head_sha, file_diffs


def run_pipeline(config: Config) -> Report:
    """Execute the full 3-layer pipeline.

    Each layer can short-circuit the pipeline with a PASS verdict.
    """
    report = Report()

    # Fetch PR context
    changed_files, all_repo_files, head_sha, file_diffs = _get_pr_context(config)

    # === Layer 1: Coverage Gate ===
    l1 = run_layer1(config.coverage_file, config.coverage_threshold, changed_files)
    report.layers.append(l1)
    if l1.short_circuit:
        report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
        return report

    # === Layer 2: File-Matching Heuristic ===
    l2 = run_layer2(changed_files, all_repo_files, config.test_patterns, config.exclude_patterns)
    report.layers.append(l2)
    if l2.short_circuit:
        report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
        return report

    # === Layer 3: AI Judgment ===
    if not config.ai_enabled:
        report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
        return report

    # Collect files that need AI review (FAIL or WARNING from Layer 2)
    files_for_ai = [
        fv.file for fv in l2.file_verdicts if fv.verdict in (Verdict.FAIL, Verdict.WARNING)
    ]
    ai_diffs = {f: file_diffs.get(f, "") for f in files_for_ai if f in file_diffs}

    # Fetch test file contents for AI context
    test_contents: dict[str, str] = {}
    for fv in l2.file_verdicts:
        if fv.verdict == Verdict.WARNING and "exists" in fv.reason:
            # Extract test file path from reason
            # Reason format: "Test file exists (tests/test_foo.py) but was not modified"
            match = re.search(r"\(([^)]+)\)", fv.reason)
            if match:
                test_path = match.group(1)
                try:
                    content_url = f"{_GITHUB_API}/repos/{config.repo}/contents/{test_path}?ref={head_sha}"
                    resp = http_requests.get(
                        content_url,
                        headers={
                            "Authorization": f"Bearer {config.github_token}",
                            "Accept": "application/vnd.github.raw+json",
                        },
                        timeout=30,
                    )
                    if resp.ok:
                        test_contents[test_path] = resp.text
                except Exception:
                    pass

    l3 = run_layer3(
        ai_diffs,
        test_contents,
        config.ai_model,
        config.github_token,
        config.ai_confidence_threshold,
    )
    report.layers.append(l3)

    report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
    return report


def main() -> None:
    """Entry point for the GitHub Action."""
    config = parse_config()

    if config.event_name != "pull_request":
        print("::notice::test-guard only runs on pull_request events. Skipping.")
        return

    if config.pr_number is None:
        print("::error::Could not determine PR number from GITHUB_REF.")
        sys.exit(1)

    report = run_pipeline(config)

    # Set exit code based on verdict
    if report.overall_verdict == Verdict.FAIL:
        print("::error::Test adequacy check FAILED.")
        sys.exit(1)
    if report.overall_verdict == Verdict.WARNING:
        print("::warning::Test adequacy warnings found (non-blocking).")
    else:
        print("::notice::Test adequacy check passed.")


if __name__ == "__main__":
    main()
