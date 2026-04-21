"""Main orchestrator — runs the 3-layer pipeline."""

from __future__ import annotations

# pyright: reportUnknownMemberType=false
import sys
import traceback

import requests

from src.config import Config, parse_config
from src.github_api import GITHUB_API_URL, create_session, get_json, get_paginated, get_text
from src.github_client import format_report, report_to_github
from src.layer1_coverage import run_layer1
from src.layer2_heuristic import run_layer2
from src.layer3_ai import run_layer3
from src.models import Report, Verdict


def _get_pr_context(
    config: Config,
    session: requests.Session,
) -> tuple[list[str], list[str], str, dict[str, str]]:
    """Fetch PR context from GitHub API.

    Returns:
        (changed_files, all_repo_files, head_sha, file_diffs)
    """
    # Get PR files
    pr_url = f"{GITHUB_API_URL}/repos/{config.repo}/pulls/{config.pr_number}/files"
    pr_files = get_paginated(session, pr_url)

    changed_files = [f["filename"] for f in pr_files]
    file_diffs = {f["filename"]: f.get("patch", "") for f in pr_files if f.get("patch")}

    # Get head SHA
    pr_detail_url = f"{GITHUB_API_URL}/repos/{config.repo}/pulls/{config.pr_number}"
    pr_detail = get_json(session, pr_detail_url)
    head_sha = pr_detail["head"]["sha"]

    # Get repo file tree (for test-file lookup)
    tree_url = f"{GITHUB_API_URL}/repos/{config.repo}/git/trees/{head_sha}?recursive=1"
    tree_resp = get_json(session, tree_url)
    all_repo_files = [item["path"] for item in tree_resp.get("tree", []) if item["type"] == "blob"]

    return changed_files, all_repo_files, head_sha, file_diffs


def run_pipeline(config: Config) -> Report:
    """Execute the full 3-layer pipeline.

    Each layer can short-circuit the pipeline with a PASS verdict.
    """
    report = Report()
    session = create_session(config.github_token)

    # Fetch PR context
    changed_files, all_repo_files, head_sha, file_diffs = _get_pr_context(config, session)

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
        if fv.matched_test is not None:
            content_url = (
                f"{GITHUB_API_URL}/repos/{config.repo}/contents/{fv.matched_test}?ref={head_sha}"
            )
            text = get_text(session, content_url)
            if text is not None:
                test_contents[fv.matched_test] = text

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
    try:
        config = parse_config()
    except Exception:
        print("::error::Failed to parse configuration.")
        traceback.print_exc()
        sys.exit(1)

    if config.event_name != "pull_request":
        print("::notice::test-guard only runs on pull_request events. Skipping.")
        return

    if config.pr_number is None:
        print("::error::Could not determine PR number from GITHUB_REF.")
        sys.exit(1)

    try:
        report = run_pipeline(config)
    except Exception:
        print("::error::Pipeline failed with an unexpected error.")
        traceback.print_exc()
        sys.exit(1)

    print(format_report(report))

    if report.overall_verdict == Verdict.FAIL:
        print("::error::Test adequacy check FAILED.")
        sys.exit(1)
    if report.overall_verdict == Verdict.WARNING:
        print("::warning::Test adequacy warnings found (non-blocking).")
    else:
        print("::notice::Test adequacy check passed.")


if __name__ == "__main__":
    main()
