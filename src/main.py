"""Main orchestrator — runs the 3-layer pipeline.

L1 and L2 are data providers that feed L3 (the actual evaluator when AI is
enabled). L1 extracts coverage_details, L2 extracts matched_test mappings
and provides file classification helpers. Both retain independent gating
for the AI-disabled fallback path.
"""

from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportPrivateUsage=false
import sys
import traceback

import requests

from src.config import Config, parse_config
from src.github_api import GITHUB_API_URL, create_session, get_json, get_paginated
from src.github_client import format_report, report_to_github
from src.layer1_coverage import run_layer1
from src.layer2_heuristic import _is_excluded, _is_test_file, _matches_source_pattern, run_layer2
from src.layer3_ai import run_layer3
from src.models import Report, Verdict


def _get_pr_context(
    config: Config,
    session: requests.Session,
) -> tuple[list[str], list[str], str, dict[str, str], set[str]]:
    pr_url = f"{GITHUB_API_URL}/repos/{config.repo}/pulls/{config.pr_number}/files"
    pr_files = get_paginated(session, pr_url)

    changed_files = [f["filename"] for f in pr_files]
    file_diffs = {f["filename"]: f.get("patch", "") for f in pr_files}
    deleted_files = {f["filename"] for f in pr_files if f.get("status") == "removed"}

    pr_detail_url = f"{GITHUB_API_URL}/repos/{config.repo}/pulls/{config.pr_number}"
    pr_detail = get_json(session, pr_detail_url)
    head_sha = pr_detail["head"]["sha"]

    tree_url = f"{GITHUB_API_URL}/repos/{config.repo}/git/trees/{head_sha}?recursive=1"
    tree_resp = get_json(session, tree_url)
    all_repo_files = [item["path"] for item in tree_resp.get("tree", []) if item["type"] == "blob"]

    return changed_files, all_repo_files, head_sha, file_diffs, deleted_files


def run_pipeline(config: Config) -> Report:
    report = Report()
    session = create_session(config.github_token)

    changed_files, all_repo_files, head_sha, file_diffs, deleted_files = _get_pr_context(
        config, session,
    )

    # === Layer 1: Coverage Gate ===
    l1 = run_layer1(config.coverage_files, config.coverage_threshold, changed_files)
    report.layers.append(l1)
    if l1.short_circuit:
        report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
        return report

    # === Layer 2: File-Matching Heuristic ===
    l2 = run_layer2(changed_files, all_repo_files, config.test_patterns, config.exclude_patterns)
    report.layers.append(l2)

    if config.ai_enabled:
        l2.short_circuit = False
    elif l2.short_circuit:
        report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
        return report

    # === Layer 3: AI Judgment ===
    if not config.ai_enabled:
        report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
        return report

    source_diffs: dict[str, str] = {}
    test_diffs: dict[str, str] = {}
    for filepath, diff in file_diffs.items():
        if _is_excluded(filepath, config.exclude_patterns):
            continue
        if _is_test_file(filepath, config.test_patterns):
            test_diffs[filepath] = diff
        elif _matches_source_pattern(filepath, config.test_patterns):
            source_diffs[filepath] = diff

    l2_matched_tests: dict[str, str | None] = {
        fv.file: fv.matched_test for fv in l2.file_verdicts
    }

    l3 = run_layer3(
        source_diffs=source_diffs,
        deleted_files=deleted_files,
        test_diffs=test_diffs,
        l2_matched_tests=l2_matched_tests,
        coverage_details=l1.coverage_details,
        coverage_threshold=config.coverage_threshold,
        model=config.ai_model,
        token=config.github_token,
        confidence_threshold=config.ai_confidence_threshold,
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
