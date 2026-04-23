#!/usr/bin/env python3
"""Local runner: replays the test-guard pipeline against a real PR.

Usage:
    python local_pr_test.py                          # defaults to matecat/MateCat#4515
    python local_pr_test.py owner/repo 123           # custom repo and PR number
    GITHUB_TOKEN=ghp_xxx python local_pr_test.py     # explicit token

Token resolution: GITHUB_TOKEN env var → `gh auth token` CLI fallback.
"""
from __future__ import annotations

import subprocess
import sys

from src.config import Config, _DEFAULT_EXCLUDE, _DEFAULT_TEST_PATTERNS
from src.github_api import create_session
from src.github_client import format_report
from src.layer1_coverage import run_layer1
from src.layer2_heuristic import _is_excluded, _is_test_file, _matches_source_pattern, run_layer2
from src.layer3_ai import run_layer3
from src.main import _get_pr_context
from src.models import Report

# ── defaults ────────────────────────────────────────────────────────────
DEFAULT_REPO = "matecat/MateCat"
DEFAULT_PR = 4515


def _resolve_token() -> str:
    import os
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    result = subprocess.run(
        ["gh", "auth", "token"], capture_output=True, text=True, check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    print("ERROR: No GITHUB_TOKEN and `gh auth token` failed.", file=sys.stderr)
    sys.exit(1)


def run_local(repo: str, pr_number: int) -> None:
    token = _resolve_token()

    exclude_patterns = [p.strip() for p in _DEFAULT_EXCLUDE.split(",") if p.strip()]
    config = Config(
        github_token=token,
        repo=repo,
        pr_number=pr_number,
        event_name="pull_request",
        coverage_files=[],
        coverage_threshold=80,
        test_patterns=_DEFAULT_TEST_PATTERNS,
        exclude_patterns=exclude_patterns,
        ai_enabled=True,
        ai_model="openai/gpt-5-mini",
        ai_confidence_threshold=0.7,
    )

    session = create_session(token)

    print(f"── Fetching PR context: {repo}#{pr_number} ──")
    changed_files, all_repo_files, head_sha, file_diffs, deleted_files = _get_pr_context(
        config, session,
    )
    print(f"   changed files : {len(changed_files)}")
    print(f"   deleted files : {len(deleted_files)}")
    print(f"   head SHA      : {head_sha[:12]}")
    for f in changed_files:
        tag = " [DEL]" if f in deleted_files else ""
        print(f"     • {f}{tag}")
    print()

    report = Report()

    # ── Layer 1 ──
    print("── Layer 1: Coverage ──")
    l1 = run_layer1(config.coverage_files, config.coverage_threshold, changed_files)
    report.layers.append(l1)
    print(f"   verdict: {l1.verdict.value}  |  short_circuit: {l1.short_circuit}")
    if l1.short_circuit:
        print("\n" + format_report(report))
        return

    # ── Layer 2 ──
    print("\n── Layer 2: Heuristic ──")
    l2 = run_layer2(changed_files, all_repo_files, config.test_patterns, config.exclude_patterns)
    report.layers.append(l2)
    print(f"   verdict: {l2.verdict.value}  |  short_circuit: {l2.short_circuit}")
    for fv in l2.file_verdicts:
        print(f"     {fv.verdict.value:7s}  {fv.file}  →  {fv.reason}")

    if config.ai_enabled:
        l2.short_circuit = False

    # ── Layer 3 ──
    print("\n── Layer 3: AI Per-File ──")
    source_diffs: dict[str, str] = {}
    test_diffs: dict[str, str] = {}
    for filepath, diff in file_diffs.items():
        if _is_excluded(filepath, config.exclude_patterns):
            continue
        if _is_test_file(filepath, config.test_patterns):
            test_diffs[filepath] = diff
        elif _matches_source_pattern(filepath, config.test_patterns):
            source_diffs[filepath] = diff

    print(f"   source files for L3: {list(source_diffs.keys())}")
    print(f"   test files for L3 : {list(test_diffs.keys())}")

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
    print(f"   verdict: {l3.verdict.value}  |  status: {l3.details}")
    for fv in l3.file_verdicts:
        print(f"     {fv.verdict.value:7s}  {fv.file}  →  {fv.reason}")

    # ── Final report ──
    print(f"\n── Overall verdict: {report.overall_verdict.value} ──")
    print()
    print(format_report(report))


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
    pr = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PR
    run_local(repo, pr)
