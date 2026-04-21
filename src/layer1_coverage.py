"""Layer 1: Diff-coverage gate.

Computes test coverage on changed/new lines using diff-cover.
If coverage >= threshold, short-circuit with PASS.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.models import LayerResult, Verdict

_DIFF_COVER_TIMEOUT = 60


def _compute_diff_coverage(coverage_file: str) -> float:
    """Run diff-cover and return the total diff coverage percentage.

    Uses diff-cover CLI which compares coverage report against git diff.
    Returns coverage as a float 0-100.
    """
    try:
        result = subprocess.run(
            [
                "diff-cover",
                coverage_file,
                "--json-report",
                "/dev/stdout",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=_DIFF_COVER_TIMEOUT,
        )
        if result.returncode != 0:
            return -1.0

        data = json.loads(result.stdout)
        total = data.get("total_percent_covered", -1.0)
        return float(total)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return -1.0


def run_layer1(
    coverage_file: str | None,
    threshold: int,
    diff_files: list[str],
) -> LayerResult:
    """Execute Layer 1 analysis.

    Args:
        coverage_file: Path to coverage report (cobertura/lcov), or None.
        threshold: Minimum coverage % to auto-pass (0-100).
        diff_files: List of changed files in the PR.

    Returns:
        LayerResult with verdict and short_circuit flag.
    """
    if not coverage_file:
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details="No coverage file provided — skipping Layer 1.",
            file_verdicts=[],
            short_circuit=False,
        )

    if not Path(coverage_file).exists():
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details=f"Coverage file not found: {coverage_file} — skipping Layer 1.",
            file_verdicts=[],
            short_circuit=False,
        )

    pct = _compute_diff_coverage(coverage_file)

    if pct < 0:
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details="diff-cover failed to compute coverage — skipping Layer 1.",
            file_verdicts=[],
            short_circuit=False,
        )

    passed = pct >= threshold
    return LayerResult(
        layer="layer1",
        verdict=Verdict.PASS if passed else Verdict.FAIL,
        details=f"Changed lines: {pct}% covered (threshold: {threshold}%)",
        file_verdicts=[],
        short_circuit=passed,
    )
