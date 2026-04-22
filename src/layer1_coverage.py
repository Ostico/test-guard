"""Layer 1: Diff-coverage data provider + fast exit.

Runs diff-cover to extract per-file changed-line coverage. When every source
file meets the threshold the pipeline short-circuits (skips L2, L3, and the
AI call entirely). Otherwise the per-file coverage_details are forwarded to
Layer 3 for use in the shortcut truth table and AI prompt.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.models import LayerResult, Verdict

_DIFF_COVER_TIMEOUT = 60


def _compute_diff_coverage(coverage_file: str) -> tuple[float, dict[str, float]]:
    """Run diff-cover and return (aggregate_pct, per_file_pct).

    Returns (-1.0, {}) on any failure.
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
            return -1.0, {}

        data = json.loads(result.stdout)
        total = float(data.get("total_percent_covered", -1.0))

        per_file: dict[str, float] = {}
        src_stats = data.get("src_stats", {})
        for filepath_key, file_stats in src_stats.items():
            pct = file_stats.get("percent_covered")
            if isinstance(pct, (int, float)) and isinstance(filepath_key, str):
                per_file[filepath_key] = float(pct)

        return total, per_file
    except (
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        FileNotFoundError,
        AttributeError,
        TypeError,
    ):
        return -1.0, {}


def run_layer1(
    coverage_file: str | None,
    threshold: int,
    diff_files: list[str],
) -> LayerResult:
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

    total_pct, per_file = _compute_diff_coverage(coverage_file)

    if total_pct < 0:
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details="diff-cover failed to compute coverage — skipping Layer 1.",
            file_verdicts=[],
            short_circuit=False,
        )

    # Per-file short-circuit: PASS only when EVERY changed source file
    # present in src_stats has coverage >= threshold AND no source file
    # is absent from src_stats. Non-source files (tests, docs) are ignored.
    source_files = [f for f in diff_files if f in per_file]
    absent_files = [
        f for f in diff_files if f not in per_file and not _is_non_source(f)
    ]
    all_above = all(per_file.get(f, 0.0) >= threshold for f in source_files)
    passed = bool(source_files) and all_above and not absent_files

    return LayerResult(
        layer="layer1",
        verdict=Verdict.PASS if passed else Verdict.FAIL,
        details=f"Changed lines: {total_pct}% covered (threshold: {threshold}%)",
        file_verdicts=[],
        short_circuit=passed,
        coverage_details=per_file,
    )


def _is_non_source(filepath: str) -> bool:
    lower = filepath.lower()
    if "/test" in lower or lower.startswith("test") or "test_" in lower:
        return True
    non_source_exts = {".md", ".txt", ".yml", ".yaml", ".json", ".toml", ".cfg", ".ini", ".lock"}
    return Path(filepath).suffix.lower() in non_source_exts
