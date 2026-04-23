"""Layer 1: Diff-coverage data provider + fast exit.

Runs diff-cover to extract per-file changed-line coverage. When every source
file meets the threshold the pipeline short-circuits (skips L2, L3, and the
AI call entirely). Otherwise the per-file coverage_details are forwarded to
Layer 3 for use in the shortcut truth table and AI prompt.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from src.models import LayerResult, Verdict

_DIFF_COVER_TIMEOUT = 60

_TRACEBACK_EXCEPTION_RE = re.compile(
    r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning))\s*:\s*",
    re.MULTILINE,
)


def _extract_stderr_message(stderr: str) -> str:
    matches = list(_TRACEBACK_EXCEPTION_RE.finditer(stderr))
    if matches:
        return stderr[matches[-1].start():].strip()
    lines = [l.strip() for l in stderr.strip().splitlines() if l.strip()]
    return lines[-1] if lines else stderr.strip()


def _compute_diff_coverage(
    coverage_files: list[str],
) -> tuple[float, dict[str, float], str]:
    """Run diff-cover and return (aggregate_pct, per_file_pct, error_reason).

    Returns (-1.0, {}, reason) on any failure.
    """
    try:
        cmd = [
            "diff-cover",
            *coverage_files,
            "--json-report",
            "/dev/stdout",
            "--quiet",
        ]
        base_ref = os.environ.get("GITHUB_BASE_REF", "").strip()
        if base_ref:
            cmd.append(f"--compare-branch=origin/{base_ref}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_DIFF_COVER_TIMEOUT,
        )
        if result.returncode != 0:
            raw_stderr = result.stderr.strip() if result.stderr else ""
            if raw_stderr:
                print(
                    f"::warning::diff-cover failed"
                    f" (exit {result.returncode}): {raw_stderr}"
                )
            reason = (
                _extract_stderr_message(raw_stderr)
                if raw_stderr
                else f"exit code {result.returncode}"
            )
            return -1.0, {}, reason

        data = json.loads(result.stdout)
        total = float(data.get("total_percent_covered", -1.0))

        per_file: dict[str, float] = {}
        src_stats = data.get("src_stats", {})
        for filepath_key, file_stats in src_stats.items():
            pct = file_stats.get("percent_covered")
            if isinstance(pct, (int, float)) and isinstance(filepath_key, str):
                per_file[filepath_key] = float(pct)

        return total, per_file, ""
    except (
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        FileNotFoundError,
        AttributeError,
        TypeError,
    ) as exc:
        print(f"::warning::diff-cover error: {exc}")
        return -1.0, {}, str(exc)


def run_layer1(
    coverage_files: list[str],
    threshold: int,
    diff_files: list[str],
) -> LayerResult:
    if not coverage_files:
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details="No coverage files provided — skipping Layer 1.",
            file_verdicts=[],
            short_circuit=False,
        )

    valid_files = [f for f in coverage_files if Path(f).exists()]
    if not valid_files:
        missing = ", ".join(coverage_files)
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details=f"No valid coverage files found ({missing}) — skipping Layer 1.",
            file_verdicts=[],
            short_circuit=False,
        )

    total_pct, per_file, error_reason = _compute_diff_coverage(valid_files)

    if total_pct < 0:
        detail = "diff-cover failed to compute coverage"
        if error_reason:
            detail += f": {error_reason}"
        detail += " — skipping Layer 1."
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details=detail,
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

    details = f"Changed lines: {total_pct}% covered (threshold: {threshold}%)"
    if not passed:
        below = [
            f"`{f}` ({per_file[f]:.0f}%)"
            for f in source_files
            if per_file[f] < threshold
        ]
        absent_formatted = [f"`{f}`" for f in absent_files]
        lines: list[str] = []
        if below:
            lines.append(f"**Below threshold:** {', '.join(below)}")
        if absent_formatted:
            lines.append(
                f"**Missing from coverage report:** {', '.join(absent_formatted)}"
            )
        if lines:
            details += "\n\n" + "\n\n".join(lines)

    return LayerResult(
        layer="layer1",
        verdict=Verdict.PASS if passed else Verdict.FAIL,
        details=details,
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
