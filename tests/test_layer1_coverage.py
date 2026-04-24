# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateUsage=false

import os
import subprocess
from unittest.mock import patch

from src.layer1_coverage import (
    _DIFF_COVER_TIMEOUT,
    _compute_diff_coverage,
    _extract_stderr_message,
    run_layer1,
)
from src.models import Verdict

_SAMPLE_SRC_STATS_JSON = (
    '{"total_percent_covered": 85.0, '
    '"src_stats": {'
    '"src/auth.py": {"covered_lines": [1,2,3], "violation_lines": [15], "percent_covered": 92.5}, '
    '"src/billing.py": {"covered_lines": [1], "violation_lines": [5,6,7], "percent_covered": 25.0}'
    "}}"
)


class TestConstants:
    def test_diff_cover_timeout_value(self):
        assert _DIFF_COVER_TIMEOUT == 60

    def test_diff_cover_timeout_type(self):
        assert isinstance(_DIFF_COVER_TIMEOUT, int)


class TestRunLayer1:
    def test_skip_when_no_coverage_file(self):
        result = run_layer1(coverage_files=[], threshold=80, diff_files=[])
        assert result.verdict == Verdict.SKIP
        assert result.short_circuit is False
        assert result.coverage_details is None

    def test_skip_when_coverage_file_missing(self, tmp_path):
        fake_path = str(tmp_path / "nonexistent.xml")
        result = run_layer1(coverage_files=[fake_path], threshold=80, diff_files=["src/foo.py"])
        assert result.verdict == Verdict.SKIP
        assert result.coverage_details is None

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_pass_when_all_files_above_threshold(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        mock_cov.return_value = (92.5, {"src/auth.py": 92.5}, "")
        result = run_layer1(
            coverage_files=[str(cov)],
            threshold=80,
            diff_files=["src/auth.py"],
        )
        assert result.verdict == Verdict.PASS
        assert result.short_circuit is True
        assert result.coverage_details == {"src/auth.py": 92.5}

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_fail_when_any_file_below_threshold(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        per_file = {"src/auth.py": 92.5, "src/billing.py": 25.0}
        mock_cov.return_value = (85.0, per_file, "")
        result = run_layer1(
            coverage_files=[str(cov)],
            threshold=80,
            diff_files=["src/auth.py", "src/billing.py"],
        )
        assert result.verdict == Verdict.FAIL
        assert result.short_circuit is False
        assert result.coverage_details == per_file
        assert any(fv.file == "src/billing.py" and fv.verdict == Verdict.FAIL for fv in result.file_verdicts)
        assert any(fv.file == "src/auth.py" and fv.verdict == Verdict.PASS for fv in result.file_verdicts)

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_no_short_circuit_when_file_absent_from_src_stats(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        mock_cov.return_value = (92.5, {"src/auth.py": 92.5}, "")
        result = run_layer1(
            coverage_files=[str(cov)],
            threshold=80,
            diff_files=["src/auth.py", "src/new_feature.py"],
        )
        assert result.verdict == Verdict.FAIL
        assert result.short_circuit is False
        assert result.coverage_details == {"src/auth.py": 92.5}
        assert any(fv.file == "src/new_feature.py" and fv.verdict == Verdict.FAIL for fv in result.file_verdicts)
        assert any("not in coverage report" in fv.reason for fv in result.file_verdicts if fv.file == "src/new_feature.py")

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_pass_exactly_at_threshold(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        mock_cov.return_value = (80.0, {"src/auth.py": 80.0}, "")
        result = run_layer1(
            coverage_files=[str(cov)],
            threshold=80,
            diff_files=["src/auth.py"],
        )
        assert result.verdict == Verdict.PASS
        assert result.short_circuit is True

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_skip_when_diff_cover_returns_negative(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        mock_cov.return_value = (-1.0, {}, "branch not found")
        result = run_layer1(
            coverage_files=[str(cov)],
            threshold=80,
            diff_files=["src/auth.py"],
        )
        assert result.verdict == Verdict.SKIP
        assert result.short_circuit is False
        assert "diff-cover failed" in result.details
        assert "branch not found" in result.details
        assert result.coverage_details is None

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_coverage_details_populated_on_pass(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        per_file = {"src/a.py": 95.0, "src/b.py": 88.0}
        mock_cov.return_value = (91.5, per_file, "")
        result = run_layer1(
           coverage_files=[str(cov)],
           threshold=80,
           diff_files=["src/a.py", "src/b.py"],
        )
        assert result.coverage_details == per_file

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_non_source_files_ignored_in_per_file_check(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        mock_cov.return_value = (92.0, {"src/auth.py": 92.0}, "")
        result = run_layer1(
           coverage_files=[str(cov)],
           threshold=80,
           diff_files=["src/auth.py", "tests/test_auth.py", "README.md"],
        )
        assert result.verdict == Verdict.PASS
        assert result.short_circuit is True

    def test_empty_diff_files_returns_skip(self, tmp_path):
        """When diff_files is empty, L1 returns SKIP regardless of coverage files."""
        cov = tmp_path / "coverage.xml"
        cov.write_text("<coverage></coverage>")
        result = run_layer1([str(cov)], 80, diff_files=[])
        assert result.verdict == Verdict.SKIP
        assert "No source files" in result.details
        assert result.file_verdicts == []
        assert not result.short_circuit


_REAL_DIFF_COVER_TRACEBACK = (
    "/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/site-packages/"
    "diff_cover/diff_cover_tool.py:315: UserWarning: The --json-report option "
    "is deprecated.\n  warnings.warn(\nTraceback (most recent call last):\n"
    "  File \"/opt/.../git_diff.py\", line 70, in diff_committed\n"
    "    return execute(\ndiff_cover.command_runner.CommandError: fatal: "
    "ambiguous argument 'origin/main...HEAD'\n\nThe above exception was the "
    "direct cause of the following exception:\n\nTraceback (most recent call "
    "last):\n  File \"/opt/.../diff_cover_tool.py\", line 364, in main\n"
    "    percent_covered = generate_coverage_report(\nValueError: \n"
    "Could not find the branch to compare to. Does 'origin/main' exist?\n"
    "the `--compare-branch` argument allows you to set a different branch."
)


class TestExtractStderrMessage:
    def test_extracts_final_exception_from_chained_traceback(self):
        result = _extract_stderr_message(_REAL_DIFF_COVER_TRACEBACK)
        assert result.startswith("ValueError:")
        assert "origin/main" in result
        assert "--compare-branch" in result
        assert "Traceback" not in result

    def test_simple_error_message_returned_as_is(self):
        assert _extract_stderr_message("some error text") == "some error text"

    def test_single_exception_line(self):
        result = _extract_stderr_message("FileNotFoundError: diff-cover not found")
        assert result == "FileNotFoundError: diff-cover not found"

    def test_empty_string_returns_empty(self):
        assert _extract_stderr_message("") == ""

    def test_multiline_non_traceback_returns_last_line(self):
        stderr = "warning: something\nactual error here"
        assert _extract_stderr_message(stderr) == "actual error here"


class TestComputeDiffCoverage:
    @patch("src.layer1_coverage.subprocess.run")
    def test_returns_aggregate_and_per_file(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout=_SAMPLE_SRC_STATS_JSON,
            stderr="",
        )
        total, per_file, error = _compute_diff_coverage(["coverage.xml"])
        assert total == 85.0
        assert per_file == {"src/auth.py": 92.5, "src/billing.py": 25.0}
        assert error == ""

    @patch("src.layer1_coverage.subprocess.run")
    def test_returns_negative_and_empty_when_non_zero(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=1,
            stdout="",
            stderr="error",
        )
        total, per_file, error = _compute_diff_coverage(["coverage.xml"])
        assert total == -1.0
        assert per_file == {}
        assert error == "error"

    @patch("src.layer1_coverage.subprocess.run")
    def test_traceback_stderr_extracts_clean_message(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=1,
            stdout="",
            stderr=_REAL_DIFF_COVER_TRACEBACK,
        )
        total, per_file, error = _compute_diff_coverage(["coverage.xml"])
        assert total == -1.0
        assert per_file == {}
        assert error.startswith("ValueError:")
        assert "origin/main" in error
        assert "Traceback" not in error

    @patch("src.layer1_coverage.subprocess.run")
    def test_returns_negative_and_empty_on_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="diff-cover", timeout=60)
        total, per_file, error = _compute_diff_coverage(["coverage.xml"])
        assert total == -1.0
        assert per_file == {}
        assert "timed out" in error.lower() or "timeout" in error.lower()

    @patch("src.layer1_coverage.subprocess.run")
    def test_returns_negative_and_empty_on_invalid_json(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout="not-json",
            stderr="",
        )
        total, per_file, error = _compute_diff_coverage(["coverage.xml"])
        assert total == -1.0
        assert per_file == {}
        assert error  # non-empty error reason

    @patch("src.layer1_coverage.subprocess.run")
    def test_returns_negative_and_empty_when_not_installed(self, mock_run):
        mock_run.side_effect = FileNotFoundError("diff-cover")
        total, per_file, error = _compute_diff_coverage(["coverage.xml"])
        assert total == -1.0
        assert per_file == {}
        assert "diff-cover" in error

    @patch("src.layer1_coverage.subprocess.run")
    def test_handles_missing_src_stats_key(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout='{"total_percent_covered": 50.0}',
            stderr="",
        )
        total, per_file, error = _compute_diff_coverage(["coverage.xml"])
        assert total == 50.0
        assert per_file == {}
        assert error == ""

    @patch("src.layer1_coverage.subprocess.run")
    def test_handles_empty_src_stats(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout='{"total_percent_covered": 0.0, "src_stats": {}}',
            stderr="",
        )
        total, per_file, error = _compute_diff_coverage(["coverage.xml"])
        assert total == 0.0
        assert per_file == {}
        assert error == ""


class TestComputeDiffCoverageMultiFile:
    @patch.dict(os.environ, {}, clear=False)
    @patch("src.layer1_coverage.subprocess.run")
    def test_passes_multiple_files_as_positional_args(self, mock_run):
        os.environ.pop("GITHUB_BASE_REF", None)
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout='{"total_percent_covered": 80.0, "src_stats": {}}',
            stderr="",
        )
        _compute_diff_coverage(["a.xml", "b.xml"])
        cmd = mock_run.call_args[0][0]
        assert cmd == ["diff-cover", "a.xml", "b.xml", "--json-report", "/dev/stdout", "--quiet"]

    @patch.dict(os.environ, {}, clear=False)
    @patch("src.layer1_coverage.subprocess.run")
    def test_single_file_list_works(self, mock_run):
        os.environ.pop("GITHUB_BASE_REF", None)
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout='{"total_percent_covered": 90.0, "src_stats": {}}',
            stderr="",
        )
        total, per_file, error = _compute_diff_coverage(["coverage.xml"])
        assert total == 90.0
        assert per_file == {}
        assert error == ""
        cmd = mock_run.call_args[0][0]
        assert cmd == ["diff-cover", "coverage.xml", "--json-report", "/dev/stdout", "--quiet"]


class TestCompareBranchAutoDetect:
    @patch.dict(os.environ, {"GITHUB_BASE_REF": "develop"})
    @patch("src.layer1_coverage.subprocess.run")
    def test_appends_compare_branch_when_base_ref_set(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout='{"total_percent_covered": 80.0, "src_stats": {}}',
            stderr="",
        )
        _compute_diff_coverage(["coverage.xml"])
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "diff-cover", "coverage.xml",
            "--json-report", "/dev/stdout", "--quiet",
            "--compare-branch=origin/develop",
        ]

    @patch.dict(os.environ, {"GITHUB_BASE_REF": "main"})
    @patch("src.layer1_coverage.subprocess.run")
    def test_appends_compare_branch_for_main(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout='{"total_percent_covered": 75.0, "src_stats": {}}',
            stderr="",
        )
        _compute_diff_coverage(["coverage.xml"])
        cmd = mock_run.call_args[0][0]
        assert "--compare-branch=origin/main" in cmd

    @patch.dict(os.environ, {}, clear=False)
    @patch("src.layer1_coverage.subprocess.run")
    def test_no_compare_branch_when_base_ref_unset(self, mock_run):
        os.environ.pop("GITHUB_BASE_REF", None)
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout='{"total_percent_covered": 80.0, "src_stats": {}}',
            stderr="",
        )
        _compute_diff_coverage(["coverage.xml"])
        cmd = mock_run.call_args[0][0]
        assert not any(arg.startswith("--compare-branch") for arg in cmd)

    @patch.dict(os.environ, {"GITHUB_BASE_REF": "  "})
    @patch("src.layer1_coverage.subprocess.run")
    def test_no_compare_branch_when_base_ref_blank(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout='{"total_percent_covered": 80.0, "src_stats": {}}',
            stderr="",
        )
        _compute_diff_coverage(["coverage.xml"])
        cmd = mock_run.call_args[0][0]
        assert not any(arg.startswith("--compare-branch") for arg in cmd)


class TestRunLayer1MultiFile:
    def test_skip_when_empty_list(self):
        result = run_layer1(coverage_files=[], threshold=80, diff_files=[])
        assert result.verdict == Verdict.SKIP
        assert result.short_circuit is False

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_filters_missing_files_and_runs_remaining(self, mock_cov, tmp_path):
        real = tmp_path / "real.xml"
        real.write_text("<xml/>")
        mock_cov.return_value = (85.0, {"src/a.py": 85.0}, "")
        result = run_layer1(
            coverage_files=[str(real), str(tmp_path / "missing.xml")],
            threshold=80,
            diff_files=["src/a.py"],
        )
        assert result.verdict == Verdict.PASS
        mock_cov.assert_called_once_with([str(real)])

    def test_skip_when_all_files_missing(self, tmp_path):
        result = run_layer1(
            coverage_files=[str(tmp_path / "nope.xml")],
            threshold=80,
            diff_files=["src/a.py"],
        )
        assert result.verdict == Verdict.SKIP
        assert "not found" in result.details.lower() or "no valid" in result.details.lower()
