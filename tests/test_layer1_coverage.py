# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateUsage=false

import subprocess
from unittest.mock import patch

from src.layer1_coverage import _DIFF_COVER_TIMEOUT, _compute_diff_coverage, run_layer1
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
        result = run_layer1(coverage_file=None, threshold=80, diff_files=[])
        assert result.verdict == Verdict.SKIP
        assert result.short_circuit is False
        assert result.coverage_details is None

    def test_skip_when_coverage_file_missing(self, tmp_path):
        fake_path = str(tmp_path / "nonexistent.xml")
        result = run_layer1(coverage_file=fake_path, threshold=80, diff_files=["src/foo.py"])
        assert result.verdict == Verdict.SKIP
        assert result.coverage_details is None

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_pass_when_all_files_above_threshold(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        mock_cov.return_value = (92.5, {"src/auth.py": 92.5})
        result = run_layer1(
            coverage_file=str(cov),
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
        mock_cov.return_value = (85.0, per_file)
        result = run_layer1(
            coverage_file=str(cov),
            threshold=80,
            diff_files=["src/auth.py", "src/billing.py"],
        )
        assert result.verdict == Verdict.FAIL
        assert result.short_circuit is False
        assert result.coverage_details == per_file

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_no_short_circuit_when_file_absent_from_src_stats(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        mock_cov.return_value = (92.5, {"src/auth.py": 92.5})
        result = run_layer1(
            coverage_file=str(cov),
            threshold=80,
            diff_files=["src/auth.py", "src/new_feature.py"],
        )
        assert result.verdict == Verdict.FAIL
        assert result.short_circuit is False
        assert result.coverage_details == {"src/auth.py": 92.5}

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_pass_exactly_at_threshold(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        mock_cov.return_value = (80.0, {"src/auth.py": 80.0})
        result = run_layer1(
            coverage_file=str(cov),
            threshold=80,
            diff_files=["src/auth.py"],
        )
        assert result.verdict == Verdict.PASS
        assert result.short_circuit is True

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_skip_when_diff_cover_returns_negative(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        mock_cov.return_value = (-1.0, {})
        result = run_layer1(
            coverage_file=str(cov),
            threshold=80,
            diff_files=["src/auth.py"],
        )
        assert result.verdict == Verdict.SKIP
        assert result.short_circuit is False
        assert "diff-cover failed" in result.details
        assert result.coverage_details is None

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_coverage_details_populated_on_pass(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        per_file = {"src/a.py": 95.0, "src/b.py": 88.0}
        mock_cov.return_value = (91.5, per_file)
        result = run_layer1(
            coverage_file=str(cov),
            threshold=80,
            diff_files=["src/a.py", "src/b.py"],
        )
        assert result.coverage_details == per_file

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_non_source_files_ignored_in_per_file_check(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        mock_cov.return_value = (92.0, {"src/auth.py": 92.0})
        result = run_layer1(
            coverage_file=str(cov),
            threshold=80,
            diff_files=["src/auth.py", "tests/test_auth.py", "README.md"],
        )
        assert result.verdict == Verdict.PASS
        assert result.short_circuit is True


class TestComputeDiffCoverage:
    @patch("src.layer1_coverage.subprocess.run")
    def test_returns_aggregate_and_per_file(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout=_SAMPLE_SRC_STATS_JSON,
            stderr="",
        )
        total, per_file = _compute_diff_coverage("coverage.xml")
        assert total == 85.0
        assert per_file == {"src/auth.py": 92.5, "src/billing.py": 25.0}

    @patch("src.layer1_coverage.subprocess.run")
    def test_returns_negative_and_empty_when_non_zero(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=1,
            stdout="",
            stderr="error",
        )
        total, per_file = _compute_diff_coverage("coverage.xml")
        assert total == -1.0
        assert per_file == {}

    @patch("src.layer1_coverage.subprocess.run")
    def test_returns_negative_and_empty_on_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="diff-cover", timeout=60)
        total, per_file = _compute_diff_coverage("coverage.xml")
        assert total == -1.0
        assert per_file == {}

    @patch("src.layer1_coverage.subprocess.run")
    def test_returns_negative_and_empty_on_invalid_json(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout="not-json",
            stderr="",
        )
        total, per_file = _compute_diff_coverage("coverage.xml")
        assert total == -1.0
        assert per_file == {}

    @patch("src.layer1_coverage.subprocess.run")
    def test_returns_negative_and_empty_when_not_installed(self, mock_run):
        mock_run.side_effect = FileNotFoundError("diff-cover")
        total, per_file = _compute_diff_coverage("coverage.xml")
        assert total == -1.0
        assert per_file == {}

    @patch("src.layer1_coverage.subprocess.run")
    def test_handles_missing_src_stats_key(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout='{"total_percent_covered": 50.0}',
            stderr="",
        )
        total, per_file = _compute_diff_coverage("coverage.xml")
        assert total == 50.0
        assert per_file == {}

    @patch("src.layer1_coverage.subprocess.run")
    def test_handles_empty_src_stats(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout='{"total_percent_covered": 0.0, "src_stats": {}}',
            stderr="",
        )
        total, per_file = _compute_diff_coverage("coverage.xml")
        assert total == 0.0
        assert per_file == {}
