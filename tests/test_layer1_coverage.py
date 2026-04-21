# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Tests for Layer 1 — diff-coverage gate."""

from unittest.mock import patch

from src.layer1_coverage import run_layer1
from src.models import Verdict


class TestRunLayer1:
    def test_skip_when_no_coverage_file(self):
        """Layer 1 returns SKIP when no coverage file is configured."""
        result = run_layer1(coverage_file=None, threshold=80, diff_files=[])
        assert result.verdict == Verdict.SKIP
        assert result.short_circuit is False

    def test_skip_when_coverage_file_missing(self, tmp_path):
        """Layer 1 returns SKIP when coverage file doesn't exist on disk."""
        fake_path = str(tmp_path / "nonexistent.xml")
        result = run_layer1(coverage_file=fake_path, threshold=80, diff_files=["src/foo.py"])
        assert result.verdict == Verdict.SKIP

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_pass_when_above_threshold(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        mock_cov.return_value = 92.5
        result = run_layer1(
            coverage_file=str(cov),
            threshold=80,
            diff_files=["src/auth.py"],
        )
        assert result.verdict == Verdict.PASS
        assert result.short_circuit is True
        assert "92.5%" in result.details

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_fail_when_below_threshold(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        mock_cov.return_value = 45.0
        result = run_layer1(
            coverage_file=str(cov),
            threshold=80,
            diff_files=["src/billing.py"],
        )
        assert result.verdict == Verdict.FAIL
        assert result.short_circuit is False
        assert "45.0%" in result.details

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_pass_exactly_at_threshold(self, mock_cov, tmp_path):
        cov = tmp_path / "coverage.xml"
        cov.write_text("<xml/>")
        mock_cov.return_value = 80.0
        result = run_layer1(
            coverage_file=str(cov),
            threshold=80,
            diff_files=["src/auth.py"],
        )
        assert result.verdict == Verdict.PASS
        assert result.short_circuit is True
