"""Tests for main orchestrator."""
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedParameter=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

import pytest
from unittest.mock import patch

from src.main import run_pipeline
from src.models import LayerResult, Verdict
from src.config import Config


@pytest.fixture
def base_config():
    return Config(
        github_token="ghp_fake",
        repo="owner/repo",
        pr_number=42,
        event_name="pull_request",
        coverage_file=None,
        coverage_threshold=80,
        test_patterns={"python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"}},
        exclude_patterns=["*.md"],
        ai_enabled=True,
        ai_model="openai/gpt-5-mini",
        ai_confidence_threshold=0.7,
    )


class TestRunPipeline:
    @patch("src.main._get_pr_context")
    @patch("src.main.run_layer1")
    @patch("src.main.report_to_github")
    def test_layer1_pass_short_circuits(self, mock_report, mock_l1, mock_ctx, base_config):
        mock_ctx.return_value = (["src/auth.py"], ["src/auth.py", "tests/test_auth.py"], "sha123", {})
        mock_l1.return_value = LayerResult("layer1", Verdict.PASS, "92%", [], True)

        report = run_pipeline(base_config)
        assert report.overall_verdict == Verdict.PASS
        assert len(report.layers) == 1  # Only layer1 ran

    @patch("src.main._get_pr_context")
    @patch("src.main.run_layer1")
    @patch("src.main.run_layer2")
    @patch("src.main.run_layer3")
    @patch("src.main.report_to_github")
    def test_full_pipeline_all_layers(
        self, mock_report, mock_l3, mock_l2, mock_l1, mock_ctx, base_config
    ):
        mock_ctx.return_value = (
            ["src/billing.py"],
            ["src/billing.py"],
            "sha123",
            {"src/billing.py": "+ new_code"},
        )
        mock_l1.return_value = LayerResult("layer1", Verdict.SKIP, "No coverage", [], False)
        mock_l2.return_value = LayerResult("layer2", Verdict.FAIL, "Missing test", [], False)
        mock_l3.return_value = LayerResult("layer3", Verdict.FAIL, "AI: fail", [], False)

        report = run_pipeline(base_config)
        assert report.overall_verdict == Verdict.FAIL
        assert len(report.layers) == 3

    @patch("src.main._get_pr_context")
    @patch("src.main.run_layer1")
    @patch("src.main.run_layer2")
    @patch("src.main.report_to_github")
    def test_ai_disabled_skips_layer3(self, mock_report, mock_l2, mock_l1, mock_ctx, base_config):
        base_config = Config(
            github_token=base_config.github_token,
            repo=base_config.repo,
            pr_number=base_config.pr_number,
            event_name=base_config.event_name,
            coverage_file=base_config.coverage_file,
            coverage_threshold=base_config.coverage_threshold,
            test_patterns=base_config.test_patterns,
            exclude_patterns=base_config.exclude_patterns,
            ai_enabled=False,
            ai_model=base_config.ai_model,
            ai_confidence_threshold=base_config.ai_confidence_threshold,
        )
        mock_ctx.return_value = (["src/x.py"], ["src/x.py"], "sha123", {})
        mock_l1.return_value = LayerResult("layer1", Verdict.SKIP, "No cov", [], False)
        mock_l2.return_value = LayerResult("layer2", Verdict.FAIL, "Missing", [], False)

        report = run_pipeline(base_config)
        assert len(report.layers) == 2  # Layer 3 was skipped
