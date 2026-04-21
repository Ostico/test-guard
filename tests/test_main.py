"""Tests for main orchestrator."""
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportPrivateUsage=false

from unittest.mock import MagicMock, call, patch

import pytest

from src.config import Config
from src.main import _get_pr_context, main, run_pipeline
from src.models import FileVerdict, LayerResult, Report, Verdict


@pytest.fixture
def base_config():
    return Config(
        github_token="ghp_fake",
        repo="owner/repo",
        pr_number=42,
        event_name="pull_request",
        coverage_file=None,
        coverage_threshold=80,
        test_patterns={
            "python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"},
        },
        exclude_patterns=["*.md"],
        ai_enabled=True,
        ai_model="openai/gpt-5-mini",
        ai_confidence_threshold=0.7,
    )


class TestRunPipeline:
    @patch("src.main._get_pr_context")
    @patch("src.main.run_layer1")
    @patch("src.main.report_to_github")
    def test_layer1_pass_short_circuits(self, _mock_report, mock_l1, mock_ctx, base_config):
        mock_ctx.return_value = (
            ["src/auth.py"],
            ["src/auth.py", "tests/test_auth.py"],
            "sha123",
            {},
        )
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
        self, _mock_report, mock_l3, mock_l2, mock_l1, mock_ctx, base_config
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
    def test_ai_disabled_skips_layer3(self, _mock_report, mock_l2, mock_l1, mock_ctx, base_config):
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

    @patch("src.main._get_pr_context")
    @patch("src.main.run_layer1")
    @patch("src.main.run_layer2")
    @patch("src.main.report_to_github")
    def test_layer2_pass_short_circuits(self, mock_report, mock_l2, mock_l1, mock_ctx, base_config):
        mock_ctx.return_value = (
            ["src/payments.py"],
            ["src/payments.py", "tests/test_payments.py"],
            "sha456",
            {},
        )
        mock_l1.return_value = LayerResult("layer1", Verdict.SKIP, "No coverage", [], False)
        mock_l2.return_value = LayerResult("layer2", Verdict.PASS, "Matched tests", [], True)

        report = run_pipeline(base_config)

        assert report.overall_verdict == Verdict.PASS
        assert len(report.layers) == 2
        mock_report.assert_called_once_with(
            report,
            base_config.github_token,
            base_config.repo,
            base_config.pr_number,
            "sha456",
        )

    @patch("src.main.get_text")
    @patch("src.main.run_layer3")
    @patch("src.main.run_layer2")
    @patch("src.main.run_layer1")
    @patch("src.main._get_pr_context")
    @patch("src.main.create_session")
    @patch("src.main.report_to_github")
    def test_uses_matched_test_for_ai_context_fetch(
        self,
        _mock_report,
        mock_create_session,
        mock_ctx,
        mock_l1,
        mock_l2,
        mock_l3,
        mock_get_text,
        base_config,
    ):
        session = MagicMock()
        mock_create_session.return_value = session
        mock_ctx.return_value = (
            ["src/auth.py"],
            ["src/auth.py", "tests/test_auth.py"],
            "sha123",
            {"src/auth.py": "+new"},
        )
        mock_l1.return_value = LayerResult("layer1", Verdict.SKIP, "No coverage", [], False)
        mock_l2.return_value = LayerResult(
            "layer2",
            Verdict.WARNING,
            "Needs review",
            [
                FileVerdict(
                    file="src/auth.py",
                    verdict=Verdict.WARNING,
                    reason="warning without parseable path",
                    layer="layer2",
                    matched_test="tests/test_auth.py",
                )
            ],
            False,
        )
        mock_l3.return_value = LayerResult("layer3", Verdict.PASS, "AI ok", [], False)
        mock_get_text.return_value = "def test_auth():\n    assert True\n"

        run_pipeline(base_config)

        mock_create_session.assert_called_once_with(base_config.github_token)
        mock_ctx.assert_called_once_with(base_config, session)
        mock_get_text.assert_called_once_with(
            session,
            "https://api.github.com/repos/owner/repo/contents/tests/test_auth.py?ref=sha123",
        )
        mock_l3.assert_called_once_with(
            {"src/auth.py": "+new"},
            {"tests/test_auth.py": "def test_auth():\n    assert True\n"},
            base_config.ai_model,
            base_config.github_token,
            base_config.ai_confidence_threshold,
        )


class TestGetPrContext:
    @patch("src.main.get_json")
    @patch("src.main.get_paginated")
    def test_fetches_pr_files_with_pagination(self, mock_get_paginated, mock_get_json, base_config):
        session = MagicMock()
        pr_files = [
            {"filename": f"src/file_{i}.py", "patch": "+change"}
            for i in range(150)
        ]
        mock_get_paginated.return_value = pr_files
        mock_get_json.side_effect = [
            {"head": {"sha": "sha123"}},
            {
                "tree": [
                    {"path": "src/file_1.py", "type": "blob"},
                    {"path": "tests/test_file_1.py", "type": "blob"},
                    {"path": "src", "type": "tree"},
                ]
            },
        ]

        changed_files, all_repo_files, head_sha, file_diffs = _get_pr_context(base_config, session)

        mock_get_paginated.assert_called_once_with(
            session,
            "https://api.github.com/repos/owner/repo/pulls/42/files",
        )
        assert mock_get_json.call_args_list == [
            call(session, "https://api.github.com/repos/owner/repo/pulls/42"),
            call(session, "https://api.github.com/repos/owner/repo/git/trees/sha123?recursive=1"),
        ]
        assert len(changed_files) == 150
        assert changed_files[0] == "src/file_0.py"
        assert head_sha == "sha123"
        assert all_repo_files == ["src/file_1.py", "tests/test_file_1.py"]
        assert file_diffs["src/file_149.py"] == "+change"


class TestMainEntryPoint:
    def _config(self, *, event_name: str = "pull_request", pr_number: int | None = 42) -> Config:
        return Config(
            github_token="ghp_fake",
            repo="owner/repo",
            pr_number=pr_number,
            event_name=event_name,
            coverage_file=None,
            coverage_threshold=80,
            test_patterns={
                "python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"},
            },
            exclude_patterns=["*.md"],
            ai_enabled=True,
            ai_model="openai/gpt-5-mini",
            ai_confidence_threshold=0.7,
        )

    @patch("src.main.run_pipeline")
    @patch("src.main.parse_config")
    def test_non_pr_event_prints_notice_and_returns(
        self,
        mock_parse_config,
        mock_run_pipeline,
        capsys: pytest.CaptureFixture[str],
    ):
        mock_parse_config.return_value = self._config(event_name="push")

        main()

        mock_run_pipeline.assert_not_called()
        assert "only runs on pull_request events. Skipping" in capsys.readouterr().out

    @patch("src.main.run_pipeline")
    @patch("src.main.parse_config")
    def test_missing_pr_number_exits(self, mock_parse_config, mock_run_pipeline):
        mock_parse_config.return_value = self._config(pr_number=None)

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        mock_run_pipeline.assert_not_called()

    @patch("src.main.traceback.print_exc")
    @patch("src.main.parse_config")
    def test_config_parse_error_exits(
        self,
        mock_parse_config,
        mock_print_exc,
        capsys: pytest.CaptureFixture[str],
    ):
        mock_parse_config.side_effect = ValueError("bad config")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        mock_print_exc.assert_called_once()
        assert "Failed to parse configuration" in capsys.readouterr().out

    @patch("src.main.traceback.print_exc")
    @patch("src.main.run_pipeline")
    @patch("src.main.parse_config")
    def test_pipeline_error_exits(
        self,
        mock_parse_config,
        mock_run_pipeline,
        mock_print_exc,
        capsys: pytest.CaptureFixture[str],
    ):
        mock_parse_config.return_value = self._config()
        mock_run_pipeline.side_effect = RuntimeError("boom")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        mock_print_exc.assert_called_once()
        assert "Pipeline failed with an unexpected error" in capsys.readouterr().out

    @patch("src.main.format_report")
    @patch("src.main.run_pipeline")
    @patch("src.main.parse_config")
    def test_pipeline_fail_verdict_exits(
        self,
        mock_parse_config,
        mock_run_pipeline,
        mock_format_report,
    ):
        mock_parse_config.return_value = self._config()
        mock_run_pipeline.return_value = Report(
            layers=[LayerResult("layer3", Verdict.FAIL, "fail", [], False)]
        )
        mock_format_report.return_value = "formatted report"

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("src.main.format_report")
    @patch("src.main.run_pipeline")
    @patch("src.main.parse_config")
    def test_pipeline_warning_verdict_prints_warning(
        self,
        mock_parse_config,
        mock_run_pipeline,
        mock_format_report,
        capsys: pytest.CaptureFixture[str],
    ):
        mock_parse_config.return_value = self._config()
        mock_run_pipeline.return_value = Report(
            layers=[LayerResult("layer3", Verdict.WARNING, "warn", [], False)]
        )
        mock_format_report.return_value = "formatted warning report"

        main()

        output = capsys.readouterr().out
        assert "formatted warning report" in output
        assert "Test adequacy warnings found (non-blocking)" in output

    @patch("src.main.format_report")
    @patch("src.main.run_pipeline")
    @patch("src.main.parse_config")
    def test_pipeline_pass_verdict_prints_notice(
        self,
        mock_parse_config,
        mock_run_pipeline,
        mock_format_report,
        capsys: pytest.CaptureFixture[str],
    ):
        mock_parse_config.return_value = self._config()
        mock_run_pipeline.return_value = Report(
            layers=[LayerResult("layer3", Verdict.PASS, "ok", [], False)]
        )
        mock_format_report.return_value = "formatted pass report"

        main()

        output = capsys.readouterr().out
        assert "formatted pass report" in output
        assert "Test adequacy check passed" in output
