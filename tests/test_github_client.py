# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Tests for GitHub client — PR comments and status checks."""

from unittest.mock import MagicMock, patch

import pytest

from src import github_client
from src.github_client import format_report, post_comment, post_status, report_to_github
from src.models import FileVerdict, LayerResult, Report, Verdict


@pytest.fixture
def sample_report() -> Report:
    return Report(
        layers=[
            LayerResult(
                layer="layer1",
                verdict=Verdict.PASS,
                details="Changed lines: 92% covered (threshold: 80%)",
                file_verdicts=[],
                short_circuit=True,
            ),
        ]
    )


@pytest.fixture
def full_report() -> Report:
    return Report(
        layers=[
            LayerResult("layer1", Verdict.FAIL, "Changed lines: 45% (threshold: 80%)", [], False),
            LayerResult(
                "layer2",
                Verdict.FAIL,
                "File matching: 1 pass, 1 fail",
                [
                    FileVerdict(
                        "src/auth.py", Verdict.PASS, "Test modified: tests/test_auth.py", "layer2"
                    ),
                    FileVerdict("src/billing.py", Verdict.FAIL, "No matching test file", "layer2"),
                ],
                False,
            ),
            LayerResult(
                "layer3",
                Verdict.WARNING,
                "AI verdict: warning (confidence: 82%)",
                [
                    FileVerdict(
                        "src/billing.py",
                        Verdict.FAIL,
                        "No edge case test for negatives",
                        "layer3",
                    ),
                ],
                False,
            ),
        ]
    )


class TestFormatReport:
    def test_short_circuit_report(self, sample_report):
        md = format_report(sample_report)
        assert "## 🧪 Test Guard Report" in md
        assert "Layer 1" in md
        assert "92%" in md
        assert "✅" in md

    def test_full_report_with_all_layers(self, full_report):
        md = format_report(full_report)
        assert "Layer 1" in md
        assert "Layer 2" in md
        assert "Layer 3" in md
        assert "src/billing.py" in md
        assert "WARNING" in md or "⚠️" in md


class TestPostComment:
    @patch("src.github_client.post_json")
    def test_posts_to_correct_endpoint(self, mock_post_json):
        mock_post_json.return_value = MagicMock(ok=True, status_code=201, text="")
        session = MagicMock()
        post_comment(
            session=session,
            repo="owner/repo",
            pr_number=42,
            body="## Report\nAll good.",
        )
        mock_post_json.assert_called_once()
        call_args = mock_post_json.call_args
        assert call_args[0][0] is session
        assert call_args[0][1] == "https://api.github.com/repos/owner/repo/issues/42/comments"
        assert call_args[0][2] == {"body": "## Report\nAll good."}

    @patch("src.github_client.post_json")
    def test_redacts_token_on_comment_warning(self, mock_post_json, capsys):
        mock_post_json.return_value = MagicMock(
            ok=False,
            status_code=400,
            text="bad token ghp_ABC123 and Bearer secret-token",
        )
        post_comment(session=MagicMock(), repo="o/r", pr_number=1, body="x")
        out = capsys.readouterr().out
        assert "ghp_ABC123" not in out
        assert "secret-token" not in out
        assert "[REDACTED]" in out


class TestPostStatus:
    @patch("src.github_client.post_json")
    def test_posts_success_status(self, mock_post_json):
        mock_post_json.return_value = MagicMock(ok=True, status_code=201, text="")
        session = MagicMock()
        post_status(
            session=session,
            repo="owner/repo",
            sha="abc123",
            state="success",
            description="All checks passed",
        )
        mock_post_json.assert_called_once()
        url = mock_post_json.call_args[0][1]
        assert "statuses/abc123" in url
        assert mock_post_json.call_args[0][0] is session

    @patch("src.github_client.post_json")
    def test_posts_failure_status(self, mock_post_json):
        mock_post_json.return_value = MagicMock(ok=True, status_code=201, text="")
        post_status(
            session=MagicMock(),
            repo="owner/repo",
            sha="abc123",
            state="failure",
            description="Missing tests for 2 files",
        )
        body = mock_post_json.call_args[0][2]
        assert body["state"] == "failure"
        assert body["context"] == "test-guard"


class TestRedaction:
    def test_redacts_known_token_patterns(self):
        redact = github_client.__dict__["_redact_response_text"]
        text = "ghp_abc123 gho_def456 github_pat_ghi789 Bearer topsecret"
        redacted = redact(text)
        assert "ghp_abc123" not in redacted
        assert "gho_def456" not in redacted
        assert "github_pat_ghi789" not in redacted
        assert "topsecret" not in redacted
        assert redacted.count("[REDACTED]") >= 4

    def test_keeps_normal_error_text(self):
        redact = github_client.__dict__["_redact_response_text"]
        text = "validation failed: missing field name"
        assert redact(text) == text


class TestReportToGitHub:
    @patch("src.github_client.post_comment")
    @patch("src.github_client.post_status")
    @patch("src.github_client.create_session")
    def test_pr_number_zero_posts_comment(
        self,
        mock_create_session,
        mock_post_status,
        mock_post_comment,
        sample_report,
    ):
        session = MagicMock()
        mock_create_session.return_value = session

        report_to_github(sample_report, "ghp_fake", "owner/repo", 0, "abc123")

        mock_post_status.assert_called_once()
        mock_post_comment.assert_called_once()
        assert mock_post_comment.call_args[0][0] is session

    @patch(
        "src.github_client.create_session",
        side_effect=RuntimeError("Bearer ghp_FAKE exploded"),
    )
    def test_sanitizes_traceback_in_error_scenarios(
        self,
        _mock_create_session,
        sample_report,
        capsys,
    ):
        report_to_github(sample_report, "ghp_secret", "owner/repo", 1, "abc123")
        out = capsys.readouterr().out
        assert "Traceback" not in out
        assert "ghp_secret" not in out
        assert "ghp_FAKE" not in out
        assert "[REDACTED]" in out
