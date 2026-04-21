"""Tests for GitHub client — PR comments and status checks."""

from unittest.mock import MagicMock, patch

import pytest

from src.github_client import format_report, post_comment, post_status
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
    @patch("src.github_client.requests.post")
    def test_posts_to_correct_endpoint(self, mock_post):
        mock_post.return_value = MagicMock(status_code=201)
        post_comment(
            token="ghp_fake",
            repo="owner/repo",
            pr_number=42,
            body="## Report\nAll good.",
        )
        mock_post.assert_called_once()
        url = mock_post.call_args[0][0]
        assert url == "https://api.github.com/repos/owner/repo/issues/42/comments"

    @patch("src.github_client.requests.post")
    def test_includes_auth_header(self, mock_post):
        mock_post.return_value = MagicMock(status_code=201)
        post_comment(token="ghp_test", repo="o/r", pr_number=1, body="x")
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer ghp_test"


class TestPostStatus:
    @patch("src.github_client.requests.post")
    def test_posts_success_status(self, mock_post):
        mock_post.return_value = MagicMock(status_code=201)
        post_status(
            token="ghp_fake",
            repo="owner/repo",
            sha="abc123",
            state="success",
            description="All checks passed",
        )
        mock_post.assert_called_once()
        url = mock_post.call_args[0][0]
        assert "statuses/abc123" in url

    @patch("src.github_client.requests.post")
    def test_posts_failure_status(self, mock_post):
        mock_post.return_value = MagicMock(status_code=201)
        post_status(
            token="ghp_fake",
            repo="owner/repo",
            sha="abc123",
            state="failure",
            description="Missing tests for 2 files",
        )
        body = mock_post.call_args[1]["json"]
        assert body["state"] == "failure"
        assert body["context"] == "test-guard"
