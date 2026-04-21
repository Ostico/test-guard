"""Tests for Layer 3 — GPT-5-mini AI judgment."""
import json
import pytest
from unittest.mock import patch, MagicMock

from src.layer3_ai import run_layer3, _build_prompt, _parse_ai_response
from src.models import FileVerdict, Verdict


class TestBuildPrompt:
    def test_includes_diff_and_test_content(self):
        prompt = _build_prompt(
            file_diffs={"src/auth.py": "- old\n+ new"},
            test_contents={"tests/test_auth.py": "def test_login(): ..."},
        )
        assert "src/auth.py" in prompt
        assert "- old" in prompt
        assert "+ new" in prompt
        assert "test_login" in prompt

    def test_handles_file_with_no_test(self):
        prompt = _build_prompt(
            file_diffs={"src/billing.py": "+ new_feature()"},
            test_contents={},
        )
        assert "src/billing.py" in prompt
        assert "No test file found" in prompt


class TestParseAiResponse:
    def test_valid_response(self):
        raw = json.dumps({
            "verdict": "warning",
            "confidence": 0.82,
            "files": [
                {"file": "src/billing.py", "verdict": "fail", "reason": "No edge case test"},
            ],
        })
        verdict, confidence, file_verdicts = _parse_ai_response(raw)
        assert verdict == Verdict.WARNING
        assert confidence == 0.82
        assert len(file_verdicts) == 1
        assert file_verdicts[0].verdict == Verdict.FAIL

    def test_invalid_json_returns_skip(self):
        verdict, confidence, file_verdicts = _parse_ai_response("not json at all")
        assert verdict == Verdict.SKIP
        assert confidence == 0.0

    def test_missing_fields_returns_skip(self):
        raw = json.dumps({"verdict": "pass"})  # missing confidence, files
        verdict, confidence, file_verdicts = _parse_ai_response(raw)
        assert verdict == Verdict.SKIP


class TestRunLayer3:
    @patch("src.layer3_ai._call_github_models")
    def test_pass_with_high_confidence(self, mock_call):
        mock_call.return_value = json.dumps({
            "verdict": "pass",
            "confidence": 0.95,
            "files": [
                {"file": "src/auth.py", "verdict": "pass", "reason": "Well tested"},
            ],
        })
        result = run_layer3(
            file_diffs={"src/auth.py": "+ new_code"},
            test_contents={"tests/test_auth.py": "def test(): ..."},
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.PASS

    @patch("src.layer3_ai._call_github_models")
    def test_low_confidence_becomes_warning(self, mock_call):
        mock_call.return_value = json.dumps({
            "verdict": "fail",
            "confidence": 0.4,
            "files": [
                {"file": "src/auth.py", "verdict": "fail", "reason": "Maybe missing"},
            ],
        })
        result = run_layer3(
            file_diffs={"src/auth.py": "+ code"},
            test_contents={},
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        # Below confidence threshold -> WARNING instead of FAIL
        assert result.verdict == Verdict.WARNING

    @patch("src.layer3_ai._call_github_models")
    def test_api_failure_returns_skip(self, mock_call):
        mock_call.side_effect = Exception("API down")
        result = run_layer3(
            file_diffs={"src/auth.py": "+ code"},
            test_contents={},
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.SKIP

    def test_empty_diffs_returns_pass(self):
        result = run_layer3(
            file_diffs={},
            test_contents={},
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.PASS
