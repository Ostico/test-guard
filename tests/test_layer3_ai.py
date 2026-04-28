# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false

import json
from unittest.mock import MagicMock, patch

from openai import APIStatusError

import src.layer3_ai as layer3_ai
from src.layer3_ai import (
    Layer3Result,
    Relevance,
    _batch_files,
    _build_ai_prompt,
    _call_ai_for_batch,
    _call_github_models,
    _estimate_file_cost,
    _estimate_tokens,
    _filter_test_diffs_for_batch,
    _is_model_forbidden,
    _is_retryable_size_error,
    _parse_ai_response,
    _resolve_models,
    _sanitize_diff,
    _validate_batch_verdicts,
    compute_test_relevance,
    evaluate_file_shortcut,
    is_trivial_diff,
    run_layer3,
)
from src.models import FileVerdict, Verdict


def _make_api_error(status_code: int, message: str = "error") -> APIStatusError:
    mock_response = MagicMock()
    mock_response.status_code = status_code
    return APIStatusError(message=message, response=mock_response, body=None)


class TestSanitizeDiff:
    def test_truncates_large_diff(self):
        result = _sanitize_diff("a" * 20, max_chars=10)
        assert result == ("a" * 10) + "...[truncated]"


class TestParseAiResponse:
    def test_valid_response(self):
        raw = json.dumps(
            {
                "verdict": "warning",
                "confidence": 0.82,
                "files": [
                    {"file": "src/billing.py", "verdict": "fail", "reason": "No edge case test"},
                ],
            }
        )
        verdict, confidence, file_verdicts = _parse_ai_response(raw)
        assert verdict == Verdict.WARNING
        assert confidence == 0.82
        assert len(file_verdicts) == 1
        assert file_verdicts[0].verdict == Verdict.FAIL

    def test_invalid_json_returns_skip(self):
        verdict, confidence, _ = _parse_ai_response("not json at all")
        assert verdict == Verdict.SKIP
        assert confidence == 0.0

    def test_unexpected_verdict_maps_to_skip(self):
        raw = json.dumps(
            {
                "verdict": "maybe",
                "confidence": 0.9,
                "files": [],
            }
        )
        verdict, confidence, file_verdicts = _parse_ai_response(raw)
        assert verdict == Verdict.SKIP
        assert confidence == 0.9
        assert file_verdicts == []


class TestLayer3Result:
    def test_error_status_with_verdicts_computes_normally(self):
        """ERROR + non-empty per_file_verdicts → compute from verdicts, not SKIP."""
        result = Layer3Result({"src/a.py": Verdict.PASS}, "ERROR")
        assert result.verdict == Verdict.PASS

    def test_error_status_no_verdicts_returns_skip(self):
        """ERROR + empty per_file_verdicts → SKIP (full fallback)."""
        result = Layer3Result({}, "ERROR")
        assert result.verdict == Verdict.SKIP

    def test_all_skip_returns_pass(self):
        result = Layer3Result({"src/a.py": Verdict.SKIP, "src/b.py": Verdict.SKIP}, "OK")
        assert result.verdict == Verdict.PASS

    def test_fail_worst_wins(self):
        result = Layer3Result({"src/a.py": Verdict.PASS, "src/b.py": Verdict.FAIL}, "OK")
        assert result.verdict == Verdict.FAIL

    def test_warning_worst_wins(self):
        result = Layer3Result({"src/a.py": Verdict.PASS, "src/b.py": Verdict.WARNING}, "OK")
        assert result.verdict == Verdict.WARNING

    def test_all_pass_returns_pass(self):
        result = Layer3Result({"src/a.py": Verdict.PASS}, "OK")
        assert result.verdict == Verdict.PASS

    def test_skip_and_pass_returns_pass(self):
        result = Layer3Result({"src/a.py": Verdict.SKIP, "src/b.py": Verdict.PASS}, "OK")
        assert result.verdict == Verdict.PASS


class TestRunLayer3:
    @patch("src.layer3_ai._call_github_models")
    def test_all_shortcuts_no_ai_called(self, mock_call: MagicMock):
        result = run_layer3(
            source_diffs={"src/trivial.py": "+ # comment"},
            deleted_files=set(),
            test_diffs={},
            l2_matched_tests={"src/trivial.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        mock_call.assert_not_called()
        assert result.verdict == Verdict.PASS

    @patch("src.layer3_ai._call_github_models")
    def test_deleted_files_produce_pass(self, mock_call: MagicMock):
        result = run_layer3(
            source_diffs={"src/old.py": "+ code"},
            deleted_files={"src/old.py"},
            test_diffs={},
            l2_matched_tests={"src/old.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        mock_call.assert_not_called()
        assert result.verdict == Verdict.PASS

    @patch("src.layer3_ai._call_github_models")
    def test_fallthrough_calls_ai(self, mock_call: MagicMock):
        mock_call.return_value = json.dumps({
            "verdict": "pass",
            "confidence": 0.95,
            "files": [{"file": "src/new.py", "verdict": "pass", "reason": "Well tested"}],
        })
        result = run_layer3(
            source_diffs={"src/new.py": "+ new_code()"},
            deleted_files=set(),
            test_diffs={"tests/test_new.py": "+ def test_new(): ..."},
            l2_matched_tests={"src/new.py": "tests/test_new.py"},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        mock_call.assert_called_once()
        assert result.verdict == Verdict.PASS

    @patch("src.layer3_ai._call_github_models")
    def test_ai_failure_returns_skip(self, mock_call: MagicMock):
        mock_call.side_effect = Exception("API down")
        result = run_layer3(
            source_diffs={"src/new.py": "+ new_code()"},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/new.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.SKIP
        assert "API down" in result.details

    @patch("src.layer3_ai._call_github_models")
    def test_ai_failure_with_shortcuts_preserves_shortcut_verdicts(self, mock_call: MagicMock):
        mock_call.side_effect = Exception("API down")
        result = run_layer3(
            source_diffs={
                "src/trivial.py": "+ # comment",
                "src/ambiguous.py": "+ complex_code()",
            },
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/trivial.py": None, "src/ambiguous.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.PASS
        assert "API down" in result.details
        file_map = {fv.file: fv for fv in result.file_verdicts}
        assert "src/trivial.py" in file_map
        assert file_map["src/trivial.py"].verdict == Verdict.SKIP
        assert "src/ambiguous.py" in file_map
        assert file_map["src/ambiguous.py"].verdict == Verdict.SKIP
        assert "deferred" in file_map["src/ambiguous.py"].reason.lower()

    def test_empty_source_diffs_returns_pass(self):
        result = run_layer3(
            source_diffs={},
            deleted_files=set(),
            test_diffs={},
            l2_matched_tests={},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.PASS

    @patch("src.layer3_ai._call_github_models")
    def test_mixed_shortcut_and_ai_fail(self, mock_call: MagicMock):
        mock_call.return_value = json.dumps({
            "verdict": "fail",
            "confidence": 0.9,
            "files": [{"file": "src/logic.py", "verdict": "fail", "reason": "No tests"}],
        })
        result = run_layer3(
            source_diffs={
                "src/trivial.py": "+ # comment",
                "src/logic.py": "+ complex_logic()",
            },
            deleted_files=set(),
            test_diffs={"tests/test_other.py": "+ def test_other(): ..."},
            l2_matched_tests={"src/trivial.py": None, "src/logic.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.FAIL

    @patch("src.layer3_ai._call_github_models")
    def test_gate4_fail_without_ai(self, mock_call: MagicMock):
        result = run_layer3(
            source_diffs={"src/billing.py": "+ bill()"},
            deleted_files=set(),
            test_diffs={},
            l2_matched_tests={"src/billing.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        mock_call.assert_not_called()
        assert result.verdict == Verdict.FAIL

    @patch("src.layer3_ai._call_github_models")
    def test_low_confidence_ai_downgrades_to_warning(self, mock_call: MagicMock):
        mock_call.return_value = json.dumps({
            "verdict": "fail",
            "confidence": 0.4,
            "files": [{"file": "src/new.py", "verdict": "fail", "reason": "Maybe"}],
        })
        result = run_layer3(
            source_diffs={"src/new.py": "+ new_code()"},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/new.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.WARNING

    @patch("src.layer3_ai._call_github_models")
    def test_coverage_ok_shortcut_passes(self, mock_call: MagicMock):
        result = run_layer3(
            source_diffs={"src/user.py": "+ real_code()"},
            deleted_files=set(),
            test_diffs={},
            l2_matched_tests={"src/user.py": None},
            coverage_details={"src/user.py": 92.0},
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        mock_call.assert_not_called()
        assert result.verdict == Verdict.PASS


class TestCallGithubModels:
    @patch("src.layer3_ai.OpenAI")
    def test_empty_choices_returns_empty_string(self, mock_openai: MagicMock):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        raw = _call_github_models(
            model="openai/gpt-5-mini",
            system_prompt="system",
            user_prompt="user",
            token="ghp_fake",
        )

        assert raw == ""


class TestVerdictSchema:
    def test_verdict_schema_is_module_level_and_has_expected_shape(self):
        schema = layer3_ai._VERDICT_SCHEMA
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert schema["required"] == ["verdict", "confidence", "files"]


class TestTestRelevanceEnum:
    def test_values(self):
        assert Relevance.YES.value == "yes"
        assert Relevance.NO.value == "no"
        assert Relevance.UNKNOWN.value == "unknown"

    def test_membership(self):
        assert len(Relevance) == 3


class TestTestRelevanceFunction:
    def test_yes_when_l2_matched(self):
        result = compute_test_relevance(
            source_file="src/auth.py",
            changed_test_files=["tests/test_auth.py"],
            l2_matched_test="tests/test_auth.py",
            test_diffs={"tests/test_auth.py": "+ def test_login(): ..."},
        )
        assert result == Relevance.YES

    def test_yes_when_test_name_contains_source_stem(self):
        result = compute_test_relevance(
            source_file="src/payment.py",
            changed_test_files=["tests/test_payment_flow.py"],
            l2_matched_test=None,
            test_diffs={"tests/test_payment_flow.py": "+ def test_pay(): ..."},
        )
        assert result == Relevance.YES

    def test_yes_when_test_diff_mentions_source_stem(self):
        result = compute_test_relevance(
            source_file="src/validator.py",
            changed_test_files=["tests/test_helpers.py"],
            l2_matched_test=None,
            test_diffs={
                "tests/test_helpers.py": "+ from src.validator import Validator\n+ v = Validator()"
            },
        )
        assert result == Relevance.YES

    def test_no_when_no_test_files_changed(self):
        result = compute_test_relevance(
            source_file="src/billing.py",
            changed_test_files=[],
            l2_matched_test=None,
            test_diffs={},
        )
        assert result == Relevance.NO

    def test_unknown_when_tests_changed_but_none_matched(self):
        result = compute_test_relevance(
            source_file="src/auth.py",
            changed_test_files=["tests/test_csv.py"],
            l2_matched_test=None,
            test_diffs={"tests/test_csv.py": "+ def test_csv_export(): ..."},
        )
        assert result == Relevance.UNKNOWN

    def test_yes_stem_match_case_insensitive(self):
        result = compute_test_relevance(
            source_file="src/UserService.py",
            changed_test_files=["tests/test_userservice.py"],
            l2_matched_test=None,
            test_diffs={"tests/test_userservice.py": "+ ..."},
        )
        assert result == Relevance.YES

    def test_yes_l2_match_trumps_no_stem_match(self):
        result = compute_test_relevance(
            source_file="src/auth.py",
            changed_test_files=["tests/integration/e2e_login.py"],
            l2_matched_test="tests/integration/e2e_login.py",
            test_diffs={"tests/integration/e2e_login.py": "+ ..."},
        )
        assert result == Relevance.YES

    def test_no_when_changed_test_files_empty_even_with_l2_match_not_changed(self):
        result = compute_test_relevance(
            source_file="src/auth.py",
            changed_test_files=[],
            l2_matched_test="tests/test_auth.py",
            test_diffs={},
        )
        assert result == Relevance.NO


class TestIsTrivialDiff:
    def test_whitespace_only_is_trivial(self):
        diff = "+   \n+\n-  \n- "
        assert is_trivial_diff(diff) is True

    def test_comment_lines_are_trivial(self):
        diff = (
            "+ # add a comment\n- // old comment\n+ /* block comment */"
            "\n+ * middle\n+ -- sql comment"
        )
        assert is_trivial_diff(diff) is True

    def test_mixed_whitespace_and_comments_is_trivial(self):
        diff = "+\n+ # comment\n-   \n- // removed comment"
        assert is_trivial_diff(diff) is True

    def test_real_code_is_not_trivial(self):
        diff = "+ def new_function():\n+     return True"
        assert is_trivial_diff(diff) is False

    def test_import_is_not_trivial(self):
        diff = "+ import os"
        assert is_trivial_diff(diff) is False

    def test_python_from_import_is_not_trivial(self):
        diff = "+ from .signals import *"
        assert is_trivial_diff(diff) is False

    def test_js_require_is_not_trivial(self):
        diff = '+ const x = require("./polyfills")'
        assert is_trivial_diff(diff) is False

    def test_php_include_is_not_trivial(self):
        diff = '+ include "bootstrap.php";'
        assert is_trivial_diff(diff) is False

    def test_cpp_include_is_not_trivial(self):
        diff = '+ #include <stdio.h>'
        assert is_trivial_diff(diff) is False

    def test_empty_diff_is_trivial(self):
        assert is_trivial_diff("") is True

    def test_context_lines_ignored(self):
        diff = " unchanged line\n+ # just a comment\n  another context"
        assert is_trivial_diff(diff) is True

    def test_mixed_trivial_and_code_is_not_trivial(self):
        diff = "+ # comment\n+ real_code = True"
        assert is_trivial_diff(diff) is False

    def test_triple_slash_comment_is_trivial(self):
        diff = "+ /// doc comment\n- /// old doc"
        assert is_trivial_diff(diff) is True

    def test_javadoc_comment_is_trivial(self):
        diff = "+ /** start\n+  * middle\n+  */ end"
        assert is_trivial_diff(diff) is True

    def test_use_statement_is_not_trivial(self):
        diff = "+ use App\\Models\\User;"
        assert is_trivial_diff(diff) is False


class TestEvaluateFileShortcut:
    def test_gate1_deleted_file_returns_skip(self):
        result = evaluate_file_shortcut(
            source_file="src/old.py",
            diff="+ anything",
            is_deleted=True,
            coverage_details={"src/old.py": 100.0},
            coverage_threshold=80.0,
            test_relevance=Relevance.YES,
        )
        assert result == Verdict.SKIP

    def test_gate2_trivial_diff_returns_skip(self):
        result = evaluate_file_shortcut(
            source_file="src/app.py",
            diff="+ # just a comment",
            is_deleted=False,
            coverage_details=None,
            coverage_threshold=80.0,
            test_relevance=Relevance.NO,
        )
        assert result == Verdict.SKIP

    def test_gate3_coverage_ok_with_yes_relevance_returns_pass(self):
        result = evaluate_file_shortcut(
            source_file="src/user.py",
            diff="+ real_code()",
            is_deleted=False,
            coverage_details={"src/user.py": 92.0},
            coverage_threshold=80.0,
            test_relevance=Relevance.YES,
        )
        assert result == Verdict.PASS

    def test_gate3_coverage_ok_with_no_relevance_returns_pass(self):
        result = evaluate_file_shortcut(
            source_file="src/user.py",
            diff="+ real_code()",
            is_deleted=False,
            coverage_details={"src/user.py": 85.0},
            coverage_threshold=80.0,
            test_relevance=Relevance.NO,
        )
        assert result == Verdict.PASS

    def test_gate3_coverage_at_exact_threshold_returns_pass(self):
        result = evaluate_file_shortcut(
            source_file="src/user.py",
            diff="+ real_code()",
            is_deleted=False,
            coverage_details={"src/user.py": 80.0},
            coverage_threshold=80.0,
            test_relevance=Relevance.UNKNOWN,
        )
        assert result == Verdict.PASS

    def test_gate4_no_tests_no_coverage_returns_fail(self):
        result = evaluate_file_shortcut(
            source_file="src/billing.py",
            diff="+ new_logic()",
            is_deleted=False,
            coverage_details=None,
            coverage_threshold=80.0,
            test_relevance=Relevance.NO,
        )
        assert result == Verdict.FAIL

    def test_gate4_no_tests_with_low_coverage_returns_fail(self):
        result = evaluate_file_shortcut(
            source_file="src/billing.py",
            diff="+ new_logic()",
            is_deleted=False,
            coverage_details={"src/billing.py": 18.0},
            coverage_threshold=80.0,
            test_relevance=Relevance.NO,
        )
        assert result == Verdict.FAIL

    def test_gate5_low_coverage_yes_relevance_returns_fail(self):
        result = evaluate_file_shortcut(
            source_file="src/auth.py",
            diff="+ auth_code()",
            is_deleted=False,
            coverage_details={"src/auth.py": 40.0},
            coverage_threshold=80.0,
            test_relevance=Relevance.YES,
        )
        assert result == Verdict.FAIL

    def test_gate6_low_coverage_unknown_relevance_returns_none(self):
        result = evaluate_file_shortcut(
            source_file="src/auth.py",
            diff="+ auth_code()",
            is_deleted=False,
            coverage_details={"src/auth.py": 40.0},
            coverage_threshold=80.0,
            test_relevance=Relevance.UNKNOWN,
        )
        assert result is None

    def test_gate7_no_coverage_yes_relevance_returns_none(self):
        result = evaluate_file_shortcut(
            source_file="src/new.py",
            diff="+ new_code()",
            is_deleted=False,
            coverage_details=None,
            coverage_threshold=80.0,
            test_relevance=Relevance.YES,
        )
        assert result is None

    def test_gate8_no_coverage_unknown_relevance_returns_none(self):
        result = evaluate_file_shortcut(
            source_file="src/new.py",
            diff="+ new_code()",
            is_deleted=False,
            coverage_details=None,
            coverage_threshold=80.0,
            test_relevance=Relevance.UNKNOWN,
        )
        assert result is None

    def test_file_absent_from_coverage_details_has_no_coverage(self):
        result = evaluate_file_shortcut(
            source_file="src/new_file.py",
            diff="+ code()",
            is_deleted=False,
            coverage_details={"src/other.py": 95.0},
            coverage_threshold=80.0,
            test_relevance=Relevance.YES,
        )
        assert result is None

    def test_gate1_takes_priority_over_gate2(self):
        result = evaluate_file_shortcut(
            source_file="src/old.py",
            diff="+ # trivial comment",
            is_deleted=True,
            coverage_details=None,
            coverage_threshold=80.0,
            test_relevance=Relevance.NO,
        )
        assert result == Verdict.SKIP

    def test_gate2_takes_priority_over_gate3(self):
        result = evaluate_file_shortcut(
            source_file="src/style.py",
            diff="+ # just reformatted",
            is_deleted=False,
            coverage_details={"src/style.py": 100.0},
            coverage_threshold=80.0,
            test_relevance=Relevance.YES,
        )
        assert result == Verdict.SKIP

    def test_empty_coverage_details_dict_means_no_coverage(self):
        result = evaluate_file_shortcut(
            source_file="src/app.py",
            diff="+ code()",
            is_deleted=False,
            coverage_details={},
            coverage_threshold=80.0,
            test_relevance=Relevance.UNKNOWN,
        )
        assert result is None


class TestBuildAiPrompt:
    def test_coverage_summary_with_data(self):
        prompt = _build_ai_prompt(
            files_for_ai=["src/validator.py"],
            source_diffs={"src/validator.py": "+ validate()"},
            test_diffs={},
            coverage_details={"src/validator.py": 65.0},
            coverage_threshold=80.0,
            matched_tests={"src/validator.py": None},
        )
        assert "65" in prompt
        assert "80" in prompt
        assert "src/validator.py" in prompt

    def test_coverage_summary_no_data(self):
        prompt = _build_ai_prompt(
            files_for_ai=["src/new.py"],
            source_diffs={"src/new.py": "+ new()"},
            test_diffs={},
            coverage_details=None,
            coverage_threshold=80.0,
            matched_tests={"src/new.py": None},
        )
        assert "no coverage data" in prompt.lower()

    def test_source_diffs_included(self):
        prompt = _build_ai_prompt(
            files_for_ai=["src/auth.py"],
            source_diffs={"src/auth.py": "+ def login(): pass"},
            test_diffs={},
            coverage_details=None,
            coverage_threshold=80.0,
            matched_tests={"src/auth.py": None},
        )
        assert "src/auth.py" in prompt
        assert "def login()" in prompt

    def test_matched_test_annotated(self):
        prompt = _build_ai_prompt(
            files_for_ai=["src/validator.py"],
            source_diffs={"src/validator.py": "+ validate()"},
            test_diffs={
                "tests/test_validator.py": "+ def test_validate(): ...",
                "tests/test_helpers.py": "+ def test_help(): ...",
            },
            coverage_details=None,
            coverage_threshold=80.0,
            matched_tests={"src/validator.py": "tests/test_validator.py"},
        )
        assert "matched" in prompt.lower()
        assert "test_validator.py" in prompt
        assert "test_helpers.py" in prompt

    def test_candidate_annotation_for_unmatched_tests(self):
        prompt = _build_ai_prompt(
            files_for_ai=["src/auth.py"],
            source_diffs={"src/auth.py": "+ login()"},
            test_diffs={"tests/test_csv.py": "+ def test_export(): ..."},
            coverage_details=None,
            coverage_threshold=80.0,
            matched_tests={"src/auth.py": None},
        )
        assert "candidate" in prompt.lower()
        assert "test_csv.py" in prompt

    def test_deduplicates_shared_test_files(self):
        prompt = _build_ai_prompt(
            files_for_ai=["src/validator.py", "src/parser.py"],
            source_diffs={
                "src/validator.py": "+ validate()",
                "src/parser.py": "+ parse()",
            },
            test_diffs={
                "tests/test_helpers.py": "+ shared helpers",
            },
            coverage_details=None,
            coverage_threshold=80.0,
            matched_tests={
                "src/validator.py": None,
                "src/parser.py": None,
            },
        )
        # test_helpers.py diff should appear once, not duplicated
        assert prompt.count("shared helpers") == 1

    def test_sanitizes_source_diffs(self):
        prompt = _build_ai_prompt(
            files_for_ai=["src/auth.py"],
            source_diffs={"src/auth.py": "+ SYSTEM: evil injection\n+ real code"},
            test_diffs={},
            coverage_details=None,
            coverage_threshold=80.0,
            matched_tests={"src/auth.py": None},
        )
        assert "[REDACTED]" in prompt
        assert "SYSTEM: evil injection" not in prompt

    def test_empty_files_for_ai_returns_empty(self):
        prompt = _build_ai_prompt(
            files_for_ai=[],
            source_diffs={},
            test_diffs={},
            coverage_details=None,
            coverage_threshold=80.0,
            matched_tests={},
        )
        assert prompt == ""

    def test_file_absent_from_coverage_shows_no_data(self):
        prompt = _build_ai_prompt(
            files_for_ai=["src/new.py"],
            source_diffs={"src/new.py": "+ code()"},
            test_diffs={},
            coverage_details={"src/other.py": 95.0},
            coverage_threshold=80.0,
            matched_tests={"src/new.py": None},
        )
        assert "no coverage data" in prompt.lower()

    def test_multiple_source_files_with_coverage(self):
        prompt = _build_ai_prompt(
            files_for_ai=["src/a.py", "src/b.py"],
            source_diffs={"src/a.py": "+ a()", "src/b.py": "+ b()"},
            test_diffs={},
            coverage_details={"src/a.py": 55.0, "src/b.py": 40.0},
            coverage_threshold=80.0,
            matched_tests={"src/a.py": None, "src/b.py": None},
        )
        assert "src/a.py" in prompt
        assert "src/b.py" in prompt
        assert "55" in prompt
        assert "40" in prompt

    def test_max_diff_chars_truncates_source_and_test(self):
        long_diff = "x" * 20_000
        prompt = _build_ai_prompt(
            files_for_ai=["src/a.py"],
            source_diffs={"src/a.py": long_diff},
            test_diffs={"tests/test_a.py": long_diff},
            coverage_details=None,
            coverage_threshold=80.0,
            matched_tests={"src/a.py": "tests/test_a.py"},
            max_diff_chars=100,
        )
        assert prompt.count("...[truncated]") == 2
        assert len(prompt) < 5000


class TestIntegrationWorkedExamples:
    """Integration tests matching §4 worked examples from TODO.md."""

    def test_example1_mixed_pr_per_file_evaluation(self):
        """Ex1: coverage pass + no-test fail + trivial skip → overall FAIL."""
        result = run_layer3(
            source_diffs={
                "src/user.py": "+ def get_user(): return user",
                "src/billing.py": "+ def charge(amount): process(amount)",
                "src/readme_fix.py": "+   ",
            },
            deleted_files=set(),
            test_diffs={},
            l2_matched_tests={
                "src/user.py": "tests/test_user.py",
                "src/billing.py": None,
                "src/readme_fix.py": None,
            },
            coverage_details={"src/user.py": 92.0, "src/billing.py": 18.0},
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.FAIL

    def test_example2_new_file_absent_from_src_stats(self):
        """Ex2: new file (no coverage, no tests) → FAIL; existing file covered → PASS."""
        result = run_layer3(
            source_diffs={
                "src/new_feature.py": "+ class Feature: pass",
                "src/existing.py": "+ x = 1",
            },
            deleted_files=set(),
            test_diffs={},
            l2_matched_tests={"src/new_feature.py": None, "src/existing.py": None},
            coverage_details={"src/existing.py": 95.0},
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.FAIL

    def test_example3_deleted_file_plus_covered_change(self):
        """Ex3: deleted → SKIP, covered change → PASS → overall PASS."""
        result = run_layer3(
            source_diffs={
                "src/legacy.py": "- def old(): ...",
                "src/auth.py": "+ def login(): ...",
            },
            deleted_files={"src/legacy.py"},
            test_diffs={},
            l2_matched_tests={"src/legacy.py": None, "src/auth.py": None},
            coverage_details={"src/auth.py": 88.0},
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.PASS

    @patch("src.layer3_ai._call_github_models")
    def test_example4_unrelated_test_ai_judges_fail(self, mock_ai):
        """Ex4: no coverage + unknown relevance → AI judges; AI says FAIL."""
        mock_ai.return_value = json.dumps({
            "verdict": "fail",
            "confidence": 0.8,
            "files": [
                {"file": "src/auth.py", "verdict": "fail", "reason": "test_csv.py is unrelated"}
            ],
        })
        result = run_layer3(
            source_diffs={"src/auth.py": "+ def authenticate(): ..."},
            deleted_files=set(),
            test_diffs={"tests/test_csv.py": "+ def test_csv_parse(): ..."},
            l2_matched_tests={"src/auth.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.FAIL
        mock_ai.assert_called_once()

    @patch("src.layer3_ai._call_github_models")
    def test_example5_ai_fallthrough_with_relevant_tests(self, mock_ai):
        """Ex5: no coverage + YES relevance → AI judges adequacy."""
        mock_ai.return_value = json.dumps({
            "verdict": "pass",
            "confidence": 0.9,
            "files": [
                {"file": "src/payment.py", "verdict": "pass", "reason": "test covers payment logic"}
            ],
        })
        result = run_layer3(
            source_diffs={"src/payment.py": "+ def charge(card): ..."},
            deleted_files=set(),
            test_diffs={"tests/test_payment_flow.py": "+ def test_charge(): ..."},
            l2_matched_tests={"src/payment.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.PASS
        mock_ai.assert_called_once()

    @patch("src.layer3_ai._call_github_models")
    def test_example6_ai_api_failure(self, mock_ai):
        """Ex6: AI fails → execution_status=ERROR → verdict=SKIP."""
        mock_ai.side_effect = RuntimeError("HTTP 500")
        result = run_layer3(
            source_diffs={"src/parser.py": "+ def parse(data): ..."},
            deleted_files=set(),
            test_diffs={"tests/test_parser_edge.py": "+ def test_edge(): ..."},
            l2_matched_tests={"src/parser.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.SKIP
        assert "HTTP 500" in result.details

    @patch("src.layer3_ai._call_github_models")
    def test_example7_unknown_relevance_below_threshold_ai_judges(self, mock_ai):
        """Ex7: coverage below threshold + UNKNOWN relevance → AI judges."""
        mock_ai.return_value = json.dumps({
            "verdict": "warning",
            "confidence": 0.6,
            "files": [
                {
                    "file": "src/validator.py",
                    "verdict": "warning",
                    "reason": "test_helpers partially covers",
                }
            ],
        })
        result = run_layer3(
            source_diffs={"src/validator.py": "+ def validate(x): ..."},
            deleted_files=set(),
            test_diffs={"tests/test_helpers.py": "+ def test_helper(): ..."},
            l2_matched_tests={"src/validator.py": None},
            coverage_details={"src/validator.py": 55.0},
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.WARNING
        mock_ai.assert_called_once()

    @patch("src.layer3_ai._call_github_models")
    def test_example8_ai_failure_after_unknown_relevance(self, mock_ai):
        """Ex8: no coverage + UNKNOWN → AI, but AI fails → SKIP."""
        mock_ai.side_effect = TimeoutError("timeout")
        result = run_layer3(
            source_diffs={"src/cache.py": "+ def evict(key): ..."},
            deleted_files=set(),
            test_diffs={"tests/test_storage.py": "+ def test_store(): ..."},
            l2_matched_tests={"src/cache.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.SKIP

    def test_example9_all_trivial_or_deleted(self):
        """Ex9: deleted + trivial → all SKIP → overall PASS (execution_status=OK)."""
        result = run_layer3(
            source_diffs={
                "src/legacy.py": "- def old(): ...",
                "src/utils.py": "+   ",
            },
            deleted_files={"src/legacy.py"},
            test_diffs={},
            l2_matched_tests={"src/legacy.py": None, "src/utils.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.PASS


# ---------------------------------------------------------------------------
# Smart batching & model fallback tests
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string_returns_one(self):
        assert _estimate_tokens("") == 1

    def test_four_chars_returns_two(self):
        assert _estimate_tokens("abcd") == 2

    def test_eight_chars_returns_three(self):
        assert _estimate_tokens("abcdefgh") == 3

    def test_single_char_returns_one(self):
        assert _estimate_tokens("x") == 1

    def test_large_text(self):
        assert _estimate_tokens("a" * 4000) == 1334  # 4000//3 + 1


class TestEstimateFileCost:
    def test_source_only_no_matched_test(self):
        cost = _estimate_file_cost(
            "src/a.py",
            source_diffs={"src/a.py": "x" * 100},
            test_diffs={},
            matched_tests={"src/a.py": None},
        )
        expected = 25 + _estimate_tokens(_sanitize_diff("x" * 100))
        assert cost == expected

    def test_source_with_matched_test_in_test_diffs(self):
        cost = _estimate_file_cost(
            "src/a.py",
            source_diffs={"src/a.py": "x" * 100},
            test_diffs={"tests/test_a.py": "y" * 200},
            matched_tests={"src/a.py": "tests/test_a.py"},
        )
        src_tokens = _estimate_tokens(_sanitize_diff("x" * 100))
        test_tokens = _estimate_tokens(_sanitize_diff("y" * 200))
        assert cost == 25 + src_tokens + 25 + test_tokens

    def test_matched_test_not_in_test_diffs_ignored(self):
        cost = _estimate_file_cost(
            "src/a.py",
            source_diffs={"src/a.py": "x" * 100},
            test_diffs={},
            matched_tests={"src/a.py": "tests/test_a.py"},
        )
        expected = 25 + _estimate_tokens(_sanitize_diff("x" * 100))
        assert cost == expected

    def test_respects_max_diff_chars(self):
        cost_default = _estimate_file_cost(
            "src/a.py",
            source_diffs={"src/a.py": "x" * 20_000},
            test_diffs={},
            matched_tests={"src/a.py": None},
        )
        cost_tight = _estimate_file_cost(
            "src/a.py",
            source_diffs={"src/a.py": "x" * 20_000},
            test_diffs={},
            matched_tests={"src/a.py": None},
            max_diff_chars=3000,
        )
        assert cost_tight < cost_default


class TestFilterTestDiffsForBatch:
    def test_matched_test_in_batch_included(self):
        result = _filter_test_diffs_for_batch(
            batch_files=["src/a.py"],
            test_diffs={"tests/test_a.py": "diff_a"},
            matched_tests={"src/a.py": "tests/test_a.py"},
        )
        assert "tests/test_a.py" in result

    def test_matched_test_outside_batch_included_as_candidate(self):
        result = _filter_test_diffs_for_batch(
            batch_files=["src/a.py"],
            test_diffs={"tests/test_b.py": "diff_b"},
            matched_tests={"src/a.py": None, "src/b.py": "tests/test_b.py"},
        )
        assert "tests/test_b.py" in result

    def test_shared_test_matched_outside_batch_still_included(self):
        """BUG 4: A test matched to an out-of-batch file should still appear
        as a candidate so the AI can judge relevance."""
        result = _filter_test_diffs_for_batch(
            batch_files=["src/auth.py"],
            test_diffs={
                "conftest.py": "diff_conftest",
                "tests/test_auth.py": "diff_auth",
            },
            matched_tests={
                "src/auth.py": "tests/test_auth.py",
                "src/config.py": "conftest.py",
            },
        )
        assert "tests/test_auth.py" in result
        assert "conftest.py" in result

    def test_unmatched_candidate_included(self):
        result = _filter_test_diffs_for_batch(
            batch_files=["src/a.py"],
            test_diffs={"tests/test_helpers.py": "diff_h"},
            matched_tests={"src/a.py": None},
        )
        assert "tests/test_helpers.py" in result

    def test_combination_includes_all_test_diffs(self):
        result = _filter_test_diffs_for_batch(
            batch_files=["src/a.py"],
            test_diffs={
                "tests/test_a.py": "diff_a",
                "tests/test_b.py": "diff_b",
                "tests/test_utils.py": "diff_u",
            },
            matched_tests={
                "src/a.py": "tests/test_a.py",
                "src/b.py": "tests/test_b.py",
            },
        )
        assert "tests/test_a.py" in result
        assert "tests/test_b.py" in result
        assert "tests/test_utils.py" in result

    def test_empty_batch_includes_all_as_candidates(self):
        result = _filter_test_diffs_for_batch(
            batch_files=[],
            test_diffs={
                "tests/test_a.py": "diff_a",
                "tests/test_unmatched.py": "diff_u",
            },
            matched_tests={"src/a.py": "tests/test_a.py"},
        )
        assert "tests/test_a.py" in result
        assert "tests/test_unmatched.py" in result


class TestBatchFiles:
    def test_empty_input(self):
        assert _batch_files([], {}, {}, {}) == []

    def test_single_file_fits_one_batch(self):
        batches = _batch_files(
            ["src/a.py"],
            source_diffs={"src/a.py": "small diff"},
            test_diffs={},
            matched_tests={"src/a.py": None},
            token_budget=5000,
        )
        assert batches == [["src/a.py"]]

    def test_multiple_small_files_pack_into_one_batch(self):
        batches = _batch_files(
            ["src/a.py", "src/b.py"],
            source_diffs={"src/a.py": "diff_a", "src/b.py": "diff_b"},
            test_diffs={},
            matched_tests={"src/a.py": None, "src/b.py": None},
            token_budget=5000,
        )
        assert len(batches) == 1
        assert batches[0] == ["src/a.py", "src/b.py"]

    def test_large_files_split_into_multiple_batches(self):
        big_diff = "x" * 10_000
        batches = _batch_files(
            ["src/a.py", "src/b.py", "src/c.py"],
            source_diffs={
                "src/a.py": big_diff,
                "src/b.py": big_diff,
                "src/c.py": big_diff,
            },
            test_diffs={},
            matched_tests={
                "src/a.py": None,
                "src/b.py": None,
                "src/c.py": None,
            },
            token_budget=3000,
        )
        assert len(batches) > 1
        all_files = [f for batch in batches for f in batch]
        assert sorted(all_files) == ["src/a.py", "src/b.py", "src/c.py"]

    def test_oversized_single_file_gets_own_batch(self):
        huge_diff = "x" * 50_000
        batches = _batch_files(
            ["src/huge.py", "src/small.py"],
            source_diffs={"src/huge.py": huge_diff, "src/small.py": "tiny"},
            test_diffs={},
            matched_tests={"src/huge.py": None, "src/small.py": None},
            token_budget=2500,
        )
        assert len(batches) == 2
        assert batches[0] == ["src/huge.py"]
        assert batches[1] == ["src/small.py"]

    def test_candidate_tests_counted_in_overhead(self):
        big_candidate = "y" * 8000
        batches_with = _batch_files(
            ["src/a.py", "src/b.py"],
            source_diffs={"src/a.py": "diff_a", "src/b.py": "diff_b"},
            test_diffs={"tests/test_unmatched.py": big_candidate},
            matched_tests={"src/a.py": None, "src/b.py": None},
            token_budget=3000,
        )
        batches_without = _batch_files(
            ["src/a.py", "src/b.py"],
            source_diffs={"src/a.py": "diff_a", "src/b.py": "diff_b"},
            test_diffs={},
            matched_tests={"src/a.py": None, "src/b.py": None},
            token_budget=3000,
        )
        assert len(batches_with) >= len(batches_without)


class TestIsRetryableSizeError:
    def test_413_with_too_large_message(self):
        exc = _make_api_error(413, "Request body too large for model")
        assert _is_retryable_size_error(exc) is True

    def test_400_with_too_large_message(self):
        exc = _make_api_error(400, "Request body too large for model")
        assert _is_retryable_size_error(exc) is True

    def test_413_without_too_large_message(self):
        exc = _make_api_error(413, "Unknown server error")
        assert _is_retryable_size_error(exc) is False

    def test_400_context_length_exceeded(self):
        exc = _make_api_error(400, "context length exceeded")
        assert _is_retryable_size_error(exc) is True

    def test_400_maximum_context_length(self):
        exc = _make_api_error(
            400, "maximum context length is 8192 tokens"
        )
        assert _is_retryable_size_error(exc) is True

    def test_413_content_too_large(self):
        exc = _make_api_error(413, "Content Too Large")
        assert _is_retryable_size_error(exc) is True

    def test_500_error(self):
        exc = _make_api_error(500, "Internal server error")
        assert _is_retryable_size_error(exc) is False

    def test_regular_exception(self):
        assert _is_retryable_size_error(RuntimeError("connection failed")) is False


class TestIsModelForbidden:
    def test_403_is_forbidden(self):
        assert _is_model_forbidden(_make_api_error(403, "Forbidden")) is True

    def test_401_is_not_forbidden(self):
        assert _is_model_forbidden(_make_api_error(401, "Unauthorized")) is False

    def test_regular_exception(self):
        assert _is_model_forbidden(RuntimeError("timeout")) is False


class TestResolveModels:
    def test_default_model_returns_full_chain(self):
        models = _resolve_models("openai/gpt-4.1-mini")
        assert models == ["openai/gpt-4.1-mini", "openai/gpt-4.1-nano"]

    def test_custom_model_returns_single(self):
        assert _resolve_models("openai/gpt-5-mini") == ["openai/gpt-5-mini"]

    def test_nano_alone_returns_single(self):
        assert _resolve_models("openai/gpt-4.1-nano") == ["openai/gpt-4.1-nano"]


class TestCallAiForBatch:
    @patch("src.layer3_ai._call_github_models")
    def test_success_on_first_try(self, mock_call: MagicMock):
        mock_call.return_value = '{"verdict":"pass","confidence":0.9,"files":[]}'
        raw, exc = _call_ai_for_batch(
            batch_files=["src/a.py"],
            source_diffs={"src/a.py": "diff"},
            test_diffs={},
            coverage_details=None,
            coverage_threshold=80.0,
            matched_tests={"src/a.py": None},
            model="openai/gpt-4.1-mini",
            system_prompt="system",
            token="ghp_fake",
        )
        assert raw is not None
        assert exc is None
        mock_call.assert_called_once()

    @patch("src.layer3_ai._call_github_models")
    def test_413_retries_with_tighter_truncation(self, mock_call: MagicMock):
        error_413 = _make_api_error(413, "Request body too large for model")
        mock_call.side_effect = [
            error_413,
            '{"verdict":"pass","confidence":0.9,"files":[]}',
        ]
        raw, exc = _call_ai_for_batch(
            batch_files=["src/a.py"],
            source_diffs={"src/a.py": "x" * 20_000},
            test_diffs={},
            coverage_details=None,
            coverage_threshold=80.0,
            matched_tests={"src/a.py": None},
            model="openai/gpt-4.1-mini",
            system_prompt="system",
            token="ghp_fake",
        )
        assert raw is not None
        assert exc is None
        assert mock_call.call_count == 2

    @patch("src.layer3_ai._call_github_models")
    def test_413_retry_also_fails(self, mock_call: MagicMock):
        error_413 = _make_api_error(413, "Request body too large for model")
        mock_call.side_effect = [error_413, error_413]
        raw, exc = _call_ai_for_batch(
            batch_files=["src/a.py"],
            source_diffs={"src/a.py": "x" * 20_000},
            test_diffs={},
            coverage_details=None,
            coverage_threshold=80.0,
            matched_tests={"src/a.py": None},
            model="openai/gpt-4.1-mini",
            system_prompt="system",
            token="ghp_fake",
        )
        assert raw is None
        assert exc is not None

    @patch("src.layer3_ai._call_github_models")
    def test_non_retryable_error_returns_immediately(self, mock_call: MagicMock):
        mock_call.side_effect = RuntimeError("connection lost")
        raw, exc = _call_ai_for_batch(
            batch_files=["src/a.py"],
            source_diffs={"src/a.py": "diff"},
            test_diffs={},
            coverage_details=None,
            coverage_threshold=80.0,
            matched_tests={"src/a.py": None},
            model="openai/gpt-4.1-mini",
            system_prompt="system",
            token="ghp_fake",
        )
        assert raw is None
        assert exc is not None
        assert "connection lost" in str(exc)
        mock_call.assert_called_once()

    @patch("src.layer3_ai._call_github_models")
    def test_includes_all_test_diffs_in_batch_prompt(self, mock_call: MagicMock):
        mock_call.return_value = '{"verdict":"pass","confidence":0.9,"files":[]}'
        _call_ai_for_batch(
            batch_files=["src/a.py"],
            source_diffs={"src/a.py": "diff"},
            test_diffs={
                "tests/test_a.py": "matched_diff",
                "tests/test_b.py": "outside_batch_diff",
            },
            coverage_details=None,
            coverage_threshold=80.0,
            matched_tests={
                "src/a.py": "tests/test_a.py",
                "src/b.py": "tests/test_b.py",
            },
            model="openai/gpt-4.1-mini",
            system_prompt="system",
            token="ghp_fake",
        )
        user_prompt = mock_call.call_args[0][2]
        assert "matched_diff" in user_prompt
        assert "outside_batch_diff" in user_prompt


class TestPromptConciseInstruction:
    def test_prompt_concise_instruction_present(self):
        """Verify that prompts/test_adequacy.txt contains the 15-word concise instruction."""
        with open("prompts/test_adequacy.txt", "r") as f:
            content = f.read()
        assert "15 words" in content, "Prompt must contain '15 words' instruction for concise reasons"


class TestRunLayer3Batching:
    @patch("src.layer3_ai._call_github_models")
    def test_403_triggers_model_fallback(self, mock_call: MagicMock):
        error_403 = _make_api_error(403, "Forbidden")
        mock_call.side_effect = [
            error_403,
            json.dumps({
                "verdict": "pass",
                "confidence": 0.9,
                "files": [{"file": "src/new.py", "verdict": "pass", "reason": "OK"}],
            }),
        ]
        result = run_layer3(
            source_diffs={"src/new.py": "+ new_code()"},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/new.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-4.1-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.PASS
        assert mock_call.call_count == 2
        assert mock_call.call_args_list[0][0][0] == "openai/gpt-4.1-mini"
        assert mock_call.call_args_list[1][0][0] == "openai/gpt-4.1-nano"

    @patch("src.layer3_ai._call_github_models")
    def test_403_no_fallback_for_custom_model(self, mock_call: MagicMock):
        mock_call.side_effect = _make_api_error(403, "Forbidden")
        result = run_layer3(
            source_diffs={"src/new.py": "+ new_code()"},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/new.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.SKIP
        mock_call.assert_called_once()

    @patch("src.layer3_ai._call_github_models")
    def test_all_models_exhausted_returns_skip(self, mock_call: MagicMock):
        mock_call.side_effect = _make_api_error(403, "Forbidden")
        result = run_layer3(
            source_diffs={"src/new.py": "+ new_code()"},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/new.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-4.1-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.SKIP
        assert mock_call.call_count == 2

    @patch("src.layer3_ai._call_github_models")
    def test_batch_count_in_details_single(self, mock_call: MagicMock):
        mock_call.return_value = json.dumps({
            "verdict": "pass",
            "confidence": 0.9,
            "files": [{"file": "src/a.py", "verdict": "pass", "reason": "OK"}],
        })
        result = run_layer3(
            source_diffs={"src/a.py": "+ code()"},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/a.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-4.1-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert "1 batch" in result.details

    @patch("src.layer3_ai._call_github_models")
    @patch("src.layer3_ai._batch_files")
    def test_model_escalation_carries_across_batches(
        self, mock_batch: MagicMock, mock_call: MagicMock
    ):
        mock_batch.return_value = [["src/a.py"], ["src/b.py"]]
        error_403 = _make_api_error(403, "Forbidden")
        mock_call.side_effect = [
            error_403,
            json.dumps({
                "verdict": "pass",
                "confidence": 0.9,
                "files": [{"file": "src/a.py", "verdict": "pass", "reason": "OK"}],
            }),
            json.dumps({
                "verdict": "pass",
                "confidence": 0.9,
                "files": [{"file": "src/b.py", "verdict": "pass", "reason": "OK"}],
            }),
        ]
        result = run_layer3(
            source_diffs={"src/a.py": "+ code()", "src/b.py": "+ code()"},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/a.py": None, "src/b.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-4.1-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.PASS
        assert mock_call.call_count == 3
        assert mock_call.call_args_list[0][0][0] == "openai/gpt-4.1-mini"
        assert mock_call.call_args_list[1][0][0] == "openai/gpt-4.1-nano"
        assert mock_call.call_args_list[2][0][0] == "openai/gpt-4.1-nano"

    @patch("src.layer3_ai._call_github_models")
    @patch("src.layer3_ai._batch_files")
    def test_remaining_batches_skip_when_models_exhausted(
        self, mock_batch: MagicMock, mock_call: MagicMock
    ):
        mock_batch.return_value = [["src/a.py"], ["src/b.py"]]
        mock_call.side_effect = _make_api_error(403, "Forbidden")
        result = run_layer3(
            source_diffs={"src/a.py": "+ code()", "src/b.py": "+ code()"},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/a.py": None, "src/b.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-4.1-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.SKIP
        file_map = {fv.file: fv for fv in result.file_verdicts}
        assert file_map["src/a.py"].verdict == Verdict.SKIP
        assert file_map["src/b.py"].verdict == Verdict.SKIP
        assert "deferred" in file_map["src/b.py"].reason.lower()


class TestTokenBudgetConstants:
    """Pin the token budget constants to their corrected values."""

    def test_input_token_limit(self):
        assert layer3_ai._INPUT_TOKEN_LIMIT == 8000

    def test_chars_per_token(self):
        assert layer3_ai._CHARS_PER_TOKEN == 3

    def test_system_overhead_tokens(self):
        assert layer3_ai._SYSTEM_OVERHEAD_TOKENS == 800

    def test_safety_factor(self):
        assert layer3_ai._SAFETY_FACTOR == 0.85

    def test_user_prompt_token_budget(self):
        expected = int((8000 - 800) * 0.85)  # 6120
        assert layer3_ai._USER_PROMPT_TOKEN_BUDGET == expected

    def test_budget_leaves_headroom_for_system_prompt(self):
        total = layer3_ai._SYSTEM_OVERHEAD_TOKENS + layer3_ai._USER_PROMPT_TOKEN_BUDGET
        assert total < layer3_ai._INPUT_TOKEN_LIMIT


# ---------------------------------------------------------------------------
# BUG 1+2: AI response validation
# ---------------------------------------------------------------------------


class TestValidateBatchVerdicts:
    """_validate_batch_verdicts filters AI output against batch membership."""

    def test_keeps_verdicts_matching_batch(self):
        verdicts = [
            FileVerdict(file="src/a.py", verdict=Verdict.PASS, reason="ok", layer="layer3"),
            FileVerdict(file="src/b.py", verdict=Verdict.FAIL, reason="bad", layer="layer3"),
        ]
        kept = _validate_batch_verdicts(verdicts, ["src/a.py", "src/b.py"])
        assert len(kept) == 2

    def test_rejects_hallucinated_files(self):
        verdicts = [
            FileVerdict(file="src/a.py", verdict=Verdict.PASS, reason="ok", layer="layer3"),
            FileVerdict(file="src/FAKE.py", verdict=Verdict.FAIL, reason="bad", layer="layer3"),
        ]
        kept = _validate_batch_verdicts(verdicts, ["src/a.py"])
        assert len(kept) == 1
        assert kept[0].file == "src/a.py"

    def test_returns_none_when_batch_files_missing(self):
        verdicts = [
            FileVerdict(file="src/a.py", verdict=Verdict.PASS, reason="ok", layer="layer3"),
        ]
        result = _validate_batch_verdicts(verdicts, ["src/a.py", "src/b.py"])
        assert result is None

    def test_empty_verdicts_returns_none(self):
        result = _validate_batch_verdicts([], ["src/a.py"])
        assert result is None

    def test_all_hallucinated_returns_none(self):
        verdicts = [
            FileVerdict(file="src/FAKE.py", verdict=Verdict.FAIL, reason="bad", layer="layer3"),
        ]
        result = _validate_batch_verdicts(verdicts, ["src/a.py"])
        assert result is None


class TestRunLayer3EmptyAiResponse:
    """BUG 1: Empty AI response must not be treated as success."""

    @patch("src.layer3_ai._call_github_models")
    def test_empty_response_defers_to_fallback(self, mock_call: MagicMock):
        """When AI returns empty content, batch files should get SKIP and
        fall back to L1+L2, not silently disappear."""
        mock_call.return_value = ""
        result = run_layer3(
            source_diffs={"src/new.py": "+ new_code()"},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/new.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-4.1-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        file_map = {fv.file: fv for fv in result.file_verdicts}
        assert "src/new.py" in file_map
        assert file_map["src/new.py"].verdict == Verdict.SKIP


class TestRunLayer3HallucinatedFiles:
    """BUG 2: AI hallucinated files must not appear in final results."""

    @patch("src.layer3_ai._call_github_models")
    def test_hallucinated_file_not_in_results(self, mock_call: MagicMock):
        mock_call.return_value = json.dumps({
            "verdict": "fail",
            "confidence": 0.9,
            "files": [
                {"file": "src/a.py", "verdict": "pass", "reason": "ok"},
                {"file": "src/HALLUCINATED.py", "verdict": "fail", "reason": "fake"},
            ],
        })
        result = run_layer3(
            source_diffs={"src/a.py": "+ code()"},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/a.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-4.1-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        file_names = {fv.file for fv in result.file_verdicts}
        assert "src/HALLUCINATED.py" not in file_names
        assert "src/a.py" in file_names

    @patch("src.layer3_ai._call_github_models")
    def test_missing_batch_file_treated_as_failure(self, mock_call: MagicMock):
        """AI omits src/b.py from response — it should get SKIP, not vanish."""
        mock_call.return_value = json.dumps({
            "verdict": "pass",
            "confidence": 0.9,
            "files": [
                {"file": "src/a.py", "verdict": "pass", "reason": "ok"},
            ],
        })
        result = run_layer3(
            source_diffs={"src/a.py": "+ code()", "src/b.py": "+ more()"},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/a.py": None, "src/b.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-4.1-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        file_map = {fv.file: fv for fv in result.file_verdicts}
        assert "src/b.py" in file_map
        assert file_map["src/b.py"].verdict == Verdict.SKIP


# ---------------------------------------------------------------------------
# BUG 3: Graceful degradation on prompt/parse failures
# ---------------------------------------------------------------------------


class TestRunLayer3PromptFileMissing:
    @patch("src.layer3_ai._PROMPT_PATH")
    def test_missing_prompt_degrades_to_skip(self, mock_path: MagicMock):
        mock_path.read_text.side_effect = FileNotFoundError("no such file")
        result = run_layer3(
            source_diffs={"src/a.py": "+ code()"},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/a.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-4.1-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        file_map = {fv.file: fv for fv in result.file_verdicts}
        assert "src/a.py" in file_map
        assert file_map["src/a.py"].verdict == Verdict.SKIP


class TestRunLayer3ParseFailure:
    @patch("src.layer3_ai._call_github_models")
    def test_schema_drift_degrades_to_skip(self, mock_call: MagicMock):
        mock_call.return_value = '{"verdict": "pass"}'
        result = run_layer3(
            source_diffs={"src/a.py": "+ code()"},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/a.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-4.1-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        file_map = {fv.file: fv for fv in result.file_verdicts}
        assert "src/a.py" in file_map
        assert file_map["src/a.py"].verdict == Verdict.SKIP
