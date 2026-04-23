# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false

import json
from unittest.mock import MagicMock, patch

import src.layer3_ai as layer3_ai
from src.layer3_ai import (
    Layer3Result,
    Relevance,
    _build_ai_prompt,
    _call_github_models,
    _parse_ai_response,
    _sanitize_diff,
    compute_test_relevance,
    evaluate_file_shortcut,
    is_trivial_diff,
    run_layer3,
)
from src.models import Verdict


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

    def test_no_files_returns_pass(self):
        result = Layer3Result({}, "OK")
        assert result.verdict == Verdict.PASS

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
