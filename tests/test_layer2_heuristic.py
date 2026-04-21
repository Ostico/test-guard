# pyright: reportPrivateUsage=false
"""Tests for Layer 2 — file-matching heuristic."""

from src.layer2_heuristic import _is_excluded, _is_test_file, _match_test_file, run_layer2
from src.models import Verdict

_PY_PATTERNS = {
    "python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"},
}


class TestIsExcluded:
    def test_markdown_excluded(self):
        assert _is_excluded("README.md", ["*.md", "docs/**"]) is True

    def test_migration_excluded(self):
        assert _is_excluded("migrations/001_init.sql", ["migrations/**"]) is True

    def test_source_not_excluded(self):
        assert _is_excluded("src/auth.py", ["*.md", "docs/**"]) is False

    def test_empty_patterns(self):
        assert _is_excluded("anything.py", []) is False


class TestMatchTestFile:
    def test_python_convention(self):
        result = _match_test_file(
            "src/auth.py",
            all_repo_files=["tests/test_auth.py", "src/auth.py"],
            patterns=_PY_PATTERNS,
        )
        assert result == "tests/test_auth.py"

    def test_php_convention(self):
        result = _match_test_file(
            "lib/Model/User.php",
            all_repo_files=["tests/Model/UserTest.php", "lib/Model/User.php"],
            patterns={"php": {"src_pattern": "**/*.php", "test_template": "**/{name}Test.php"}},
        )
        assert result == "tests/Model/UserTest.php"

    def test_no_match_found(self):
        result = _match_test_file(
            "src/billing.py",
            all_repo_files=["src/billing.py", "tests/test_auth.py"],
            patterns=_PY_PATTERNS,
        )
        assert result is None

    def test_test_file_is_not_matched_against_itself(self):
        result = _match_test_file(
            "tests/test_auth.py",
            all_repo_files=["tests/test_auth.py"],
            patterns=_PY_PATTERNS,
        )
        assert result is None

    def test_returns_none_when_source_matches_no_language_pattern(self):
        result = _match_test_file(
            "src/custom.xyz",
            all_repo_files=["src/custom.xyz", "tests/test_custom.xyz"],
            patterns=_PY_PATTERNS,
        )
        assert result is None


class TestIsTestFile:
    def test_detects_tests_suffix_for_dotnet(self):
        patterns = {
            "csharp": {"src_pattern": "**/*.cs", "test_template": "tests/{name}Tests.cs"},
        }
        assert _is_test_file("src/AuthTests.cs", patterns) is True

    def test_detects_dot_test_pattern(self):
        patterns = {
            "typescript": {"src_pattern": "**/*.ts", "test_template": "**/{name}.test.tsx"},
        }
        assert _is_test_file("src/Auth.test.tsx", patterns) is True

    def test_detects_dot_spec_pattern(self):
        patterns = {
            "typescript": {"src_pattern": "**/*.ts", "test_template": "**/{name}.spec.ts"},
        }
        assert _is_test_file("src/Auth.spec.ts", patterns) is True

    def test_detects_underscore_test_suffix(self):
        patterns = {
            "go": {"src_pattern": "**/*.go", "test_template": "**/{name}_test.go"},
        }
        assert _is_test_file("src/auth_test.go", patterns) is True

    def test_detects_underscore_spec_suffix(self):
        patterns = {
            "ruby": {"src_pattern": "**/*.rb", "test_template": "spec/{name}_spec.rb"},
        }
        assert _is_test_file("lib/auth_spec.rb", patterns) is True

    def test_detects_spec_suffix_for_scala(self):
        patterns = {
            "scala": {"src_pattern": "**/*.scala", "test_template": "**/{name}Spec.scala"},
        }
        assert _is_test_file("src/AuthSpec.scala", patterns) is True

    def test_detects_dunder_tests_directory(self):
        patterns = {
            "javascript": {"src_pattern": "**/*.js", "test_template": "__tests__/{name}.js"},
        }
        assert _is_test_file("__tests__/Auth.js", patterns) is True


class TestRunLayer2:
    def test_all_files_covered(self):
        result = run_layer2(
            changed_files=["src/auth.py", "tests/test_auth.py"],
            all_repo_files=["src/auth.py", "tests/test_auth.py"],
            patterns=_PY_PATTERNS,
            exclude_patterns=["*.md"],
        )
        assert result.verdict == Verdict.PASS
        assert len(result.file_verdicts) == 1
        assert result.file_verdicts[0].matched_test == "tests/test_auth.py"

    def test_missing_test_file(self):
        result = run_layer2(
            changed_files=["src/billing.py"],
            all_repo_files=["src/billing.py"],
            patterns=_PY_PATTERNS,
            exclude_patterns=["*.md"],
        )
        assert result.verdict == Verdict.FAIL
        assert len(result.file_verdicts) == 1
        assert result.file_verdicts[0].file == "src/billing.py"
        assert result.file_verdicts[0].matched_test is None

    def test_excluded_files_skip(self):
        result = run_layer2(
            changed_files=["README.md", "migrations/001.sql"],
            all_repo_files=["README.md", "migrations/001.sql"],
            patterns=_PY_PATTERNS,
            exclude_patterns=["*.md", "migrations/**"],
        )
        assert result.verdict == Verdict.PASS

    def test_mixed_covered_and_missing(self):
        result = run_layer2(
            changed_files=["src/auth.py", "src/billing.py", "tests/test_auth.py"],
            all_repo_files=["src/auth.py", "src/billing.py", "tests/test_auth.py"],
            patterns=_PY_PATTERNS,
            exclude_patterns=[],
        )
        assert result.verdict == Verdict.FAIL
        verdicts_by_file = {fv.file: fv.verdict for fv in result.file_verdicts}
        assert verdicts_by_file["src/auth.py"] == Verdict.PASS
        assert verdicts_by_file["src/billing.py"] == Verdict.FAIL

    def test_ambiguous_files_collected(self):
        result = run_layer2(
            changed_files=["src/auth.py"],
            all_repo_files=["src/auth.py", "tests/test_auth.py"],
            patterns=_PY_PATTERNS,
            exclude_patterns=[],
        )
        assert result.file_verdicts[0].verdict == Verdict.WARNING
        assert result.file_verdicts[0].matched_test == "tests/test_auth.py"
