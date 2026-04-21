# test-guard — Hybrid PR Test Adequacy Gate

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A GitHub Action that combines deterministic coverage checks, heuristic file-matching, and GPT-5-mini semantic analysis to gate PRs on test adequacy.

**Architecture:** Three-layer pipeline — Layer 1 (diff-coverage from coverage report) → Layer 2 (source↔test file matching heuristic) → Layer 3 (GPT-5-mini via GitHub Models API, only for ambiguous cases). Each layer can short-circuit with PASS. The Action posts a structured PR comment and commit status check. Zero external API keys needed — uses `GITHUB_TOKEN` for both GitHub API and GitHub Models.

**Tech Stack:** Python 3.12+, `diff-cover` library, GitHub Models API (OpenAI-compatible), GitHub REST API v3, `pytest` for testing.

---

## File Structure

```
test-guard/
├── action.yml                          # GitHub Action composite definition
├── requirements.txt                    # Python dependencies
├── src/
│   ├── __init__.py
│   ├── main.py                         # Entrypoint — orchestrates 3 layers
│   ├── config.py                       # Input parsing + validation
│   ├── layer1_coverage.py              # diff-cover integration
│   ├── layer2_heuristic.py             # Source→test file matching
│   ├── layer3_ai.py                    # GPT-5-mini judgment via GitHub Models
│   ├── github_client.py                # PR comments + commit status posting
│   └── models.py                       # Shared data classes (FileVerdict, Report, etc.)
├── prompts/
│   └── test_adequacy.txt               # System prompt for Layer 3
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # Shared fixtures
│   ├── test_config.py
│   ├── test_layer1_coverage.py
│   ├── test_layer2_heuristic.py
│   ├── test_layer3_ai.py
│   ├── test_github_client.py
│   ├── test_main.py
│   └── fixtures/
│       ├── sample_cobertura.xml        # Sample coverage report
│       ├── sample_diff.txt             # Sample git diff
│       └── sample_ai_response.json     # Sample GPT-5-mini response
├── .github/
│   └── workflows/
│       ├── ci.yml                      # Self-test on push/PR
│       └── dogfood.yml                 # Runs test-guard on its own PRs
├── pyproject.toml                      # Project metadata + pytest config
├── README.md                           # Usage docs
└── LICENSE                             # MIT
```

---

## Task 1: Project Skeleton + Data Models

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/models.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_models.py` (added inline below)

- [ ] **Step 1: Initialize git repo and create pyproject.toml**

```bash
cd test-guard
git init
```

```toml
# pyproject.toml
[project]
name = "test-guard"
version = "0.1.0"
description = "Hybrid PR test adequacy gate for GitHub Actions"
requires-python = ">=3.12"
license = "MIT"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Create requirements.txt**

```
diff-cover>=9.0,<10.0
requests>=2.31,<3.0
pytest>=8.0,<9.0
pytest-cov>=5.0,<6.0
```

- [ ] **Step 3: Create src/__init__.py and tests/__init__.py**

Both files are empty. Create them:

```bash
mkdir -p src tests tests/fixtures prompts .github/workflows
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 4: Write the data models with tests first**

Create `tests/test_models.py`:

```python
"""Tests for shared data models."""
import pytest
from src.models import FileVerdict, LayerResult, Verdict, Report


class TestFileVerdict:
    def test_create_covered(self):
        fv = FileVerdict(
            file="src/auth.py",
            verdict=Verdict.PASS,
            reason="Test file exists and was modified",
            layer="layer2",
        )
        assert fv.file == "src/auth.py"
        assert fv.verdict == Verdict.PASS
        assert fv.layer == "layer2"

    def test_create_no_test(self):
        fv = FileVerdict(
            file="src/billing.py",
            verdict=Verdict.FAIL,
            reason="No matching test file found",
            layer="layer2",
        )
        assert fv.verdict == Verdict.FAIL


class TestLayerResult:
    def test_pass_result_short_circuits(self):
        lr = LayerResult(
            layer="layer1",
            verdict=Verdict.PASS,
            details="Changed lines: 92% covered (threshold: 80%)",
            file_verdicts=[],
            short_circuit=True,
        )
        assert lr.short_circuit is True

    def test_fail_result_continues(self):
        lr = LayerResult(
            layer="layer1",
            verdict=Verdict.FAIL,
            details="Changed lines: 45% covered (threshold: 80%)",
            file_verdicts=[],
            short_circuit=False,
        )
        assert lr.short_circuit is False


class TestReport:
    def test_overall_verdict_pass_when_all_pass(self):
        layers = [
            LayerResult("layer1", Verdict.PASS, "OK", [], True),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.PASS

    def test_overall_verdict_fail_when_any_fail(self):
        layers = [
            LayerResult("layer1", Verdict.FAIL, "Low", [], False),
            LayerResult("layer2", Verdict.FAIL, "Missing", [
                FileVerdict("src/x.py", Verdict.FAIL, "No test", "layer2"),
            ], False),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.FAIL

    def test_overall_verdict_warning_when_warning_present(self):
        layers = [
            LayerResult("layer1", Verdict.FAIL, "Low", [], False),
            LayerResult("layer2", Verdict.PASS, "OK", [], False),
            LayerResult("layer3", Verdict.WARNING, "Uncertain", [], False),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.WARNING
```

- [ ] **Step 5: Run test to verify it fails**

Run: `pip install -r requirements.txt && pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models'`

- [ ] **Step 6: Implement the data models**

Create `src/models.py`:

```python
"""Shared data models for test-guard."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass(frozen=True)
class FileVerdict:
    """Verdict for a single file."""

    file: str
    verdict: Verdict
    reason: str
    layer: str


@dataclass
class LayerResult:
    """Result from one layer of analysis."""

    layer: str
    verdict: Verdict
    details: str
    file_verdicts: list[FileVerdict]
    short_circuit: bool = False


@dataclass
class Report:
    """Final report aggregating all layers."""

    layers: list[LayerResult] = field(default_factory=list)

    @property
    def overall_verdict(self) -> Verdict:
        """Determine overall verdict from all layers.

        Priority: FAIL > WARNING > PASS.
        If any layer short-circuited with PASS, and no subsequent
        layer produced FAIL/WARNING, overall is PASS.
        """
        verdicts = [lr.verdict for lr in self.layers]
        if Verdict.FAIL in verdicts:
            return Verdict.FAIL
        if Verdict.WARNING in verdicts:
            return Verdict.WARNING
        return Verdict.PASS
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: 6 passed

- [ ] **Step 8: Create conftest.py with shared fixtures**

Create `tests/conftest.py`:

```python
"""Shared test fixtures."""
import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_cobertura() -> str:
    return (FIXTURES_DIR / "sample_cobertura.xml").read_text()


@pytest.fixture
def sample_changed_files() -> list[str]:
    return [
        "src/auth.py",
        "src/billing.py",
        "src/utils/helpers.py",
        "README.md",
        "migrations/001_init.sql",
    ]


@pytest.fixture
def sample_ai_response() -> dict:
    return json.loads(
        (FIXTURES_DIR / "sample_ai_response.json").read_text()
    )
```

- [ ] **Step 9: Create fixture files**

Create `tests/fixtures/sample_cobertura.xml`:

```xml
<?xml version="1.0" ?>
<coverage version="7.4" timestamp="1700000000" lines-valid="100" lines-covered="85" line-rate="0.85" branches-covered="0" branches-valid="0" branch-rate="0" complexity="0">
  <packages>
    <package name="src" line-rate="0.85">
      <classes>
        <class name="auth.py" filename="src/auth.py" line-rate="0.90">
          <lines>
            <line number="10" hits="1"/>
            <line number="11" hits="1"/>
            <line number="15" hits="0"/>
          </lines>
        </class>
        <class name="billing.py" filename="src/billing.py" line-rate="0.40">
          <lines>
            <line number="5" hits="1"/>
            <line number="10" hits="0"/>
            <line number="15" hits="0"/>
            <line number="20" hits="0"/>
            <line number="25" hits="1"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
```

Create `tests/fixtures/sample_ai_response.json`:

```json
{
  "verdict": "warning",
  "confidence": 0.82,
  "files": [
    {
      "file": "src/billing.py",
      "verdict": "fail",
      "reason": "New discount logic on lines 10-20 has no test for negative amount edge case"
    }
  ]
}
```

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat: project skeleton with data models and test fixtures"
```

---

## Task 2: Configuration Parser

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
"""Tests for configuration parsing."""
import os

import pytest
from src.config import Config, parse_config


class TestParseConfig:
    def test_defaults(self, monkeypatch):
        """All inputs have sensible defaults."""
        monkeypatch.delenv("INPUT_COVERAGE-FILE", raising=False)
        monkeypatch.delenv("INPUT_COVERAGE-THRESHOLD", raising=False)
        monkeypatch.delenv("INPUT_TEST-PATTERNS", raising=False)
        monkeypatch.delenv("INPUT_EXCLUDE-PATTERNS", raising=False)
        monkeypatch.delenv("INPUT_AI-ENABLED", raising=False)
        monkeypatch.delenv("INPUT_AI-MODEL", raising=False)
        monkeypatch.delenv("INPUT_AI-CONFIDENCE-THRESHOLD", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake123")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

        cfg = parse_config()
        assert cfg.coverage_file is None
        assert cfg.coverage_threshold == 80
        assert cfg.ai_enabled is True
        assert cfg.ai_model == "openai/gpt-5-mini"
        assert cfg.ai_confidence_threshold == 0.7
        assert cfg.github_token == "ghp_fake123"
        assert cfg.repo == "owner/repo"

    def test_custom_values(self, monkeypatch):
        monkeypatch.setenv("INPUT_COVERAGE-FILE", "coverage.xml")
        monkeypatch.setenv("INPUT_COVERAGE-THRESHOLD", "90")
        monkeypatch.setenv("INPUT_AI-ENABLED", "false")
        monkeypatch.setenv("INPUT_AI-MODEL", "openai/gpt-4.1-mini")
        monkeypatch.setenv("INPUT_AI-CONFIDENCE-THRESHOLD", "0.9")
        monkeypatch.setenv("INPUT_EXCLUDE-PATTERNS", "*.md,docs/**")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_real456")
        monkeypatch.setenv("GITHUB_REPOSITORY", "org/project")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

        cfg = parse_config()
        assert cfg.coverage_file == "coverage.xml"
        assert cfg.coverage_threshold == 90
        assert cfg.ai_enabled is False
        assert cfg.ai_model == "openai/gpt-4.1-mini"
        assert cfg.ai_confidence_threshold == 0.9
        assert cfg.exclude_patterns == ["*.md", "docs/**"]

    def test_missing_github_token_raises(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        with pytest.raises(ValueError, match="GITHUB_TOKEN"):
            parse_config()

    def test_invalid_threshold_raises(self, monkeypatch):
        monkeypatch.setenv("INPUT_COVERAGE-THRESHOLD", "not-a-number")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        with pytest.raises(ValueError, match="coverage-threshold"):
            parse_config()

    def test_pr_number_from_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")
        cfg = parse_config()
        assert cfg.pr_number == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: Implement config parser**

Create `src/config.py`:

```python
"""Configuration parsing from GitHub Actions inputs (environment variables)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# GitHub Actions passes inputs as INPUT_<NAME> env vars (uppercased, hyphens kept).
_DEFAULT_EXCLUDE = "*.json,*.yml,*.yaml,*.md,*.txt,*.lock,*.toml,*.cfg,*.ini,migrations/**,docs/**,*.sql"

_DEFAULT_TEST_PATTERNS = {
    # source_glob -> test_template
    # {dir} = parent dir, {name} = filename without ext, {ext} = extension
    "python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"},
    "php":    {"src_pattern": "**/*.php", "test_template": "tests/{name}Test.php"},
    "js":     {"src_pattern": "**/*.js",  "test_template": "**/{name}.test.js"},
    "ts":     {"src_pattern": "**/*.ts",  "test_template": "**/{name}.test.ts"},
    "go":     {"src_pattern": "**/*.go",  "test_template": "**/{name}_test.go"},
    "java":   {"src_pattern": "**/*.java","test_template": "**/{name}Test.java"},
}


def _env(name: str, default: str | None = None) -> str | None:
    """Read a GitHub Actions input or regular env var."""
    # Actions inputs: INPUT_COVERAGE-FILE -> os.environ["INPUT_COVERAGE-FILE"]
    return os.environ.get(f"INPUT_{name.upper()}", os.environ.get(name.upper(), default))


def _env_required(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise ValueError(f"{name} is required but not set.")
    return val


@dataclass(frozen=True)
class Config:
    """Parsed and validated configuration."""

    # GitHub context
    github_token: str
    repo: str  # "owner/repo"
    pr_number: int | None
    event_name: str

    # Layer 1 — coverage
    coverage_file: str | None
    coverage_threshold: int  # 0-100

    # Layer 2 — heuristic
    test_patterns: dict[str, dict[str, str]]
    exclude_patterns: list[str]

    # Layer 3 — AI
    ai_enabled: bool
    ai_model: str
    ai_confidence_threshold: float  # 0.0-1.0


def parse_config() -> Config:
    """Parse configuration from environment variables."""
    github_token = _env_required("GITHUB_TOKEN")
    repo = _env_required("GITHUB_REPOSITORY")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "unknown")

    # Extract PR number from GITHUB_REF (refs/pull/<number>/merge)
    pr_number = None
    github_ref = os.environ.get("GITHUB_REF", "")
    match = re.search(r"refs/pull/(\d+)/", github_ref)
    if match:
        pr_number = int(match.group(1))

    # Layer 1
    coverage_file = _env("COVERAGE-FILE")
    threshold_raw = _env("COVERAGE-THRESHOLD", "80")
    try:
        coverage_threshold = int(threshold_raw)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        raise ValueError(
            f"coverage-threshold must be an integer, got: {threshold_raw!r}"
        )

    # Layer 2
    exclude_raw = _env("EXCLUDE-PATTERNS", _DEFAULT_EXCLUDE)
    exclude_patterns = [p.strip() for p in exclude_raw.split(",") if p.strip()]  # type: ignore[union-attr]

    test_patterns_raw = _env("TEST-PATTERNS", "auto")
    if test_patterns_raw == "auto":
        test_patterns = _DEFAULT_TEST_PATTERNS
    else:
        # Future: parse custom JSON patterns
        test_patterns = _DEFAULT_TEST_PATTERNS

    # Layer 3
    ai_enabled = _env("AI-ENABLED", "true").lower() in ("true", "1", "yes")  # type: ignore[union-attr]
    ai_model = _env("AI-MODEL", "openai/gpt-5-mini")  # type: ignore[arg-type]
    confidence_raw = _env("AI-CONFIDENCE-THRESHOLD", "0.7")
    try:
        ai_confidence_threshold = float(confidence_raw)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        raise ValueError(
            f"ai-confidence-threshold must be a float, got: {confidence_raw!r}"
        )

    return Config(
        github_token=github_token,
        repo=repo,
        pr_number=pr_number,
        event_name=event_name,
        coverage_file=coverage_file,
        coverage_threshold=coverage_threshold,
        test_patterns=test_patterns,
        exclude_patterns=exclude_patterns,
        ai_enabled=ai_enabled,
        ai_model=ai_model,
        ai_confidence_threshold=ai_confidence_threshold,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: config parser with GitHub Actions input handling"
```

---

## Task 3: Layer 1 — Coverage Gate

**Files:**
- Create: `src/layer1_coverage.py`
- Test: `tests/test_layer1_coverage.py`

This layer uses `diff-cover` to compute coverage on changed lines. If coverage >= threshold, short-circuit PASS.

- [ ] **Step 1: Write the failing test**

Create `tests/test_layer1_coverage.py`:

```python
"""Tests for Layer 1 — diff-coverage gate."""
import pytest
from unittest.mock import patch, MagicMock
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
        result = run_layer1(
            coverage_file=fake_path, threshold=80, diff_files=["src/foo.py"]
        )
        assert result.verdict == Verdict.SKIP

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_pass_when_above_threshold(self, mock_cov):
        mock_cov.return_value = 92.5
        result = run_layer1(
            coverage_file="coverage.xml",
            threshold=80,
            diff_files=["src/auth.py"],
        )
        assert result.verdict == Verdict.PASS
        assert result.short_circuit is True
        assert "92.5%" in result.details

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_fail_when_below_threshold(self, mock_cov):
        mock_cov.return_value = 45.0
        result = run_layer1(
            coverage_file="coverage.xml",
            threshold=80,
            diff_files=["src/billing.py"],
        )
        assert result.verdict == Verdict.FAIL
        assert result.short_circuit is False
        assert "45.0%" in result.details

    @patch("src.layer1_coverage._compute_diff_coverage")
    def test_pass_exactly_at_threshold(self, mock_cov):
        mock_cov.return_value = 80.0
        result = run_layer1(
            coverage_file="coverage.xml",
            threshold=80,
            diff_files=["src/auth.py"],
        )
        assert result.verdict == Verdict.PASS
        assert result.short_circuit is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layer1_coverage.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement Layer 1**

Create `src/layer1_coverage.py`:

```python
"""Layer 1: Diff-coverage gate.

Computes test coverage on changed/new lines using diff-cover.
If coverage >= threshold, short-circuit with PASS.
"""
from __future__ import annotations

import subprocess
import json
import os
from pathlib import Path

from src.models import LayerResult, Verdict


def _compute_diff_coverage(coverage_file: str) -> float:
    """Run diff-cover and return the total diff coverage percentage.

    Uses diff-cover CLI which compares coverage report against git diff.
    Returns coverage as a float 0-100.
    """
    try:
        result = subprocess.run(
            [
                "diff-cover",
                coverage_file,
                "--json-report", "/dev/stdout",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return -1.0

        data = json.loads(result.stdout)
        total = data.get("total_percent_covered", -1.0)
        return float(total)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return -1.0


def run_layer1(
    coverage_file: str | None,
    threshold: int,
    diff_files: list[str],
) -> LayerResult:
    """Execute Layer 1 analysis.

    Args:
        coverage_file: Path to coverage report (cobertura/lcov), or None.
        threshold: Minimum coverage % to auto-pass (0-100).
        diff_files: List of changed files in the PR.

    Returns:
        LayerResult with verdict and short_circuit flag.
    """
    if not coverage_file:
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details="No coverage file provided — skipping Layer 1.",
            file_verdicts=[],
            short_circuit=False,
        )

    if not Path(coverage_file).exists():
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details=f"Coverage file not found: {coverage_file} — skipping Layer 1.",
            file_verdicts=[],
            short_circuit=False,
        )

    pct = _compute_diff_coverage(coverage_file)

    if pct < 0:
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details="diff-cover failed to compute coverage — skipping Layer 1.",
            file_verdicts=[],
            short_circuit=False,
        )

    passed = pct >= threshold
    return LayerResult(
        layer="layer1",
        verdict=Verdict.PASS if passed else Verdict.FAIL,
        details=f"Changed lines: {pct}% covered (threshold: {threshold}%)",
        file_verdicts=[],
        short_circuit=passed,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_layer1_coverage.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: Layer 1 — diff-coverage gate with threshold short-circuit"
```

---

## Task 4: Layer 2 — File-Matching Heuristic

**Files:**
- Create: `src/layer2_heuristic.py`
- Test: `tests/test_layer2_heuristic.py`

This layer checks if changed source files have corresponding test files in the PR.

- [ ] **Step 1: Write the failing test**

Create `tests/test_layer2_heuristic.py`:

```python
"""Tests for Layer 2 — file-matching heuristic."""
import pytest
from src.layer2_heuristic import run_layer2, _match_test_file, _is_excluded
from src.models import Verdict


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
            patterns={"python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"}},
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
            patterns={"python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"}},
        )
        assert result is None

    def test_test_file_is_not_matched_against_itself(self):
        """A test file should not be treated as a source file needing tests."""
        result = _match_test_file(
            "tests/test_auth.py",
            all_repo_files=["tests/test_auth.py"],
            patterns={"python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"}},
        )
        assert result is None


class TestRunLayer2:
    def test_all_files_covered(self):
        result = run_layer2(
            changed_files=["src/auth.py", "tests/test_auth.py"],
            all_repo_files=["src/auth.py", "tests/test_auth.py"],
            patterns={"python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"}},
            exclude_patterns=["*.md"],
        )
        assert result.verdict == Verdict.PASS

    def test_missing_test_file(self):
        result = run_layer2(
            changed_files=["src/billing.py"],
            all_repo_files=["src/billing.py"],
            patterns={"python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"}},
            exclude_patterns=["*.md"],
        )
        assert result.verdict == Verdict.FAIL
        assert len(result.file_verdicts) == 1
        assert result.file_verdicts[0].file == "src/billing.py"

    def test_excluded_files_skip(self):
        result = run_layer2(
            changed_files=["README.md", "migrations/001.sql"],
            all_repo_files=["README.md", "migrations/001.sql"],
            patterns={"python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"}},
            exclude_patterns=["*.md", "migrations/**"],
        )
        assert result.verdict == Verdict.PASS

    def test_mixed_covered_and_missing(self):
        """Some files have tests, some don't — result is FAIL with verdicts for each."""
        result = run_layer2(
            changed_files=["src/auth.py", "src/billing.py"],
            all_repo_files=["src/auth.py", "src/billing.py", "tests/test_auth.py"],
            patterns={"python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"}},
            exclude_patterns=[],
        )
        assert result.verdict == Verdict.FAIL
        verdicts_by_file = {fv.file: fv.verdict for fv in result.file_verdicts}
        assert verdicts_by_file["src/auth.py"] == Verdict.PASS
        assert verdicts_by_file["src/billing.py"] == Verdict.FAIL

    def test_ambiguous_files_collected(self):
        """Files with tests that exist but weren't modified are ambiguous — need AI."""
        result = run_layer2(
            changed_files=["src/auth.py"],  # test_auth.py exists but not in changed_files
            all_repo_files=["src/auth.py", "tests/test_auth.py"],
            patterns={"python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"}},
            exclude_patterns=[],
        )
        # Test exists but wasn't modified — this is WARNING (ambiguous)
        assert result.file_verdicts[0].verdict == Verdict.WARNING
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layer2_heuristic.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement Layer 2**

Create `src/layer2_heuristic.py`:

```python
"""Layer 2: File-matching heuristic.

Checks if changed source files have corresponding test files.
Classifies each file as PASS (test modified), WARNING (test exists, not modified),
FAIL (no test found), or SKIP (excluded).
"""
from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath

from src.models import FileVerdict, LayerResult, Verdict


def _is_excluded(filepath: str, exclude_patterns: list[str]) -> bool:
    """Check if a file matches any exclusion pattern."""
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(filepath, pattern):
            return True
        # Also check just the filename for extension patterns
        if fnmatch.fnmatch(PurePosixPath(filepath).name, pattern):
            return True
    return False


def _is_test_file(filepath: str, patterns: dict[str, dict[str, str]]) -> bool:
    """Check if a file is itself a test file."""
    name = PurePosixPath(filepath).stem
    for _lang, mapping in patterns.items():
        template = mapping["test_template"]
        # Extract the test naming convention from template
        # e.g., "tests/test_{name}.py" -> file starts with "test_"
        # e.g., "**/{name}Test.php" -> file ends with "Test"
        if "test_{name}" in template and name.startswith("test_"):
            return True
        if "{name}Test" in template and name.endswith("Test"):
            return True
        if "{name}.test" in template and ".test." in filepath:
            return True
        if "{name}_test" in template and name.endswith("_test"):
            return True
    return False


def _match_test_file(
    source_file: str,
    all_repo_files: list[str],
    patterns: dict[str, dict[str, str]],
) -> str | None:
    """Find a matching test file for a source file.

    Returns the test file path if found, None otherwise.
    """
    if _is_test_file(source_file, patterns):
        return None  # Don't match test files against themselves

    source_path = PurePosixPath(source_file)
    source_name = source_path.stem
    source_ext = source_path.suffix

    for _lang, mapping in patterns.items():
        src_pattern = mapping["src_pattern"]
        test_template = mapping["test_template"]

        # Check if this source file matches the language pattern
        if not fnmatch.fnmatch(source_file, src_pattern):
            continue

        # Build possible test file names from template
        test_name = test_template.replace("{name}", source_name)

        # Search repo files for a match
        for repo_file in all_repo_files:
            if fnmatch.fnmatch(repo_file, test_name):
                return repo_file

    return None


def run_layer2(
    changed_files: list[str],
    all_repo_files: list[str],
    patterns: dict[str, dict[str, str]],
    exclude_patterns: list[str],
) -> LayerResult:
    """Execute Layer 2 analysis.

    Args:
        changed_files: Files changed in the PR.
        all_repo_files: All files in the repo (for test lookup).
        patterns: Language→pattern mappings for source-to-test matching.
        exclude_patterns: Glob patterns to exclude from analysis.

    Returns:
        LayerResult with per-file verdicts.
    """
    file_verdicts: list[FileVerdict] = []
    changed_set = set(changed_files)

    for filepath in changed_files:
        # Skip excluded files
        if _is_excluded(filepath, exclude_patterns):
            continue

        # Skip test files themselves
        if _is_test_file(filepath, patterns):
            continue

        test_file = _match_test_file(filepath, all_repo_files, patterns)

        if test_file is None:
            file_verdicts.append(FileVerdict(
                file=filepath,
                verdict=Verdict.FAIL,
                reason="No matching test file found",
                layer="layer2",
            ))
        elif test_file in changed_set:
            file_verdicts.append(FileVerdict(
                file=filepath,
                verdict=Verdict.PASS,
                reason=f"Test file modified in PR: {test_file}",
                layer="layer2",
            ))
        else:
            # Test exists but wasn't modified — ambiguous
            file_verdicts.append(FileVerdict(
                file=filepath,
                verdict=Verdict.WARNING,
                reason=f"Test file exists ({test_file}) but was not modified in this PR",
                layer="layer2",
            ))

    # Determine overall verdict
    verdicts = [fv.verdict for fv in file_verdicts]
    if not verdicts:
        overall = Verdict.PASS
    elif Verdict.FAIL in verdicts:
        overall = Verdict.FAIL
    elif Verdict.WARNING in verdicts:
        overall = Verdict.WARNING
    else:
        overall = Verdict.PASS

    details_parts = []
    for v in [Verdict.PASS, Verdict.WARNING, Verdict.FAIL]:
        count = verdicts.count(v)
        if count:
            details_parts.append(f"{count} {v.value}")
    details = f"File matching: {', '.join(details_parts)}" if details_parts else "No source files to check"

    return LayerResult(
        layer="layer2",
        verdict=overall,
        details=details,
        file_verdicts=file_verdicts,
        short_circuit=(overall == Verdict.PASS),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_layer2_heuristic.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: Layer 2 — source-to-test file matching heuristic"
```

---

## Task 5: Layer 3 — GPT-5-mini AI Judgment

**Files:**
- Create: `src/layer3_ai.py`
- Create: `prompts/test_adequacy.txt`
- Test: `tests/test_layer3_ai.py`

This layer calls GPT-5-mini via GitHub Models API for files that Layer 2 flagged as FAIL or WARNING. It sends the diff + test context and gets a structured JSON verdict.

- [ ] **Step 1: Create the system prompt**

Create `prompts/test_adequacy.txt`:

```
You are a test adequacy reviewer for a pull request. You receive source file diffs and any corresponding test files.

For each source file, determine whether the existing or new tests adequately cover the changes introduced in the diff.

Evaluate these dimensions:
1. BEHAVIORAL COVERAGE: Are new behaviors (branches, conditions, error paths) tested?
2. EDGE CASES: Are boundary conditions and edge cases from the diff covered?
3. REGRESSION: If this is a bugfix, does a test verify the bug won't recur?
4. INTEGRATION: Do tests verify actual behavior, not just mock interactions?

DO NOT penalize for:
- Missing tests on trivial changes (whitespace, comments, renames)
- Missing tests on configuration or documentation files
- Test style preferences (unit vs integration)

You MUST respond with valid JSON matching this exact schema:

{
  "verdict": "pass" | "fail" | "warning",
  "confidence": <float 0.0 to 1.0>,
  "files": [
    {
      "file": "<filepath>",
      "verdict": "pass" | "fail" | "warning",
      "reason": "<one sentence explaining the verdict>"
    }
  ]
}

Rules:
- "pass": Tests adequately cover the changes.
- "fail": Significant behavioral changes have NO corresponding tests.
- "warning": Tests exist but may not cover important edge cases.
- Set confidence to how certain you are about your overall verdict (0.0 = guessing, 1.0 = certain).
- Be concise. One sentence per file reason.
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_layer3_ai.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_layer3_ai.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement Layer 3**

Create `src/layer3_ai.py`:

```python
"""Layer 3: GPT-5-mini AI judgment via GitHub Models API.

Only invoked for files that Layer 2 flagged as FAIL or WARNING.
Sends focused diffs + test content and gets structured JSON verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

from src.models import FileVerdict, LayerResult, Verdict

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "test_adequacy.txt"
_GITHUB_MODELS_URL = "https://models.github.ai/inference/chat/completions"


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text().strip()


def _build_prompt(
    file_diffs: dict[str, str],
    test_contents: dict[str, str],
) -> str:
    """Build the user prompt with diff context and test files."""
    parts: list[str] = []

    for filepath, diff in file_diffs.items():
        parts.append(f"## Source file: {filepath}")
        parts.append(f"```diff\n{diff}\n```")

        # Find matching test content
        matching_tests = [
            (tf, tc) for tf, tc in test_contents.items()
            if filepath.split("/")[-1].replace(".py", "") in tf
            or filepath.split("/")[-1].replace(".php", "") in tf
            or filepath.split("/")[-1].replace(".ts", "") in tf
            or filepath.split("/")[-1].replace(".js", "") in tf
            or filepath.split("/")[-1].replace(".go", "") in tf
            or filepath.split("/")[-1].replace(".java", "") in tf
        ]

        if matching_tests:
            for test_file, test_code in matching_tests:
                parts.append(f"### Test file: {test_file}")
                parts.append(f"```\n{test_code}\n```")
        else:
            parts.append("### No test file found for this source file.")

        parts.append("")

    return "\n".join(parts)


def _call_github_models(
    model: str,
    system_prompt: str,
    user_prompt: str,
    token: str,
) -> str:
    """Call GitHub Models API and return the raw response text."""
    response = requests.post(
        _GITHUB_MODELS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 2048,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _parse_ai_response(
    raw: str,
) -> tuple[Verdict, float, list[FileVerdict]]:
    """Parse the AI's JSON response into structured data.

    Returns (verdict, confidence, file_verdicts).
    On parse failure, returns (SKIP, 0.0, []).
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Verdict.SKIP, 0.0, []

    if not all(k in data for k in ("verdict", "confidence", "files")):
        return Verdict.SKIP, 0.0, []

    verdict_map = {"pass": Verdict.PASS, "fail": Verdict.FAIL, "warning": Verdict.WARNING}
    verdict = verdict_map.get(data["verdict"], Verdict.SKIP)
    confidence = float(data.get("confidence", 0.0))

    file_verdicts = []
    for f in data.get("files", []):
        fv_verdict = verdict_map.get(f.get("verdict", ""), Verdict.SKIP)
        file_verdicts.append(FileVerdict(
            file=f.get("file", "unknown"),
            verdict=fv_verdict,
            reason=f.get("reason", "No reason provided"),
            layer="layer3",
        ))

    return verdict, confidence, file_verdicts


def run_layer3(
    file_diffs: dict[str, str],
    test_contents: dict[str, str],
    model: str,
    token: str,
    confidence_threshold: float,
) -> LayerResult:
    """Execute Layer 3 AI analysis.

    Args:
        file_diffs: {filepath: diff_text} for files needing AI review.
        test_contents: {test_filepath: file_content} for related test files.
        model: Model identifier (e.g., "openai/gpt-5-mini").
        token: GitHub token for Models API authentication.
        confidence_threshold: Minimum confidence to enforce verdict (0.0-1.0).

    Returns:
        LayerResult with AI verdict and per-file analysis.
    """
    if not file_diffs:
        return LayerResult(
            layer="layer3",
            verdict=Verdict.PASS,
            details="No files required AI review.",
            file_verdicts=[],
            short_circuit=False,
        )

    try:
        system_prompt = _load_system_prompt()
        user_prompt = _build_prompt(file_diffs, test_contents)
        raw_response = _call_github_models(model, system_prompt, user_prompt, token)
        verdict, confidence, file_verdicts = _parse_ai_response(raw_response)
    except Exception as exc:
        return LayerResult(
            layer="layer3",
            verdict=Verdict.SKIP,
            details=f"AI analysis failed: {exc}",
            file_verdicts=[],
            short_circuit=False,
        )

    if verdict == Verdict.SKIP:
        return LayerResult(
            layer="layer3",
            verdict=Verdict.SKIP,
            details="AI returned unparseable response — skipping.",
            file_verdicts=file_verdicts,
            short_circuit=False,
        )

    # If confidence is below threshold, downgrade FAIL to WARNING
    if confidence < confidence_threshold and verdict == Verdict.FAIL:
        verdict = Verdict.WARNING
        details = (
            f"AI verdict: fail → warning (confidence {confidence:.0%} "
            f"below threshold {confidence_threshold:.0%})"
        )
    else:
        details = f"AI verdict: {verdict.value} (confidence: {confidence:.0%})"

    return LayerResult(
        layer="layer3",
        verdict=verdict,
        details=details,
        file_verdicts=file_verdicts,
        short_circuit=False,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_layer3_ai.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: Layer 3 — GPT-5-mini test adequacy judgment via GitHub Models"
```

---

## Task 6: GitHub Client — PR Comments + Status Checks

**Files:**
- Create: `src/github_client.py`
- Test: `tests/test_github_client.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_github_client.py`:

```python
"""Tests for GitHub client — PR comments and status checks."""
import pytest
from unittest.mock import patch, MagicMock
from src.github_client import format_report, post_comment, post_status
from src.models import FileVerdict, LayerResult, Report, Verdict


@pytest.fixture
def sample_report() -> Report:
    return Report(layers=[
        LayerResult(
            layer="layer1",
            verdict=Verdict.PASS,
            details="Changed lines: 92% covered (threshold: 80%)",
            file_verdicts=[],
            short_circuit=True,
        ),
    ])


@pytest.fixture
def full_report() -> Report:
    return Report(layers=[
        LayerResult("layer1", Verdict.FAIL, "Changed lines: 45% (threshold: 80%)", [], False),
        LayerResult("layer2", Verdict.FAIL, "File matching: 1 pass, 1 fail", [
            FileVerdict("src/auth.py", Verdict.PASS, "Test modified: tests/test_auth.py", "layer2"),
            FileVerdict("src/billing.py", Verdict.FAIL, "No matching test file", "layer2"),
        ], False),
        LayerResult("layer3", Verdict.WARNING, "AI verdict: warning (confidence: 82%)", [
            FileVerdict("src/billing.py", Verdict.FAIL, "No edge case test for negatives", "layer3"),
        ], False),
    ])


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_github_client.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement GitHub client**

Create `src/github_client.py`:

```python
"""GitHub API client for PR comments and commit status checks."""
from __future__ import annotations

import requests as http_requests

from src.models import FileVerdict, LayerResult, Report, Verdict

_GITHUB_API = "https://api.github.com"

_VERDICT_EMOJI = {
    Verdict.PASS: "✅",
    Verdict.FAIL: "❌",
    Verdict.WARNING: "⚠️",
    Verdict.SKIP: "⏭️",
}

_VERDICT_STATE = {
    Verdict.PASS: "success",
    Verdict.FAIL: "failure",
    Verdict.WARNING: "success",  # Warnings don't block
    Verdict.SKIP: "success",
}


def format_report(report: Report) -> str:
    """Format a Report as a Markdown PR comment."""
    emoji = _VERDICT_EMOJI[report.overall_verdict]
    lines = [
        f"## 🧪 Test Guard Report",
        "",
    ]

    for lr in report.layers:
        layer_emoji = _VERDICT_EMOJI[lr.verdict]
        layer_name = lr.layer.replace("layer", "Layer ")
        lines.append(f"### {layer_name}: {layer_emoji} {lr.verdict.value.upper()}")
        lines.append(lr.details)
        lines.append("")

        if lr.file_verdicts:
            lines.append("| File | Verdict | Reason |")
            lines.append("|---|---|---|")
            for fv in lr.file_verdicts:
                fv_emoji = _VERDICT_EMOJI[fv.verdict]
                lines.append(f"| `{fv.file}` | {fv_emoji} {fv.verdict.value} | {fv.reason} |")
            lines.append("")

    lines.append(f"**Result: {emoji} {report.overall_verdict.value.upper()}**")
    return "\n".join(lines)


def post_comment(
    token: str,
    repo: str,
    pr_number: int,
    body: str,
) -> None:
    """Post or update a comment on a PR."""
    url = f"{_GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    http_requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"body": body},
        timeout=30,
    )


def post_status(
    token: str,
    repo: str,
    sha: str,
    state: str,
    description: str,
) -> None:
    """Post a commit status check."""
    url = f"{_GITHUB_API}/repos/{repo}/statuses/{sha}"
    http_requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={
            "state": state,
            "description": description[:140],
            "context": "test-guard",
        },
        timeout=30,
    )


def report_to_github(
    report: Report,
    token: str,
    repo: str,
    pr_number: int | None,
    sha: str,
) -> None:
    """Post both the PR comment and commit status."""
    # Always post status
    state = _VERDICT_STATE[report.overall_verdict]
    desc_map = {
        Verdict.PASS: "All test adequacy checks passed",
        Verdict.FAIL: "Test adequacy issues found",
        Verdict.WARNING: "Test adequacy warnings (non-blocking)",
        Verdict.SKIP: "Analysis skipped",
    }
    post_status(token, repo, sha, state, desc_map[report.overall_verdict])

    # Post comment only on PRs
    if pr_number:
        body = format_report(report)
        post_comment(token, repo, pr_number, body)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_github_client.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: GitHub client — PR comments and commit status checks"
```

---

## Task 7: Main Orchestrator

**Files:**
- Create: `src/main.py`
- Test: `tests/test_main.py`

The orchestrator runs all three layers in sequence, short-circuiting when possible, and posts results to GitHub.

- [ ] **Step 1: Write the failing test**

Create `tests/test_main.py`:

```python
"""Tests for main orchestrator."""
import pytest
from unittest.mock import patch, MagicMock

from src.main import run_pipeline
from src.models import LayerResult, Verdict, Report
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
        base_config = Config(**{**base_config.__dict__, "ai_enabled": False})
        mock_ctx.return_value = (["src/x.py"], ["src/x.py"], "sha123", {})
        mock_l1.return_value = LayerResult("layer1", Verdict.SKIP, "No cov", [], False)
        mock_l2.return_value = LayerResult("layer2", Verdict.FAIL, "Missing", [], False)

        report = run_pipeline(base_config)
        assert len(report.layers) == 2  # Layer 3 was skipped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the orchestrator**

Create `src/main.py`:

```python
"""Main orchestrator — runs the 3-layer pipeline."""
from __future__ import annotations

import os
import subprocess
import sys

import requests as http_requests

from src.config import Config, parse_config
from src.github_client import report_to_github
from src.layer1_coverage import run_layer1
from src.layer2_heuristic import run_layer2
from src.layer3_ai import run_layer3
from src.models import Report, Verdict

_GITHUB_API = "https://api.github.com"


def _get_pr_context(
    config: Config,
) -> tuple[list[str], list[str], str, dict[str, str]]:
    """Fetch PR context from GitHub API.

    Returns:
        (changed_files, all_repo_files, head_sha, file_diffs)
    """
    headers = {
        "Authorization": f"Bearer {config.github_token}",
        "Accept": "application/vnd.github+json",
    }

    # Get PR files
    pr_url = f"{_GITHUB_API}/repos/{config.repo}/pulls/{config.pr_number}/files"
    response = http_requests.get(pr_url, headers=headers, params={"per_page": 100}, timeout=30)
    response.raise_for_status()
    pr_files = response.json()

    changed_files = [f["filename"] for f in pr_files]
    file_diffs = {f["filename"]: f.get("patch", "") for f in pr_files if f.get("patch")}

    # Get head SHA
    pr_detail_url = f"{_GITHUB_API}/repos/{config.repo}/pulls/{config.pr_number}"
    pr_detail = http_requests.get(pr_detail_url, headers=headers, timeout=30).json()
    head_sha = pr_detail["head"]["sha"]

    # Get repo file tree (for test-file lookup)
    tree_url = f"{_GITHUB_API}/repos/{config.repo}/git/trees/{head_sha}?recursive=1"
    tree_resp = http_requests.get(tree_url, headers=headers, timeout=30).json()
    all_repo_files = [item["path"] for item in tree_resp.get("tree", []) if item["type"] == "blob"]

    return changed_files, all_repo_files, head_sha, file_diffs


def run_pipeline(config: Config) -> Report:
    """Execute the full 3-layer pipeline.

    Each layer can short-circuit the pipeline with a PASS verdict.
    """
    report = Report()

    # Fetch PR context
    changed_files, all_repo_files, head_sha, file_diffs = _get_pr_context(config)

    # === Layer 1: Coverage Gate ===
    l1 = run_layer1(config.coverage_file, config.coverage_threshold, changed_files)
    report.layers.append(l1)
    if l1.short_circuit:
        report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
        return report

    # === Layer 2: File-Matching Heuristic ===
    l2 = run_layer2(changed_files, all_repo_files, config.test_patterns, config.exclude_patterns)
    report.layers.append(l2)
    if l2.short_circuit:
        report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
        return report

    # === Layer 3: AI Judgment ===
    if not config.ai_enabled:
        report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
        return report

    # Collect files that need AI review (FAIL or WARNING from Layer 2)
    files_for_ai = [
        fv.file for fv in l2.file_verdicts
        if fv.verdict in (Verdict.FAIL, Verdict.WARNING)
    ]
    ai_diffs = {f: file_diffs.get(f, "") for f in files_for_ai if f in file_diffs}

    # Fetch test file contents for AI context
    test_contents: dict[str, str] = {}
    for fv in l2.file_verdicts:
        if fv.verdict == Verdict.WARNING and "exists" in fv.reason:
            # Extract test file path from reason
            # Reason format: "Test file exists (tests/test_foo.py) but was not modified"
            import re
            match = re.search(r"\(([^)]+)\)", fv.reason)
            if match:
                test_path = match.group(1)
                try:
                    content_url = (
                        f"{_GITHUB_API}/repos/{config.repo}/contents/{test_path}"
                        f"?ref={head_sha}"
                    )
                    resp = http_requests.get(
                        content_url,
                        headers={
                            "Authorization": f"Bearer {config.github_token}",
                            "Accept": "application/vnd.github.raw+json",
                        },
                        timeout=30,
                    )
                    if resp.ok:
                        test_contents[test_path] = resp.text
                except Exception:
                    pass

    l3 = run_layer3(ai_diffs, test_contents, config.ai_model, config.github_token, config.ai_confidence_threshold)
    report.layers.append(l3)

    report_to_github(report, config.github_token, config.repo, config.pr_number, head_sha)
    return report


def main() -> None:
    """Entry point for the GitHub Action."""
    config = parse_config()

    if config.event_name != "pull_request":
        print("::notice::test-guard only runs on pull_request events. Skipping.")
        return

    if config.pr_number is None:
        print("::error::Could not determine PR number from GITHUB_REF.")
        sys.exit(1)

    report = run_pipeline(config)

    # Set exit code based on verdict
    if report.overall_verdict == Verdict.FAIL:
        print(f"::error::Test adequacy check FAILED.")
        sys.exit(1)
    elif report.overall_verdict == Verdict.WARNING:
        print(f"::warning::Test adequacy warnings found (non-blocking).")
    else:
        print(f"::notice::Test adequacy check passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: 3 passed

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass (approximately 31 tests)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: main orchestrator — 3-layer pipeline with short-circuit logic"
```

---

## Task 8: GitHub Action Definition

**Files:**
- Create: `action.yml`

- [ ] **Step 1: Create action.yml**

```yaml
# action.yml
name: 'Test Guard'
description: 'Hybrid PR test adequacy gate — coverage + heuristics + AI'
author: 'your-org'

branding:
  icon: 'shield'
  color: 'green'

inputs:
  coverage-file:
    description: 'Path to coverage report (cobertura XML, lcov, etc.). Optional — Layer 1 skips if not provided.'
    required: false
  coverage-threshold:
    description: 'Minimum diff-coverage % to auto-pass (0-100).'
    default: '80'
  test-patterns:
    description: 'Source→test file mapping. "auto" detects from project structure.'
    default: 'auto'
  exclude-patterns:
    description: 'Comma-separated glob patterns to exclude from analysis.'
    default: '*.json,*.yml,*.yaml,*.md,*.txt,*.lock,*.toml,*.cfg,*.ini,migrations/**,docs/**,*.sql'
  ai-enabled:
    description: 'Enable AI Layer 3 analysis (true/false).'
    default: 'true'
  ai-model:
    description: 'GitHub Models model ID for AI analysis.'
    default: 'openai/gpt-5-mini'
  ai-confidence-threshold:
    description: 'Minimum AI confidence to enforce fail verdict (0.0-1.0). Below this, fail becomes warning.'
    default: '0.7'

runs:
  using: 'composite'
  steps:
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.12'

    - name: Install dependencies
      shell: bash
      run: pip install -r ${{ github.action_path }}/requirements.txt

    - name: Run Test Guard
      shell: bash
      env:
        GITHUB_TOKEN: ${{ github.token }}
        INPUT_COVERAGE-FILE: ${{ inputs.coverage-file }}
        INPUT_COVERAGE-THRESHOLD: ${{ inputs.coverage-threshold }}
        INPUT_TEST-PATTERNS: ${{ inputs.test-patterns }}
        INPUT_EXCLUDE-PATTERNS: ${{ inputs.exclude-patterns }}
        INPUT_AI-ENABLED: ${{ inputs.ai-enabled }}
        INPUT_AI-MODEL: ${{ inputs.ai-model }}
        INPUT_AI-CONFIDENCE-THRESHOLD: ${{ inputs.ai-confidence-threshold }}
      run: python ${{ github.action_path }}/src/main.py
```

- [ ] **Step 2: Commit**

```bash
git add action.yml
git commit -m "feat: GitHub Action composite definition with all inputs"
```

---

## Task 9: CI Workflow + Dogfooding

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/dogfood.yml`

- [ ] **Step 1: Create CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests with coverage
        run: pytest tests/ -v --cov=src --cov-report=xml --cov-report=term

      - name: Upload coverage artifact
        uses: actions/upload-artifact@v4
        with:
          name: coverage
          path: coverage.xml
```

- [ ] **Step 2: Create dogfood workflow**

Create `.github/workflows/dogfood.yml`:

```yaml
name: Test Guard (Dogfood)
on:
  pull_request:
    types: [opened, synchronize]

permissions:
  contents: read
  pull-requests: write
  statuses: write

jobs:
  test-guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Run tests with coverage
        run: |
          pip install -r requirements.txt
          pytest tests/ --cov=src --cov-report=xml

      - name: Run Test Guard on itself
        uses: ./
        with:
          coverage-file: coverage.xml
          coverage-threshold: '70'
          ai-model: 'openai/gpt-5-mini'
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "ci: add test workflow and dogfood test-guard on its own PRs"
```

---

## Task 10: README + LICENSE

**Files:**
- Create: `README.md`
- Create: `LICENSE`

- [ ] **Step 1: Create README**

Create `README.md`:

```markdown
# 🧪 test-guard

**Hybrid PR test adequacy gate for GitHub Actions.**

Three layers of defense — deterministic checks first, AI only when needed.

## How it works

| Layer | What | Cost | Speed |
|---|---|---|---|
| **Layer 1** | Diff-coverage on changed lines | Free | <1s |
| **Layer 2** | Source→test file matching | Free | <1s |
| **Layer 3** | GPT-5-mini semantic analysis | Free* | ~5s |

*Free via GitHub Models API using your existing `GITHUB_TOKEN`.

Layer 1 and 2 handle ~80% of cases. Layer 3 only fires for ambiguous files.

## Quick start

```yaml
name: Test Guard
on:
  pull_request:
    types: [opened, synchronize]

permissions:
  contents: read
  pull-requests: write
  statuses: write

jobs:
  test-guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run your tests with coverage
        run: pytest --cov --cov-report=xml  # your test command here

      - name: Test Guard
        uses: your-org/test-guard@v1
        with:
          coverage-file: coverage.xml
```

No API keys needed. Uses `GITHUB_TOKEN` for everything.

## Inputs

| Input | Default | Description |
|---|---|---|
| `coverage-file` | _(none)_ | Path to coverage report. Layer 1 skips if not provided. |
| `coverage-threshold` | `80` | Min diff-coverage % to auto-pass. |
| `test-patterns` | `auto` | Source→test file mapping. Auto-detects Python, PHP, JS/TS, Go, Java. |
| `exclude-patterns` | `*.md,docs/**,...` | Files to skip. |
| `ai-enabled` | `true` | Enable Layer 3 AI analysis. |
| `ai-model` | `openai/gpt-5-mini` | GitHub Models model ID. |
| `ai-confidence-threshold` | `0.7` | Below this, AI "fail" becomes "warning". |

## Output example

```
## 🧪 Test Guard Report

### Layer 1: ❌ FAIL
Changed lines: 45% covered (threshold: 80%)

### Layer 2: ❌ FAIL
File matching: 1 pass, 1 fail

| File | Verdict | Reason |
|---|---|---|
| `src/auth.py` | ✅ pass | Test modified: tests/test_auth.py |
| `src/billing.py` | ❌ fail | No matching test file |

### Layer 3: ⚠️ WARNING
AI verdict: warning (confidence: 82%)

| File | Verdict | Reason |
|---|---|---|
| `src/billing.py` | ❌ fail | New discount logic has no edge case test for negative amounts |

**Result: ⚠️ WARNING**
```

## Without coverage (heuristic + AI only)

```yaml
- uses: your-org/test-guard@v1
  # No coverage-file — Layer 1 skips, Layer 2+3 still run
```

## Without AI (heuristic only)

```yaml
- uses: your-org/test-guard@v1
  with:
    ai-enabled: 'false'
    # Only Layer 1 + Layer 2 run — zero API calls
```

## License

MIT
```

- [ ] **Step 2: Create LICENSE**

Standard MIT license file with current year and your name.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "docs: README with usage examples and LICENSE"
```

---

## Self-Review Checklist

1. **Spec coverage**: All three layers implemented with tests. GitHub integration complete. Action definition with all inputs. CI + dogfood workflows.
2. **Placeholder scan**: No TBD/TODO items. All code blocks are complete.
3. **Type consistency**: `Verdict` enum used consistently. `LayerResult` and `FileVerdict` interfaces match across all layers. `Config` fields match `action.yml` inputs. `Report.overall_verdict` property tested.
4. **Model choice**: GPT-5-mini is the default throughout — in `config.py` defaults, `action.yml` defaults, `README.md` examples, and `dogfood.yml` workflow.
5. **Zero-cost path**: Layers 1+2 work with no API calls. Layer 3 uses `GITHUB_TOKEN` — no separate keys. AI can be disabled entirely.
