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
