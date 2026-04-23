"""Configuration parsing from GitHub Actions inputs (environment variables)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import overload

# GitHub Actions passes inputs as INPUT_<NAME> env vars (uppercased, hyphens kept).
_DEFAULT_EXCLUDE = (
    # Data / markup / generic config (no source language match)
    "*.json,*.yml,*.yaml,*.md,*.txt,*.lock,*.toml,*.cfg,*.ini,*.sql,"
    # Directories
    "migrations/**,docs/**,"
    # JS/TS config conventions (match **/*.js / **/*.ts but are not source)
    "*.config.js,*.config.ts,*.config.mjs,*.config.cjs,"
    "Gruntfile.js,Gulpfile.js,"
    # Python config conventions (match **/*.py but are not source)
    "conftest.py,setup.py,manage.py,noxfile.py,fabfile.py,"
    # Rust build script (matches **/*.rs but is not source)
    "build.rs"
)
_DEFAULT_COVERAGE_THRESHOLD = 80
_DEFAULT_AI_MODEL = "openai/gpt-5-mini"
_DEFAULT_AI_CONFIDENCE_THRESHOLD = 0.7
_DEFAULT_AI_ENABLED_VALUES = ("true", "1", "yes")

_DEFAULT_TEST_PATTERNS = {
    # source_glob -> test_template
    # {name} = filename without extension
    # Multiple entries per language to cover all common naming conventions.
    #
    # --- Python ---
    "python": {"src_pattern": "**/*.py", "test_template": "tests/test_{name}.py"},
    "python-suffix": {"src_pattern": "**/*.py", "test_template": "**/{name}_test.py"},
    #
    # --- JavaScript ---
    "js-test": {"src_pattern": "**/*.js", "test_template": "**/{name}.test.js"},
    "js-spec": {"src_pattern": "**/*.js", "test_template": "**/{name}.spec.js"},
    "js-dir": {"src_pattern": "**/*.js", "test_template": "**/__tests__/{name}.js"},
    #
    # --- JSX ---
    "jsx-test": {"src_pattern": "**/*.jsx", "test_template": "**/{name}.test.jsx"},
    "jsx-spec": {"src_pattern": "**/*.jsx", "test_template": "**/{name}.spec.jsx"},
    "jsx-dir": {"src_pattern": "**/*.jsx", "test_template": "**/__tests__/{name}.jsx"},
    #
    # --- TypeScript ---
    "ts-test": {"src_pattern": "**/*.ts", "test_template": "**/{name}.test.ts"},
    "ts-spec": {"src_pattern": "**/*.ts", "test_template": "**/{name}.spec.ts"},
    "ts-dir": {"src_pattern": "**/*.ts", "test_template": "**/__tests__/{name}.ts"},
    #
    # --- TSX ---
    "tsx-test": {"src_pattern": "**/*.tsx", "test_template": "**/{name}.test.tsx"},
    "tsx-spec": {"src_pattern": "**/*.tsx", "test_template": "**/{name}.spec.tsx"},
    "tsx-dir": {"src_pattern": "**/*.tsx", "test_template": "**/__tests__/{name}.tsx"},
    #
    # --- PHP ---
    "php": {"src_pattern": "**/*.php", "test_template": "**/{name}Test.php"},
    #
    # --- Go ---
    "go": {"src_pattern": "**/*.go", "test_template": "**/{name}_test.go"},
    #
    # --- Java ---
    "java": {"src_pattern": "**/*.java", "test_template": "**/{name}Test.java"},
    #
    # --- Kotlin ---
    "kotlin": {"src_pattern": "**/*.kt", "test_template": "**/{name}Test.kt"},
    #
    # --- Ruby ---
    "ruby-spec": {"src_pattern": "**/*.rb", "test_template": "**/{name}_spec.rb"},
    "ruby-test": {"src_pattern": "**/*.rb", "test_template": "**/test_{name}.rb"},
    #
    # --- Rust (integration tests; inline #[cfg(test)] detected by Layer 3 AI) ---
    "rust": {"src_pattern": "**/*.rs", "test_template": "tests/{name}.rs"},
    #
    # --- C# ---
    "csharp": {"src_pattern": "**/*.cs", "test_template": "**/{name}Tests.cs"},
    "csharp-single": {"src_pattern": "**/*.cs", "test_template": "**/{name}Test.cs"},
    #
    # --- Swift ---
    "swift": {"src_pattern": "**/*.swift", "test_template": "**/{name}Tests.swift"},
    "swift-single": {"src_pattern": "**/*.swift", "test_template": "**/{name}Test.swift"},
    #
    # --- Scala ---
    "scala-spec": {"src_pattern": "**/*.scala", "test_template": "**/{name}Spec.scala"},
    "scala-test": {"src_pattern": "**/*.scala", "test_template": "**/{name}Test.scala"},
    #
    # --- C ---
    "c": {"src_pattern": "**/*.c", "test_template": "**/test_{name}.c"},
    #
    # --- C++ ---
    "cpp": {"src_pattern": "**/*.cpp", "test_template": "**/test_{name}.cpp"},
    "cpp-cc": {"src_pattern": "**/*.cc", "test_template": "**/test_{name}.cc"},
    "cpp-cxx": {"src_pattern": "**/*.cxx", "test_template": "**/test_{name}.cxx"},
    #
    # --- Elixir ---
    "elixir": {"src_pattern": "**/*.ex", "test_template": "test/**/{name}_test.exs"},
    #
    # --- Dart ---
    "dart": {"src_pattern": "**/*.dart", "test_template": "test/**/{name}_test.dart"},
    #
    # --- Lua ---
    "lua": {"src_pattern": "**/*.lua", "test_template": "**/test_{name}.lua"},
    "lua-spec": {"src_pattern": "**/*.lua", "test_template": "**/{name}_spec.lua"},
}


@overload
def _env(name: str, default: str) -> str: ...
@overload
def _env(name: str, default: None = None) -> str | None: ...
def _env(name: str, default: str | None = None) -> str | None:
    """Read a GitHub Actions input or regular env var."""
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
    threshold_raw = _env("COVERAGE-THRESHOLD", str(_DEFAULT_COVERAGE_THRESHOLD))
    try:
        coverage_threshold = int(threshold_raw)
    except (ValueError, TypeError) as err:
        raise ValueError(f"coverage-threshold must be an integer, got: {threshold_raw!r}") from err
    if not (0 <= coverage_threshold <= 100):
        raise ValueError(f"coverage-threshold must be 0-100, got: {coverage_threshold}")

    # Layer 2
    exclude_raw = _env("EXCLUDE-PATTERNS", _DEFAULT_EXCLUDE)
    exclude_patterns = [p.strip() for p in exclude_raw.split(",") if p.strip()]

    test_patterns_raw = _env("TEST-PATTERNS", "auto")
    if test_patterns_raw == "auto":
        test_patterns = _DEFAULT_TEST_PATTERNS
    else:
        raise ValueError(
            "Custom test patterns not yet supported. Use 'auto' or omit the TEST-PATTERNS input."
        )

    # Layer 3
    ai_enabled = _env("AI-ENABLED", "true").lower() in _DEFAULT_AI_ENABLED_VALUES
    ai_model = _env("AI-MODEL", _DEFAULT_AI_MODEL)
    confidence_raw = _env(
        "AI-CONFIDENCE-THRESHOLD", str(_DEFAULT_AI_CONFIDENCE_THRESHOLD)
    )
    try:
        ai_confidence_threshold = float(confidence_raw)
    except (ValueError, TypeError) as err:
        raise ValueError(
            f"ai-confidence-threshold must be a float, got: {confidence_raw!r}"
        ) from err
    if not (0.0 <= ai_confidence_threshold <= 1.0):
        raise ValueError(
            f"ai-confidence-threshold must be 0.0-1.0, got: {ai_confidence_threshold}"
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
