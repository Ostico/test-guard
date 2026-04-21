# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false
"""Tests for configuration parsing."""

import pytest

from src.config import parse_config


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
