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
