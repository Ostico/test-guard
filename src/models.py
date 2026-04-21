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
