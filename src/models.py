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
    matched_test: str | None = None


@dataclass
class LayerResult:
    """Result from one layer of analysis."""

    layer: str
    verdict: Verdict
    details: str
    file_verdicts: list[FileVerdict]
    short_circuit: bool = False
    coverage_details: dict[str, float] | None = None


@dataclass
class Report:
    """Final report aggregating all layers."""

    layers: list[LayerResult] = field(default_factory=lambda: list[LayerResult]())

    @property
    def overall_verdict(self) -> Verdict:
        verdicts = [lr.verdict for lr in self.layers]
        if verdicts and all(verdict == Verdict.SKIP for verdict in verdicts):
            return Verdict.SKIP

        layer3_results = [lr for lr in self.layers if lr.layer == "layer3"]
        if layer3_results:
            l3_verdict = layer3_results[-1].verdict
            if l3_verdict != Verdict.SKIP:
                return l3_verdict

        non_skip = [v for v in verdicts if v != Verdict.SKIP]
        if not non_skip:
            return Verdict.SKIP
        if Verdict.FAIL in non_skip:
            return Verdict.FAIL
        if Verdict.WARNING in non_skip:
            return Verdict.WARNING
        return Verdict.PASS
