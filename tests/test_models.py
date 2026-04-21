"""Tests for shared data models."""

from src.models import FileVerdict, LayerResult, Report, Verdict


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
            LayerResult(
                "layer2",
                Verdict.FAIL,
                "Missing",
                [
                    FileVerdict("src/x.py", Verdict.FAIL, "No test", "layer2"),
                ],
                False,
            ),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.FAIL

    def test_overall_verdict_warning_when_warning_present(self):
        layers = [
            LayerResult("layer1", Verdict.SKIP, "No coverage", [], False),
            LayerResult("layer2", Verdict.PASS, "OK", [], False),
            LayerResult("layer3", Verdict.WARNING, "Uncertain", [], False),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.WARNING
