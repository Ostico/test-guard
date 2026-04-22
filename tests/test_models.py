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

    def test_matched_test_defaults_to_none(self):
        fv = FileVerdict(
            file="src/payments.py",
            verdict=Verdict.PASS,
            reason="Matched by convention",
            layer="layer2",
        )
        assert fv.matched_test is None


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

    def test_overall_verdict_skip_when_all_layers_skip(self):
        layers = [
            LayerResult("layer1", Verdict.SKIP, "No coverage data", [], False),
            LayerResult("layer2", Verdict.SKIP, "No relevant files", [], False),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.SKIP


class TestLayerResultCoverageDetails:
    def test_coverage_details_defaults_to_none(self):
        lr = LayerResult(
            layer="layer1",
            verdict=Verdict.PASS,
            details="OK",
            file_verdicts=[],
        )
        assert lr.coverage_details is None

    def test_coverage_details_stores_per_file_data(self):
        details = {"src/auth.py": 92.5, "src/billing.py": 18.0}
        lr = LayerResult(
            layer="layer1",
            verdict=Verdict.FAIL,
            details="Mixed coverage",
            file_verdicts=[],
            coverage_details=details,
        )
        assert lr.coverage_details == {"src/auth.py": 92.5, "src/billing.py": 18.0}

    def test_coverage_details_empty_dict(self):
        lr = LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details="No data",
            file_verdicts=[],
            coverage_details={},
        )
        assert lr.coverage_details == {}


class TestReportOverallVerdictLayer3Override:
    def test_layer3_pass_overrides_layer2_fail(self):
        layers = [
            LayerResult("layer1", Verdict.SKIP, "No coverage", [], False),
            LayerResult("layer2", Verdict.FAIL, "No name-matched test", [], False),
            LayerResult("layer3", Verdict.PASS, "AI confirmed tests exist", [], False),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.PASS

    def test_layer3_fail_overrides_layer1_pass_and_layer2_pass(self):
        layers = [
            LayerResult("layer1", Verdict.PASS, "Coverage OK", [], True),
            LayerResult("layer2", Verdict.PASS, "Tests matched", [], False),
            LayerResult("layer3", Verdict.FAIL, "AI found untested logic", [], False),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.FAIL

    def test_layer3_warning_overrides_other_passes(self):
        layers = [
            LayerResult("layer1", Verdict.PASS, "OK", [], True),
            LayerResult("layer2", Verdict.PASS, "OK", [], False),
            LayerResult("layer3", Verdict.WARNING, "Marginal coverage", [], False),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.WARNING

    def test_layer3_skip_falls_back_to_l1_l2_worst_wins(self):
        layers = [
            LayerResult("layer1", Verdict.SKIP, "No coverage", [], False),
            LayerResult("layer2", Verdict.FAIL, "No test matched", [], False),
            LayerResult("layer3", Verdict.SKIP, "AI API failed", [], False),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.FAIL

    def test_layer3_skip_falls_back_to_l1_l2_pass(self):
        layers = [
            LayerResult("layer1", Verdict.PASS, "Coverage OK", [], True),
            LayerResult("layer2", Verdict.PASS, "Tests matched", [], False),
            LayerResult("layer3", Verdict.SKIP, "AI API failed", [], False),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.PASS

    def test_layer3_skip_falls_back_l1_pass_l2_warning(self):
        layers = [
            LayerResult("layer1", Verdict.PASS, "OK", [], True),
            LayerResult("layer2", Verdict.WARNING, "Partial match", [], False),
            LayerResult("layer3", Verdict.SKIP, "AI API failed", [], False),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.WARNING

    def test_no_layer3_uses_original_worst_wins(self):
        layers = [
            LayerResult("layer1", Verdict.PASS, "OK", [], True),
            LayerResult("layer2", Verdict.FAIL, "Missing test", [], False),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.FAIL

    def test_layer3_pass_overrides_despite_l1_fail(self):
        layers = [
            LayerResult("layer1", Verdict.FAIL, "Low coverage", [], False),
            LayerResult("layer2", Verdict.FAIL, "No match", [], False),
            LayerResult("layer3", Verdict.PASS, "AI confirmed adequate", [], False),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.PASS

    def test_all_three_layers_skip(self):
        layers = [
            LayerResult("layer1", Verdict.SKIP, "No coverage", [], False),
            LayerResult("layer2", Verdict.SKIP, "No files", [], False),
            LayerResult("layer3", Verdict.SKIP, "API down", [], False),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.SKIP

    def test_layer3_skip_fallback_all_non_l3_skip(self):
        layers = [
            LayerResult("layer1", Verdict.SKIP, "No coverage", [], False),
            LayerResult("layer2", Verdict.SKIP, "No files", [], False),
            LayerResult("layer3", Verdict.SKIP, "API down", [], False),
        ]
        report = Report(layers=layers)
        assert report.overall_verdict == Verdict.SKIP
