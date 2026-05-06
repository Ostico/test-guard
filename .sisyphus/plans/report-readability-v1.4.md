# Report Readability v1.4.0

## TL;DR

> **Quick Summary**: Make Test-Guard's report output understandable to developers who've never seen it before. Five targeted fixes: pre-filter noisy files from L1, fix misleading coverage header, add outcome TL;DR, rename layers to human labels, and instruct AI to be concise.
> 
> **Deliverables**:
> - Pre-filtered source files for L1 (no test files, configs, docs in coverage table)
> - Honest L1 header when no source files appear in coverage data
> - One-line outcome TL;DR at top of every report
> - Human-readable layer names ("Coverage Analysis" instead of "Layer 1")
> - Concise AI reasons (~15 words) via prompt constraint
> - Updated README example output
> 
> **Estimated Effort**: Medium (5 fixes + tests + README)
> **Parallel Execution**: YES — 2 waves + final verification
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 5 → Task 6 → Task 7 → F1-F4

---

## Context

### Original Request
After deploying v1.3.5 with per-file L1 tables, real-world output on Matecat PR #4515 revealed readability problems. A developer unfamiliar with Test-Guard sees 10 files with ❌ FAIL in L1, then ❌ FAIL in L2, then ✅ PASS in L3, with the final result being PASS — confusing whiplash. The internal layer numbering and "(advisory)" jargon doesn't help.

### Interview Summary
**Key Discussions**:
- User wants the answer first (TL;DR), details second
- Layer numbers are internal architecture; users should see what each section does
- "(advisory)" is jargon — the TL;DR already tells you what matters
- AI reasons should be concise in the prompt, not truncated in Python (value preservation)
- L1 receives unfiltered `changed_files` including test files, configs, extensionless files — L2 already has proper filters that should be shared

**Research Findings**:
- Real Matecat PR had 10 changed files; only 4 were source files after proper filtering
- JS files never appear in PHP coverage XML — cross-language gap is structural, not a bug
- `_is_excluded`, `_is_test_file`, `_matches_source_pattern` are already imported in `main.py` line 22

### Metis Review
**Identified Gaps** (addressed):
- Fix 2 reframed: issue is the uninformative coverage message when `source_files` is empty, not specifically "100% covered"
- Fix 4 has test blast radius: 8+ assertions across 2 test files reference "Layer 1/2/3" and "(advisory)"
- Internal layer identifiers (`"layer1"`, `"layer2"`, `"layer3"`) must NOT change — only display mapping in `format_report()`
- When L1 receives empty pre-filtered list (only test files changed), verdict should be SKIP not FAIL
- TL;DR message must cover all verdict variants: PASS, FAIL, WARNING, SKIP

---

## Work Objectives

### Core Objective
Transform Test-Guard's report from an internal diagnostic dump into a developer-friendly assessment that answers "did my PR pass?" in one line, with supporting evidence below.

### Concrete Deliverables
- `src/main.py`: Pre-filter `changed_files` before `run_layer1()` call
- `src/layer1_coverage.py`: Handle empty `diff_files` (SKIP) and empty `source_files` (better message)
- `src/github_client.py`: TL;DR line + layer display name mapping + no "(advisory)"
- `prompts/test_adequacy.txt`: Concise reason instruction
- `README.md`: Updated example output
- Tests updated for all behavior changes

### Definition of Done
- [ ] `cd /home/hashashiyyin/tools/working_dir/test-guard && source .venv/bin/activate && pytest -v` — all tests pass
- [ ] Report output matches the approved example (4 files, TL;DR, human names, no advisory, concise AI)
- [ ] No internal layer identifiers changed (only display mapping)

### Must Have
- TL;DR as first line of report body (after `## 🧪 Test-Guard Report`)
- Layer display names: "Coverage Analysis", "Test File Matching", "Per-File Evaluation"
- Pre-filtered source-only files reaching L1
- L1 SKIP verdict when pre-filtered `diff_files` is empty
- AI conciseness instruction in prompt (not Python truncation)

### Must NOT Have (Guardrails)
- **DO NOT** change internal layer identifiers (`"layer1"`, `"layer2"`, `"layer3"`) anywhere — display mapping ONLY in `format_report()`
- **DO NOT** refactor layer identifiers to Enum or constants
- **DO NOT** change `run_layer1()` function signature
- **DO NOT** add Python-side truncation of AI reasons
- **DO NOT** restructure `format_report()` beyond adding display dict + TL;DR line
- **DO NOT** remove `_is_non_source()` from L1 (keep belt+suspenders even after pre-filtering)
- **DO NOT** add HTML, collapsible sections, or color styling to reports
- **DO NOT** make layer display names configurable (hardcoded dict is fine)
- **DO NOT** update source file docstrings that mention "Layer 1/2/3" — internal docs are fine

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest, 282 existing tests)
- **Automated tests**: TDD — write/update tests first (red), then implement (green)
- **Framework**: pytest
- **Each task follows**: RED (failing test) → GREEN (minimal impl) → REFACTOR

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **All tasks**: Use Bash — `pytest -v -k "test_name"` for individual tests, `pytest -v` for full regression

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Independent foundation fixes):
├── Task 1: Concise AI prompt (Fix 5) [quick]
├── Task 2: Layer display names + drop advisory (Fix 4) [unspecified-high]
└── Task 3: Pre-filter diff_files for L1 (Fix 1) [unspecified-high]

Wave 2 (Depends on Wave 1):
├── Task 4: L1 empty source_files message (Fix 2, depends: Task 3) [quick]
├── Task 5: TL;DR at top of report (Fix 3, depends: Task 2) [quick]
└── Task 6: README example output update (depends: Tasks 1-5) [quick]

Wave 3 (Regression):
└── Task 7: Full regression + safety verification [quick]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1 | — | 6 |
| 2 | — | 5, 6 |
| 3 | — | 4, 6 |
| 4 | 3 | 6 |
| 5 | 2 | 6 |
| 6 | 1, 2, 3, 4, 5 | 7 |
| 7 | 6 | F1-F4 |

### Agent Dispatch Summary

- **Wave 1**: 3 tasks — T1 `quick`, T2 `unspecified-high`, T3 `unspecified-high`
- **Wave 2**: 3 tasks — T4 `quick`, T5 `quick`, T6 `quick`
- **Wave 3**: 1 task — T7 `quick`
- **FINAL**: 4 tasks — F1 `oracle`, F2 `unspecified-high`, F3 `unspecified-high`, F4 `deep`

---

## TODOs

- [x] 1. Concise AI prompt constraint (Fix 5)

  **What to do**:
  - RED: Add test in `tests/test_layer3_ai.py` that reads `prompts/test_adequacy.txt` and asserts it contains a word-limit instruction for reasons (e.g., `"15 words"` or `"one sentence"`)
  - GREEN: Edit `prompts/test_adequacy.txt` line 41 — change `"Be concise — one sentence per file reason."` to include explicit word limit: `"Be concise — one sentence, maximum 15 words per file reason. Focus on WHAT is tested, not HOW."`
  - Verify test passes

  **Must NOT do**:
  - Do NOT add Python-side truncation in `layer3_ai.py`
  - Do NOT change the JSON output schema
  - Do NOT modify anything except the prompt file and its test

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-file prompt edit + one test assertion
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Task 6 (README)
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `prompts/test_adequacy.txt:41` — Current line: `"Be concise — one sentence per file reason."` — this is the exact line to edit

  **Test References**:
  - `tests/test_layer3_ai.py` — Existing L3 test file; add a new test class or method here

  **Acceptance Criteria**:

  **TDD:**
  - [ ] Test file reads prompt and asserts word-limit instruction present
  - [ ] `source .venv/bin/activate && pytest tests/test_layer3_ai.py -k "test_prompt_concise" -v` → PASS

  **QA Scenarios:**

  ```
  Scenario: Prompt contains conciseness instruction
    Tool: Bash
    Preconditions: prompts/test_adequacy.txt exists
    Steps:
      1. Run: grep -c "15 words" prompts/test_adequacy.txt
      2. Assert output is "1"
    Expected Result: Exactly one line contains "15 words"
    Evidence: .sisyphus/evidence/task-1-prompt-concise.txt

  Scenario: JSON schema unchanged
    Tool: Bash
    Preconditions: prompt file edited
    Steps:
      1. Run: grep -c "valid JSON" prompts/test_adequacy.txt
      2. Assert output ≥ 1
    Expected Result: JSON schema instruction still present
    Evidence: .sisyphus/evidence/task-1-schema-intact.txt
  ```

  **Commit**: YES (commit 1)
  - Message: `✨ feat(prompt): instruct AI to produce concise reasons`
  - Files: `prompts/test_adequacy.txt`, `tests/test_layer3_ai.py`
  - Pre-commit: `source .venv/bin/activate && pytest -v`

- [x] 2. Layer display names + drop advisory (Fix 4)

  **What to do**:
  - RED: Update tests in `tests/test_github_client.py`:
    - Change assertions from `"Layer 1"` → `"Coverage Analysis"`, `"Layer 2"` → `"Test File Matching"`, `"Layer 3"` → `"Per-File Evaluation"` in all affected tests (lines 67, 73-75, 288)
    - Update `test_layer2_advisory_when_layer3_present` (line 79): assert `"(advisory)"` is NOT in output (invert the assertion)
    - Update `test_layer2_not_advisory_without_layer3` (line 85): keep asserting no `"(advisory)"` — this test stays as-is but rename for clarity
    - All updated tests should FAIL (red) against current code
  - GREEN: Edit `src/github_client.py` `format_report()`:
    - Add display name dict: `_LAYER_DISPLAY_NAMES = {"layer1": "Coverage Analysis", "layer2": "Test File Matching", "layer3": "Per-File Evaluation"}`
    - Replace line 51 `layer_name = lr.layer.replace("layer", "Layer ")` with `layer_name = _LAYER_DISPLAY_NAMES.get(lr.layer, lr.layer)`
    - Remove line 54 advisory suffix logic entirely (delete the `suffix` variable and `{suffix}` from the f-string)
    - Run tests — all should pass (green)
  - CRITICAL: Do NOT change any internal `"layer1"` / `"layer2"` / `"layer3"` string identifiers in `models.py`, `main.py`, `layer1_coverage.py`, `layer2_heuristic.py`, or `layer3_ai.py`

  **Must NOT do**:
  - Do NOT change internal layer identifiers — display mapping ONLY
  - Do NOT refactor `format_report()` structure beyond the dict + suffix removal
  - Do NOT make display names configurable via action.yml

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multiple test assertions to update precisely; risk of breaking existing tests
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 5 (TL;DR) and 6 (README)
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/github_client.py:39-68` — `format_report()` function — the ONLY function to modify
  - `src/github_client.py:51` — Current: `layer_name = lr.layer.replace("layer", "Layer ")` — replace with dict lookup
  - `src/github_client.py:52-54` — Current advisory suffix logic — DELETE these 3 lines

  **Test References**:
  - `tests/test_github_client.py:67` — `assert "Layer 1" in md` → change to `"Coverage Analysis"`
  - `tests/test_github_client.py:73-75` — Three assertions with `"Layer 1/2/3"` → change all three
  - `tests/test_github_client.py:79-83` — `test_layer2_advisory_when_layer3_present` — invert assertions (advisory ABSENT)
  - `tests/test_github_client.py:85-94` — `test_layer2_not_advisory_without_layer3` — keep as-is (already asserts no advisory)
  - `tests/test_github_client.py:288` — `assert "Layer 1" in summary` → change to `"Coverage Analysis"`

  **Safety References**:
  - `src/models.py:77` — `lr.layer == "layer3"` — MUST NOT be touched (verdict authority logic)
  - `src/github_client.py:47` — `lr.layer == "layer3"` — internal check, keep as-is

  **Acceptance Criteria**:

  **TDD:**
  - [ ] Updated test assertions fail against current code (red confirmed)
  - [ ] `source .venv/bin/activate && pytest tests/test_github_client.py -v` → PASS after implementation

  **QA Scenarios:**

  ```
  Scenario: New display names present in output
    Tool: Bash
    Preconditions: format_report() updated
    Steps:
      1. Run: source .venv/bin/activate && pytest tests/test_github_client.py -k "test_format" -v
      2. Assert all PASS
      3. Run: python3 -c "from src.github_client import format_report; from src.models import *; r=Report(layers=[LayerResult('layer1',Verdict.PASS,'details',[]),LayerResult('layer2',Verdict.PASS,'details',[]),LayerResult('layer3',Verdict.PASS,'details',[])]); md=format_report(r); assert 'Coverage Analysis' in md; assert 'Test File Matching' in md; assert 'Per-File Evaluation' in md; assert 'Layer 1' not in md; assert '(advisory)' not in md; print('ALL CHECKS PASSED')"
    Expected Result: "ALL CHECKS PASSED" printed, no assertion errors
    Evidence: .sisyphus/evidence/task-2-display-names.txt

  Scenario: Internal identifiers preserved
    Tool: Bash
    Preconditions: format_report() updated
    Steps:
      1. Run: grep -c '"layer1"\|"layer2"\|"layer3"' src/models.py src/layer1_coverage.py src/layer2_heuristic.py src/layer3_ai.py src/main.py
      2. Assert total count > 0 (internal identifiers still present in source files)
      3. Run: grep -c '"layer1"\|"layer2"\|"layer3"' src/github_client.py
      4. Assert count > 0 (dict keys and has_layer3 check still use internal identifiers)
    Expected Result: Internal identifiers preserved in all source files
    Evidence: .sisyphus/evidence/task-2-identifiers-preserved.txt
  ```

  **Commit**: YES (commit 2)
  - Message: `✨ feat(report): rename layers to human-readable names and drop advisory suffix`
  - Files: `src/github_client.py`, `tests/test_github_client.py`
  - Pre-commit: `source .venv/bin/activate && pytest -v`

- [x] 3. Pre-filter diff_files before L1 (Fix 1) + L1 empty diff_files handling

  **What to do**:
  - RED: Add tests:
    - In `tests/test_main.py`: test that when `changed_files` contains only test files + excluded files, L1 receives an empty list and returns SKIP verdict
    - In `tests/test_layer1_coverage.py`: test that `run_layer1(coverage_files, threshold, diff_files=[])` returns `LayerResult` with `verdict=Verdict.SKIP` and `details="No source files to analyze"`
  - GREEN — Part A (`src/layer1_coverage.py`): Add early return at top of `run_layer1()` (after the `not coverage_files` check), before `_compute_diff_coverage`:
    ```python
    if not diff_files:
        return LayerResult(
            layer="layer1",
            verdict=Verdict.SKIP,
            details="No source files to analyze — all changed files are tests or excluded.",
            file_verdicts=[],
            short_circuit=False,
        )
    ```
  - GREEN — Part B (`src/main.py`): Before line 69 (`l1 = run_layer1(...)`), add pre-filtering:
    ```python
    # Pre-filter changed_files for L1: exclude test files, excluded patterns,
    # and non-source files so L1 only evaluates actual source code.
    l1_files = [
        f for f in changed_files
        if not _is_excluded(f, config.exclude_patterns)
        and not _is_test_file(f, config.test_patterns)
        and _matches_source_pattern(f, config.test_patterns)
    ]
    ```
    Then pass `l1_files` instead of `changed_files` to `run_layer1()`
  - IMPORTANT: Keep `_is_non_source()` in `layer1_coverage.py` unchanged (belt+suspenders)
  - IMPORTANT: Do NOT change `run_layer1()` function signature — it still accepts `diff_files: list[str]`

  **Must NOT do**:
  - Do NOT change `run_layer1()` signature
  - Do NOT remove `_is_non_source()` from L1
  - Do NOT modify L2 or L3 filtering logic

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Touches two files with test/production coupling; requires understanding filter semantics
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Task 4 (L1 empty message)
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/main.py:94-104` — Existing L3 diff splitting uses same `_is_excluded`, `_is_test_file`, `_matches_source_pattern` pattern — COPY THIS PATTERN for L1 pre-filtering
  - `src/main.py:22` — Already imports: `from src.layer2_heuristic import _is_excluded, _is_test_file, _matches_source_pattern, run_layer2` — no new imports needed
  - `src/main.py:69` — Current: `l1 = run_layer1(config.coverage_files, config.coverage_threshold, changed_files)` — change `changed_files` to `l1_files`

  **API/Type References**:
  - `src/layer1_coverage.py:103-107` — `run_layer1(coverage_files, threshold, diff_files)` signature — do NOT change
  - `src/layer1_coverage.py:108-115` — Existing early return pattern for `not coverage_files` — follow same pattern for `not diff_files`
  - `src/layer2_heuristic.py` — `_is_excluded(filepath, exclude_patterns)`, `_is_test_file(filepath, test_patterns)`, `_matches_source_pattern(filepath, test_patterns)` — these are the three filter functions

  **Test References**:
  - `tests/test_main.py` — Add integration-level test for pre-filtering behavior
  - `tests/test_layer1_coverage.py` — Add unit test for empty `diff_files` early return

  **Acceptance Criteria**:

  **TDD:**
  - [ ] New tests fail before implementation (red)
  - [ ] `source .venv/bin/activate && pytest tests/test_layer1_coverage.py -k "test_empty_diff_files" -v` → PASS
  - [ ] `source .venv/bin/activate && pytest tests/test_main.py -v` → PASS

  **QA Scenarios:**

  ```
  Scenario: L1 returns SKIP when diff_files is empty
    Tool: Bash
    Preconditions: layer1_coverage.py updated with early return
    Steps:
      1. Run: source .venv/bin/activate && python3 -c "
         from src.layer1_coverage import run_layer1
         from src.models import Verdict
         result = run_layer1(['coverage.xml'], 80, [])
         assert result.verdict == Verdict.SKIP, f'Expected SKIP, got {result.verdict}'
         assert 'No source files' in result.details
         assert result.file_verdicts == []
         print('SKIP ON EMPTY: PASSED')
         "
    Expected Result: "SKIP ON EMPTY: PASSED"
    Evidence: .sisyphus/evidence/task-3-l1-skip-empty.txt

  Scenario: Pre-filter removes test and excluded files from L1 input
    Tool: Bash
    Preconditions: main.py updated with pre-filtering
    Steps:
      1. Run: source .venv/bin/activate && python3 -c "
         from src.layer2_heuristic import _is_excluded, _is_test_file, _matches_source_pattern
         files = ['src/auth.py', 'tests/test_auth.py', 'jest.config.js', 'docker', 'phpstan.neon', 'src/auth.test.js']
         patterns = {}  # default patterns
         exclude = ['*.config.js']
         filtered = [f for f in files if not _is_excluded(f, exclude) and not _is_test_file(f, patterns) and _matches_source_pattern(f, patterns)]
         print(f'Filtered: {filtered}')
         assert 'tests/test_auth.py' not in filtered
         assert 'jest.config.js' not in filtered
         assert 'src/auth.py' in filtered
         print('FILTER: PASSED')
         "
    Expected Result: "FILTER: PASSED" — only source files remain
    Evidence: .sisyphus/evidence/task-3-prefilter.txt
  ```

  **Commit**: YES (commit 3)
  - Message: `✨ feat(pipeline): pre-filter diff_files before L1 to exclude non-source files`
  - Files: `src/main.py`, `src/layer1_coverage.py`, `tests/test_main.py`, `tests/test_layer1_coverage.py`
  - Pre-commit: `source .venv/bin/activate && pytest -v`

- [x] 4. L1 empty source_files message (Fix 2)

  **What to do**:
  - RED: Add test in `tests/test_layer1_coverage.py`: when `diff_files` has source files but NONE appear in coverage XML (`per_file` is empty), L1's `details` should contain `"No changed source files found in coverage report"` instead of `"Changed lines: X% covered"`
  - GREEN: Edit `src/layer1_coverage.py` lines 182-189. After computing `source_files` and `absent_files` (line 147-150), add a branch:
    ```python
    if not source_files:
        # No changed source files found in coverage data — all are absent.
        # Show an honest message instead of a vacuous coverage percentage.
        details = f"No changed source files found in coverage report (threshold: {threshold}%)"
    else:
        details = f"Changed lines: {total_pct}% covered (threshold: {threshold}%)"
    ```
    Then use `details` in the `LayerResult` constructor at line 185
  - The verdict logic (`passed = bool(source_files) and ...`) remains unchanged — this only fixes the display message

  **Must NOT do**:
  - Do NOT change the verdict logic (FAIL when absent_files exist is correct)
  - Do NOT change `run_layer1()` signature
  - Do NOT change the `_is_non_source()` function

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single conditional branch + one test
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6)
  - **Blocks**: Task 6 (README)
  - **Blocked By**: Task 3 (pre-filtering changes what reaches L1)

  **References**:

  **Pattern References**:
  - `src/layer1_coverage.py:147-150` — `source_files` and `absent_files` computation — this is WHERE to branch
  - `src/layer1_coverage.py:182-189` — Current `LayerResult` construction with `details=f"Changed lines: {total_pct}% covered..."` — this is WHAT to change

  **Test References**:
  - `tests/test_layer1_coverage.py` — Existing L1 tests; add test for empty `source_files` scenario

  **Acceptance Criteria**:

  **TDD:**
  - [ ] Test for empty source_files message fails before implementation (red)
  - [ ] `source .venv/bin/activate && pytest tests/test_layer1_coverage.py -k "test_no_source_files_in_coverage" -v` → PASS

  **QA Scenarios:**

  ```
  Scenario: L1 shows honest message when no files in coverage
    Tool: Bash
    Preconditions: layer1_coverage.py updated
    Steps:
      1. Run: source .venv/bin/activate && pytest tests/test_layer1_coverage.py -k "test_no_source_files" -v
      2. Assert PASS
    Expected Result: Test passes — details contains "No changed source files found in coverage report"
    Evidence: .sisyphus/evidence/task-4-empty-source-message.txt

  Scenario: Normal coverage message still works when source files present
    Tool: Bash
    Preconditions: layer1_coverage.py updated
    Steps:
      1. Run: source .venv/bin/activate && pytest tests/test_layer1_coverage.py -k "test_pass" -v
      2. Assert PASS — existing tests still expect "Changed lines: X% covered"
    Expected Result: Existing tests unbroken
    Evidence: .sisyphus/evidence/task-4-existing-coverage-message.txt
  ```

  **Commit**: YES (commit 3 — same commit as Task 3, since both touch L1)
  - Message: (combined with Task 3 commit)
  - Files: `src/layer1_coverage.py`, `tests/test_layer1_coverage.py`
  - Pre-commit: `source .venv/bin/activate && pytest -v`

- [x] 5. TL;DR at top of report (Fix 3)

  **What to do**:
  - RED: Add tests in `tests/test_github_client.py`:
    - Test PASS report: first content line after `## 🧪 Test-Guard Report` contains `"✅ PASS"` and a human-readable message
    - Test FAIL report: first content line contains `"❌ FAIL"`
    - Test WARNING report: first content line contains `"⚠️ WARNING"`
    - Test SKIP report: first content line contains `"⏭️ SKIP"`
    - Verify TL;DR verdict matches `report.overall_verdict`
  - GREEN: Edit `src/github_client.py` `format_report()`. After the `## 🧪 Test-Guard Report` header line (line 43-44), add:
    ```python
    _TLDR_MESSAGES = {
        Verdict.PASS: "All changed source files have adequate test coverage.",
        Verdict.FAIL: "Some changed source files lack adequate test coverage.",
        Verdict.WARNING: "Test coverage has minor gaps — review recommended.",
        Verdict.SKIP: "Unable to evaluate — no layers produced a verdict.",
    }
    # ... in format_report():
    tldr_msg = _TLDR_MESSAGES[report.overall_verdict]
    lines.append(f"**{emoji} {report.overall_verdict.value.upper()}** — {tldr_msg}")
    lines.append("")
    ```
  - The TL;DR line uses the same `emoji` variable already computed from `report.overall_verdict`

  **Must NOT do**:
  - Do NOT reference layer names in TL;DR (it's verdict-level, not layer-level)
  - Do NOT add HTML or collapsible sections
  - Do NOT change the `**Result: ...**` line at the bottom (keep both TL;DR and Result)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Add a dict + 2 lines to format_report + test assertions
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6)
  - **Blocks**: Task 6 (README)
  - **Blocked By**: Task 2 (layer display names must land first so tests reference correct names)

  **References**:

  **Pattern References**:
  - `src/github_client.py:39-68` — `format_report()` — insert TL;DR after lines 43-44
  - `src/github_client.py:41` — `emoji = _VERDICT_EMOJI[report.overall_verdict]` — reuse this emoji variable
  - `src/github_client.py:67` — `lines.append(f"**Result: {emoji} ...")` — the bottom Result line stays; TL;DR goes at top

  **API/Type References**:
  - `src/models.py:9-18` — `Verdict` enum values — one TL;DR message per variant

  **Test References**:
  - `tests/test_github_client.py` — Existing `format_report` tests; add TL;DR-specific assertions

  **Acceptance Criteria**:

  **TDD:**
  - [ ] TL;DR tests fail before implementation (red)
  - [ ] `source .venv/bin/activate && pytest tests/test_github_client.py -k "test_tldr" -v` → PASS

  **QA Scenarios:**

  ```
  Scenario: PASS report has TL;DR at top
    Tool: Bash
    Preconditions: format_report() updated with TL;DR
    Steps:
      1. Run: source .venv/bin/activate && python3 -c "
         from src.github_client import format_report
         from src.models import *
         r = Report(layers=[LayerResult('layer3', Verdict.PASS, 'ok', [])])
         md = format_report(r)
         lines = md.strip().splitlines()
         assert lines[0] == '## 🧪 Test-Guard Report'
         assert '✅ PASS' in lines[2], f'Expected TL;DR with PASS, got: {lines[2]}'
         assert 'adequate test coverage' in lines[2]
         print('TLDR PASS: OK')
         "
    Expected Result: "TLDR PASS: OK"
    Evidence: .sisyphus/evidence/task-5-tldr-pass.txt

  Scenario: FAIL report has TL;DR at top
    Tool: Bash
    Preconditions: format_report() updated
    Steps:
      1. Run: source .venv/bin/activate && python3 -c "
         from src.github_client import format_report
         from src.models import *
         r = Report(layers=[LayerResult('layer1', Verdict.FAIL, 'bad', [])])
         md = format_report(r)
         lines = md.strip().splitlines()
         assert '❌ FAIL' in lines[2], f'Expected TL;DR with FAIL, got: {lines[2]}'
         assert 'lack adequate' in lines[2]
         print('TLDR FAIL: OK')
         "
    Expected Result: "TLDR FAIL: OK"
    Evidence: .sisyphus/evidence/task-5-tldr-fail.txt
  ```

  **Commit**: YES (commit 4)
  - Message: `✨ feat(report): add outcome TL;DR as first line of report body`
  - Files: `src/github_client.py`, `tests/test_github_client.py`
  - Pre-commit: `source .venv/bin/activate && pytest -v`

- [x] 6. README example output update

  **What to do**:
  - Update the `### Example Output` section in `README.md` to reflect v1.4.0 format:
    - Add TL;DR line after `## 🧪 Test-Guard Report`
    - Replace `"### Layer 1:"` with `"### Coverage Analysis:"`
    - Replace `"### Layer 2:"` with `"### Test File Matching:"`
    - Replace `"### Layer 3:"` with `"### Per-File Evaluation:"`
    - Remove `"(advisory)"` from Layer 2 heading
    - Shorten the AI reason in the L3 table to ~15 words
  - Also update any other README references to "Layer 1/2/3" in the reporting section (but NOT in the "How It Works" / "Pipeline" section — those describe internal architecture and are fine)

  **Must NOT do**:
  - Do NOT change the Pipeline/How It Works section layer references (internal architecture docs)
  - Do NOT change the input table or any non-report sections

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Documentation-only edit
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Tasks 1-5)
  - **Blocks**: Task 7 (regression)
  - **Blocked By**: Tasks 1, 2, 3, 4, 5

  **References**:

  **Pattern References**:
  - `README.md` — Search for `### Example Output` section — this is the ONLY section to update

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: README example uses new layer names
    Tool: Bash
    Preconditions: README.md updated
    Steps:
      1. Run: grep -c "Coverage Analysis\|Test File Matching\|Per-File Evaluation" README.md
      2. Assert count ≥ 3 (one per layer in example)
      3. Run: sed -n '/Example Output/,/^---$/p' README.md | grep -c "Layer 1\|Layer 2\|Layer 3"
      4. Assert count = 0 (no old names in example section)
    Expected Result: New names present, old names absent in example
    Evidence: .sisyphus/evidence/task-6-readme-names.txt

  Scenario: README example has TL;DR line
    Tool: Bash
    Preconditions: README.md updated
    Steps:
      1. Run: sed -n '/Example Output/,/^---$/p' README.md | grep -c "⚠️ WARNING"
      2. Assert count ≥ 2 (TL;DR + Result line)
    Expected Result: TL;DR present in example
    Evidence: .sisyphus/evidence/task-6-readme-tldr.txt
  ```

  **Commit**: YES (commit 5)
  - Message: `📝 docs(readme): update example output for v1.4.0 report format`
  - Files: `README.md`
  - Pre-commit: `source .venv/bin/activate && pytest -v`

- [x] 7. Full regression + safety verification

  **What to do**:
  - Run complete test suite: `cd /home/hashashiyyin/tools/working_dir/test-guard && source .venv/bin/activate && pytest -v`
  - Verify ALL tests pass (282 existing + new tests)
  - Run safety check: `grep -rn '"layer1"\|"layer2"\|"layer3"' src/models.py src/layer1_coverage.py src/layer2_heuristic.py src/layer3_ai.py src/main.py` — verify internal identifiers PRESERVED
  - Run display check: verify `format_report()` output contains new names, not old
  - Tag `v1.4.0` and update floating `v1` tag

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Run commands and verify output
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after everything)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 6

  **References**:
  - All source files and test files from Tasks 1-6

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: Full test suite passes
    Tool: Bash
    Steps:
      1. Run: cd /home/hashashiyyin/tools/working_dir/test-guard && source .venv/bin/activate && pytest -v
      2. Assert exit code 0
      3. Assert "passed" in output, "failed" count = 0
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-7-full-regression.txt

  Scenario: Internal layer identifiers preserved
    Tool: Bash
    Steps:
      1. Run: grep -c '"layer1"\|"layer2"\|"layer3"' src/models.py src/layer1_coverage.py src/layer2_heuristic.py src/layer3_ai.py src/main.py
      2. Assert total > 0
    Expected Result: Internal identifiers still present
    Evidence: .sisyphus/evidence/task-7-identifiers.txt
  ```

  **Commit**: YES (commit 6 — tag only)
  - `git tag v1.4.0` + `git tag -f v1` + `git push origin HEAD` + `git push origin v1.4.0` + `git push origin -f v1`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, grep for strings). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `pytest -v` full suite. Review all changed files for: unused imports, dead code, inconsistent naming. Check AI slop: excessive comments, over-abstraction, generic names (data/result/item/temp). Verify no internal layer identifiers (`"layer1"`, `"layer2"`, `"layer3"`) were renamed.
  Output: `Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Run the local PR replay tool against Matecat PR #4515 fixture data. Verify output matches the approved example: 4 files in L1 table (not 10), TL;DR present, human layer names, no "(advisory)", concise AI reasons. Save actual output to evidence.
  Output: `Scenarios [N/N pass] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

| Commit | Message | Files | Pre-commit |
|--------|---------|-------|------------|
| 1 | `✨ feat(prompt): instruct AI to produce concise reasons` | `prompts/test_adequacy.txt`, `tests/test_layer3_ai.py` | `pytest -v` |
| 2 | `✨ feat(report): rename layers to human-readable names and drop advisory suffix` | `src/github_client.py`, `tests/test_github_client.py` | `pytest -v` |
| 3 | `✨ feat(pipeline): pre-filter diff_files before L1 to exclude non-source files` | `src/main.py`, `src/layer1_coverage.py`, `tests/test_main.py`, `tests/test_layer1_coverage.py` | `pytest -v` |
| 4 | `✨ feat(report): add outcome TL;DR as first line of report body` | `src/github_client.py`, `tests/test_github_client.py` | `pytest -v` |
| 5 | `📝 docs(readme): update example output for v1.4.0 report format` | `README.md` | `pytest -v` |
| 6 | Tag `v1.4.0` + update floating `v1` tag | — | `pytest -v` |

---

## Success Criteria

### Verification Commands
```bash
cd /home/hashashiyyin/tools/working_dir/test-guard && source .venv/bin/activate && pytest -v
# Expected: ALL tests pass (282 existing + new tests, 0 failures)

grep -c "Layer 1\|Layer 2\|Layer 3\|(advisory)" src/github_client.py
# Expected: 0 (no old display names in format_report logic — internal identifiers stay as-is)

grep -c '"layer1"\|"layer2"\|"layer3"' src/models.py src/layer1_coverage.py src/layer2_heuristic.py src/layer3_ai.py src/main.py
# Expected: >0 (internal identifiers PRESERVED — they must NOT be renamed)
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] README example matches actual output format
