# 🧪 Test-Guard

**A GitHub Action that gates pull requests on test adequacy.** Test-Guard runs a 3-layer hybrid pipeline — fast deterministic checks first, semantic AI analysis only for files that can't be resolved automatically.

## How It Works

Test-Guard evaluates every source file in your PR independently through three layers:

```text
PR Opened
   │
   ▼
Layer 1: Diff Coverage ─── [all files ≥ threshold] ──► PASS (done)
   │
   ▼
Layer 2: File Matching ─── advisory hints for Layer 3
   │                        (short-circuits only when AI is disabled)
   ▼
Layer 3: Per-File Analysis
   ├── Gate 1-8: Deterministic shortcuts (coverage + test relevance + triviality)
   │   resolves most files without AI
   └── AI fallthrough: only ambiguous files sent to GPT-5-mini
       │
       ▼
   Layer 3 verdict overrides all other layers
```

**Key design principle:** Layer 3 performs a from-scratch per-file evaluation. It doesn't inherit Layer 2's verdicts — it uses Layer 2 hints as advisory input alongside coverage data, test diffs, and triviality detection to reach its own conclusions.

---

## The 3-Layer Pipeline

### Layer 1: Diff Coverage

Calculates the percentage of changed lines covered by existing tests, per file.

- **Mechanism:** Runs `diff-cover` against your coverage report (XML or LCOV).
- **Short-circuit:** PASS when **every** source file meets or exceeds the threshold.
- **Output:** Per-file coverage percentages forwarded to Layer 3 for use in shortcuts.
- **SKIP:** No coverage file provided or diff-cover fails. Pipeline continues.

### Layer 2: File-Matching Heuristic

Checks whether each modified source file has a corresponding test file using naming conventions across 19 languages.

- **Silently skips:** Excluded files, test files themselves, unrecognized extensions.
- **Per-file verdicts:**
  - **PASS:** Matching test file found and modified in this PR.
  - **WARNING:** Matching test file exists in the repo but wasn't modified.
  - **FAIL:** No matching test file found.
- **Advisory mode (AI enabled):** Layer 2 never short-circuits. Its matched-test hints feed into Layer 3 but don't determine the final verdict.
- **Gate mode (AI disabled):** Layer 2 short-circuits on all-PASS, and its verdict is final.

### Layer 3: Per-File AI Analysis

The authoritative layer. Evaluates each source file through deterministic shortcuts first, falling back to AI only for files that can't be resolved.

**Deterministic shortcuts (Gates 1–8):**

Each source file is evaluated against these gates in order. The first matching gate produces a verdict and skips AI for that file:

| Gate | Condition | Verdict | Rationale |
|:-----|:----------|:--------|:----------|
| 1 | File was deleted | SKIP | No remaining code to test |
| 2 | Diff is trivial (whitespace/comments only) | SKIP | Trivial changes don't need tests |
| 3 | Coverage ≥ threshold (any test relevance) | PASS | Existing tests already cover the changes |
| 4 | No relevant tests in PR + no/low coverage | FAIL | No evidence of test coverage at all |
| 5 | Coverage < threshold + relevant tests exist (YES) | FAIL | Tests exist but don't cover enough |
| 6 | Coverage < threshold + ambiguous test relevance (UNKNOWN) | → AI | AI determines if changed tests actually target this file |
| 7 | No coverage data + relevant tests exist (YES) | → AI | AI cross-references test diffs against source diffs |
| 8 | No coverage data + ambiguous test relevance (UNKNOWN) | → AI | Most uncertain case — AI judges both relevance and adequacy |

**Test relevance** is a tri-state (YES / NO / UNKNOWN):
- **YES:** Layer 2 matched a test, or a changed test file's name/content references the source file.
- **NO:** No test files were changed in this PR at all.
- **UNKNOWN:** Test files were changed but none could be linked to this source file.

**AI fallthrough:** Only files reaching Gates 6–8 are sent to the AI. The prompt includes source diffs, all changed test diffs (with matched/candidate annotations), per-file coverage data, and Layer 2 hints. AI returns a per-file verdict with a confidence score — FAIL verdicts below the confidence threshold are downgraded to WARNING.

**AI failure handling:** If the API call fails, Layer 3 returns SKIP, and the final verdict falls back to Layer 1 + Layer 2 worst-wins (degraded strict mode).

---

## Verdict System

Layer 3's verdict is authoritative when it runs:

| Scenario | Final Verdict |
|:---------|:-------------|
| Layer 3 returns PASS/FAIL/WARNING | Layer 3's verdict (overrides L1 and L2) |
| Layer 3 returns SKIP (AI failure) | Worst of Layer 1 + Layer 2 (fallback) |
| AI disabled (no Layer 3) | Worst of Layer 1 + Layer 2 |

Priority within a layer: **FAIL > WARNING > PASS > SKIP**.

| Verdict | Meaning | Exit Code | Status Check |
|:--------|:--------|:----------|:-------------|
| **PASS** | All changes adequately tested | 0 | ✅ Success |
| **FAIL** | Tests missing or inadequate — blocks the PR | 1 | ❌ Failure |
| **WARNING** | Minor gaps or low-confidence AI result — non-blocking | 0 | ✅ Success |
| **SKIP** | All layers skipped (no coverage file, AI disabled, etc.) | 0 | ✅ Success |

---

## Quick Start

Test-Guard uses the GitHub Models API with your standard `GITHUB_TOKEN`. No external API keys required.

```yaml
name: Test-Guard
on:
  pull_request:
    types: [opened, synchronize]

permissions:
  contents: read
  pull-requests: write
  checks: write
  models: read  # Required for Layer 3 AI analysis

jobs:
  test-guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run tests with coverage
        run: pytest --cov --cov-report=xml  # your test command

      - name: Test-Guard
        uses: ostico/test-guard@v1
        with:
          coverage-file: coverage.xml
```

> The `models: read` permission is required for AI analysis. If you set `ai-enabled: 'false'`, you can omit it.

---

## Inputs

| Input | Default | Description |
|:------|:--------|:------------|
| `coverage-file` | _(none)_ | Path to coverage report (Cobertura, Clover, JaCoCo, or LCOV). Layer 1 skips if omitted. |
| `coverage-threshold` | `80` | Minimum diff-coverage % to auto-pass. |
| `test-patterns` | `auto` | Source-to-test mapping. Auto-detects 19 languages. |
| `exclude-patterns` | _(see below)_ | Comma-separated glob patterns to skip. |
| `ai-enabled` | `true` | Enable Layer 3 AI analysis. |
| `ai-model` | `openai/gpt-5-mini` | GitHub Models model ID. |
| `ai-confidence-threshold` | `0.7` | AI FAIL verdicts below this confidence become WARNING. |

**Default exclude patterns:**

```text
*.json, *.yml, *.yaml, *.md, *.txt, *.lock, *.toml, *.cfg, *.ini, *.sql,
migrations/**, docs/**,
*.config.js, *.config.ts, *.config.mjs, *.config.cjs, Gruntfile.js, Gulpfile.js,
conftest.py, setup.py, manage.py, noxfile.py, fabfile.py,
build.rs
```

---

## Reporting

Test-Guard reports results in two places:

1. **PR Comment:** Markdown report with per-layer results and a per-file verdict table.
2. **Check Run:** A **Test-Guard** check appears in the PR's checks tab. Can be set as a required status check to block merges.

### Example Output

```markdown
## 🧪 Test-Guard Report

### Layer 1: ❌ FAIL
Changed lines: 45% covered (threshold: 80%)

### Layer 2: ❌ FAIL (advisory)
File matching: 1 pass, 1 fail

| File | Verdict | Reason |
|:-----|:--------|:-------|
| `src/auth.py` | ✅ pass | Test modified: tests/test_auth.py |
| `src/billing.py` | ❌ fail | No matching test file |

### Layer 3: ⚠️ WARNING
Per-file analysis (2 shortcut, 1 AI):

| File | Verdict | Reason |
|:-----|:--------|:-------|
| `src/auth.py` | ✅ pass | shortcut → coverage 92% ≥ 80% |
| `src/utils.py` | ⏭️ skip | shortcut → trivial (whitespace only) |
| `src/billing.py` | ⚠️ warning | AI: new discount logic partially covered (confidence: 62%) |

**Result: ⚠️ WARNING** (Layer 3 authoritative)
```

---

## Supported Languages

Layer 2 auto-detects test files for 19 languages:

| Language | Test conventions |
|:---------|:----------------|
| Python | `tests/test_{name}.py`, `**/{name}_test.py` |
| JavaScript | `**/{name}.test.js`, `**/{name}.spec.js`, `**/__tests__/{name}.js` |
| JSX | `**/{name}.test.jsx`, `**/{name}.spec.jsx`, `**/__tests__/{name}.jsx` |
| TypeScript | `**/{name}.test.ts`, `**/{name}.spec.ts`, `**/__tests__/{name}.ts` |
| TSX | `**/{name}.test.tsx`, `**/{name}.spec.tsx`, `**/__tests__/{name}.tsx` |
| PHP | `tests/{name}Test.php` |
| Go | `**/{name}_test.go` |
| Java | `**/{name}Test.java` |
| Kotlin | `**/{name}Test.kt` |
| Ruby | `**/{name}_spec.rb`, `**/test_{name}.rb` |
| Rust | `tests/{name}.rs` |
| C# | `**/{name}Tests.cs`, `**/{name}Test.cs` |
| Swift | `**/{name}Tests.swift`, `**/{name}Test.swift` |
| Scala | `**/{name}Spec.scala`, `**/{name}Test.scala` |
| C | `**/test_{name}.c` |
| C++ | `**/test_{name}.cpp`, `**/test_{name}.cc`, `**/test_{name}.cxx` |
| Elixir | `test/**/{name}_test.exs` |
| Dart | `test/**/{name}_test.dart` |
| Lua | `**/test_{name}.lua`, `**/{name}_spec.lua` |

---

## Supported Coverage Formats

Layer 1 uses [diff-cover](https://github.com/Bachmann1234/diff-cover), which auto-detects:

| Format | Extension | Detection |
|:-------|:----------|:----------|
| Cobertura | `.xml` | Default for XML |
| Clover | `.xml` | `[@clover]` root attribute |
| JaCoCo | `.xml` | JaCoCo DTD structure |
| LCOV | `.info` | Any non-XML file |

---

## Examples

### Coverage + heuristics + AI (default)
```yaml
- uses: ostico/test-guard@v1
  with:
    coverage-file: coverage.xml
```

### Heuristics + AI only (no coverage file)
```yaml
- uses: ostico/test-guard@v1
# Layer 1 skips, Layer 3 shortcuts use test relevance only
```

### Heuristics only (no AI)
```yaml
- uses: ostico/test-guard@v1
  with:
    ai-enabled: 'false'
# Layer 2 becomes the gate (short-circuits on all-PASS)
```

### Strict threshold with AI
```yaml
- uses: ostico/test-guard@v1
  with:
    coverage-file: coverage.xml
    coverage-threshold: '95'
    ai-confidence-threshold: '0.8'
```

---

## License

MIT
