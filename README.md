# 🧪 Test-Guard

**Test-Guard is a GitHub Action that gates pull requests on test adequacy.** It uses a 3-layer hybrid pipeline to ensure your changes are covered by tests, moving from fast deterministic checks to semantic AI analysis only when necessary.

## How it works

Test-Guard processes your PR through three sequential layers. Each layer acts as a gate. If a layer produces an all-PASS verdict, the pipeline short-circuits and stops early. This ensures cheap and fast checks run first, while expensive AI analysis only executes when truly needed for ambiguous changes.

### Pipeline flow

```text
PR Opened
   │
   ▼
Layer 1: Diff Coverage ─── [PASS] ──► Done
   │
   ▼
Layer 2: File Matching ─── [PASS] ──► Done
   │
   ▼
Layer 3: AI Analysis   ─── [PASS/FAIL/WARN] ──► Final Verdict
```

---

## The 3-Layer Pipeline

### Layer 1: Diff Coverage
This layer calculates the percentage of your changed lines that are covered by existing tests.

*   **Mechanism:** Uses `diff-cover` to compare your coverage report (XML or LCOV) against the `git diff`.
*   **Trigger:** Runs if a `coverage-file` is provided.
*   **Outcomes:**
    *   **PASS:** Coverage meets or exceeds the `coverage-threshold` (default: 80%).
    *   **FAIL:** Coverage is below the threshold. The pipeline proceeds to Layer 2.
    *   **SKIP:** No coverage file provided or the tool fails. The pipeline proceeds to Layer 2.
*   **Short-circuit:** Stops the pipeline if the verdict is PASS.

### Layer 2: File-Matching Heuristic
This layer checks if every modified source file has a corresponding test file that was also modified in the PR.

*   **Mechanism:** Maps source files to test files using language-specific naming conventions across 19 supported languages.
*   **Trigger:** Always runs if Layer 1 didn't short-circuit.
*   **File filtering:** Before checking for test files, Layer 2 silently skips three categories of changed files:
    1. Files matching exclusion patterns (config, docs, data formats, per-language conventions)
    2. Test files themselves (no need to require tests for tests)
    3. Files whose extension doesn't match any known source language (e.g., `Dockerfile`, `.neon`, `.env`)
*   **Per-file verdicts** (for remaining source files):
    *   **PASS:** A matching test file was found and modified in this PR.
    *   **WARNING:** A matching test file exists in the repo but was not modified.
    *   **FAIL:** No matching test file could be found for the source file.
*   **Short-circuit:** Stops the pipeline if ALL file verdicts are PASS.

### Layer 3: AI Semantic Analysis
The final layer uses AI to evaluate whether your tests actually cover the new behavior and edge cases in your code.

*   **Mechanism:** Sends sanitized diffs and related test file contents to GPT-5-mini via the GitHub Models API. The AI evaluates behavioral coverage, edge cases, regression protection, and integration quality. It returns a structured JSON verdict with per-file analysis and a confidence score.
*   **Trigger:** Runs for files marked FAIL or WARNING by Layer 2, provided `ai-enabled` is true.
*   **Per-file verdicts:**
    *   **PASS:** AI confirms the tests are adequate for the changes.
    *   **FAIL:** AI identifies significant behavioral changes with no corresponding tests.
    *   **WARNING:** AI found issues but the confidence score is below the `ai-confidence-threshold` (default: 0.7), downgrading a FAIL to a non-blocking warning.
*   The AI does **not** penalize trivial changes (whitespace, comments, renames), config files, or documentation.

---

## Verdict System

Test-Guard computes a global verdict from all layers with priority: **FAIL > WARNING > PASS > SKIP**.

| Verdict | Meaning | Exit Code | Status Check |
| :--- | :--- | :--- | :--- |
| **PASS** | All changes are adequately tested. | 0 | Success |
| **FAIL** | Tests are missing or inadequate. Blocks the PR. | 1 | Failure |
| **WARNING** | Minor issues or low-confidence AI failure. Non-blocking. | 0 | Success |
| **SKIP** | All layers skipped (e.g., no coverage file, AI disabled). | 0 | Success |

---

## Reporting

Test-Guard provides feedback in two ways:
1.  **PR Comment:** A detailed markdown report is posted to the pull request with per-layer results and a per-file verdict table.
2.  **Check Run:** A check run named **Test-Guard** appears in the PR's checks tab with a full markdown summary. This can be used as a required status check to block merges if the check fails.

---

## Supported Languages

Layer 2 auto-detects test files for 19 languages:

| Language | Test conventions |
| :--- | :--- |
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

## Quick Start

Test-Guard is free to use via the GitHub Models API and your standard `GITHUB_TOKEN`. No external API keys are required.

```yaml
name: Test-Guard
on:
  pull_request:
    types: [opened, synchronize]

permissions:
  contents: read
  pull-requests: write
      checks: write
  models: read # Required for Layer 3 AI analysis

jobs:
  test-guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run tests with coverage
        run: pytest --cov --cov-report=xml  # your test command here

      - name: Test-Guard
        uses: ostico/test-guard@v1
        with:
          coverage-file: coverage.xml
```

> **Note:** The `models: read` permission is mandatory for AI analysis. If you set `ai-enabled: 'false'`, you can omit this permission.

---

## Inputs

| Input | Default | Description |
| :--- | :--- | :--- |
| `coverage-file` | _(none)_ | Path to coverage report (Cobertura, Clover, JaCoCo, or LCOV). Layer 1 skips if not provided. |
| `coverage-threshold` | `80` | Minimum diff-coverage % to auto-pass. |
| `test-patterns` | `auto` | Source-to-test mapping. Auto-detects 19 languages (see table). |
| `exclude-patterns` | _(see below)_ | Comma-separated glob patterns to skip. |
| `ai-enabled` | `true` | Enable Layer 3 AI analysis. |
| `ai-model` | `openai/gpt-5-mini` | GitHub Models model ID. |
| `ai-confidence-threshold` | `0.7` | AI failures below this score become warnings. |

**Default exclude patterns:**

```text
*.json, *.yml, *.yaml, *.md, *.txt, *.lock, *.toml, *.cfg, *.ini, *.sql,
migrations/**, docs/**,
*.config.js, *.config.ts, *.config.mjs, *.config.cjs, Gruntfile.js, Gulpfile.js,
conftest.py, setup.py, manage.py, noxfile.py, fabfile.py,
build.rs
```

These cover data/markup files, common directories, and per-language configuration conventions (JS/TS, Python, Rust) that match source file extensions but should never require tests.

---

## Output Example

```markdown
## 🧪 Test-Guard Report

### Layer 1: ❌ FAIL
Changed lines: 45% covered (threshold: 80%)

### Layer 2: ❌ FAIL
File matching: 1 pass, 1 fail

| File | Verdict | Reason |
| :--- | :--- | :--- |
| `src/auth.py` | ✅ pass | Test modified: tests/test_auth.py |
| `src/billing.py` | ❌ fail | No matching test file |

### Layer 3: ⚠️ WARNING
AI verdict: warning (confidence: 82%)

| File | Verdict | Reason |
| :--- | :--- | :--- |
| `src/billing.py` | ❌ fail | New discount logic has no edge case test for negative amounts |

**Result: ⚠️ WARNING**
```

---

## Supported Coverage Formats

Layer 1 uses [diff-cover](https://github.com/Bachmann1234/diff-cover), which automatically detects these formats:

| Format | Extension | Detection |
| :--- | :--- | :--- |
| **Cobertura** | `.xml` | Default fallback for XML |
| **Clover** | `.xml` | Detected by `[@clover]` root attribute |
| **JaCoCo** | `.xml` | Detected by JaCoCo DTD structure |
| **LCOV** | `.info` | Any non-XML file |

---

## Examples

### Without coverage (Heuristic + AI only)
```yaml
- uses: ostico/test-guard@v1
  # Layer 1 skips if coverage-file is omitted
```

### Without AI (Heuristic only)
```yaml
- uses: ostico/test-guard@v1
  with:
    ai-enabled: 'false'
```

---

## License

MIT
