# 🧪 test-guard

**Hybrid PR test adequacy gate for GitHub Actions.**

Three layers of defense — deterministic checks first, AI only when needed.

## How it works

| Layer       | What                           | Cost  | Speed |
|-------------|--------------------------------|-------|-------|
| **Layer 1** | Diff-coverage on changed lines | Free  | <1s   |
| **Layer 2** | Source→test file matching      | Free  | <1s   |
| **Layer 3** | GPT-5-mini semantic analysis   | Free* | ~5s   |

*Free via GitHub Models API using your existing `GITHUB_TOKEN`.

Layer 1 and 2 handle ~80% of cases. Layer 3 only fires for ambiguous files.

## Quick start

```yaml
name: Test Guard
on:
  pull_request:
    types: [ opened, synchronize ]

permissions:
  contents: read
  pull-requests: write
  statuses: write

jobs:
  test-guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run your tests with coverage
        run: pytest --cov --cov-report=xml  # your test command here

      - name: Test Guard
        uses: ostico/test-guard@v1
        with:
          coverage-file: coverage.xml
```

No API keys needed. Uses `GITHUB_TOKEN` for everything.

## Inputs

| Input                     | Default             | Description                                                          |
|---------------------------|---------------------|----------------------------------------------------------------------|
| `coverage-file`           | _(none)_            | Path to coverage report (Cobertura XML, Clover XML, JaCoCo XML, or LCOV). Layer 1 skips if not provided. |
| `coverage-threshold`      | `80`                | Min diff-coverage % to auto-pass.                                    |
| `test-patterns`           | `auto`              | Source→test file mapping. Auto-detects Python, PHP, JS/TS, Go, Java. |
| `exclude-patterns`        | `*.md,docs/**,...`  | Files to skip.                                                       |
| `ai-enabled`              | `true`              | Enable Layer 3 AI analysis.                                          |
| `ai-model`                | `openai/gpt-5-mini` | GitHub Models model ID.                                              |
| `ai-confidence-threshold` | `0.7`               | Below this, AI "fail" becomes "warning".                             |

## Output example

```markdown
## 🧪 Test Guard Report

### Layer 1: ❌ FAIL
Changed lines: 45% covered (threshold: 80%)

### Layer 2: ❌ FAIL
File matching: 1 pass, 1 fail

| File | Verdict | Reason |
|---|---|---|
| `src/auth.py` | ✅ pass | Test modified: tests/test_auth.py |
| `src/billing.py` | ❌ fail | No matching test file |

### Layer 3: ⚠️ WARNING
AI verdict: warning (confidence: 82%)

| File | Verdict | Reason |
|---|---|---|
| `src/billing.py` | ❌ fail | New discount logic has no edge case test for negative amounts |

**Result: ⚠️ WARNING**
```

## Supported coverage formats

Layer 1 uses [diff-cover](https://github.com/Bachmann1234/diff-cover) under the hood, which auto-detects:

| Format         | File extension | Detection                              |
|----------------|----------------|----------------------------------------|
| **Cobertura**  | `.xml`         | Default fallback for XML files         |
| **Clover**     | `.xml`         | Auto-detected by `[@clover]` root attr |
| **JaCoCo**     | `.xml`         | Auto-detected by JaCoCo DTD structure  |
| **LCOV**       | `.info`, other | Any non-XML file                       |

```yaml
# Examples
coverage-file: coverage.xml        # Cobertura / Clover / JaCoCo (auto-detected)
coverage-file: lcov.info           # LCOV
coverage-file: coverage/lcov.info  # LCOV (path works too)
```

## Without coverage (heuristic + AI only)

```yaml
- uses: ostico/test-guard@v1
  # No coverage-file — Layer 1 skips, Layer 2+3 still run
```

## Without AI (heuristic only)

```yaml
- uses: ostico/test-guard@v1
  with:
    ai-enabled: 'false'
    # Only Layer 1 + Layer 2 run — zero API calls
```

## License

MIT
