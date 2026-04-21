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

## Supported languages

Layer 2 auto-detects test files for **15 languages** out of the box:

| Language   | Test conventions                                           |
|------------|------------------------------------------------------------|
| Python     | `test_{name}.py`, `{name}_test.py`                         |
| JavaScript | `{name}.test.js`, `{name}.spec.js`, `__tests__/{name}.js` |
| JSX        | `{name}.test.jsx`, `{name}.spec.jsx`, `__tests__/{name}.jsx` |
| TypeScript | `{name}.test.ts`, `{name}.spec.ts`, `__tests__/{name}.ts` |
| TSX        | `{name}.test.tsx`, `{name}.spec.tsx`, `__tests__/{name}.tsx` |
| PHP        | `{name}Test.php`                                           |
| Go         | `{name}_test.go`                                           |
| Java       | `{name}Test.java`                                          |
| Kotlin     | `{name}Test.kt`                                            |
| Ruby       | `{name}_spec.rb`, `test_{name}.rb`                         |
| Rust       | `tests/{name}.rs`                                          |
| C#         | `{name}Tests.cs`, `{name}Test.cs`                          |
| Swift      | `{name}Tests.swift`, `{name}Test.swift`                    |
| Scala      | `{name}Spec.scala`, `{name}Test.scala`                     |
| C/C++      | `test_{name}.c`, `test_{name}.cpp`, `test_{name}.cc`       |
| Elixir     | `{name}_test.exs`                                          |
| Dart       | `{name}_test.dart`                                         |
| Lua        | `test_{name}.lua`, `{name}_spec.lua`                       |

Layer 1 accepts any coverage format (see below). Layer 3 is language-agnostic.

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
  models: read

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

> **Required**: `models: read` permission is mandatory when AI is enabled (the default).
> Without it, Layer 3 will get a 403 error. If you set `ai-enabled: 'false'`, you can omit it.

## Inputs

| Input                     | Default             | Description                                                          |
|---------------------------|---------------------|----------------------------------------------------------------------|
| `coverage-file`           | _(none)_            | Path to coverage report (Cobertura XML, Clover XML, JaCoCo XML, or LCOV). Layer 1 skips if not provided. |
| `coverage-threshold`      | `80`                | Min diff-coverage % to auto-pass.                                    |
| `test-patterns`           | `auto`              | Source→test file mapping. Auto-detects 15+ languages (see table above). |
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
