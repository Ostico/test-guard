# Diff-compaction benchmark

Reproducible measurement of the Layer 3 diff-compaction added in the
`unidiff` work (`src/layer3_ai.py`: `_compact_diff` / `_trim_hunk` /
`_truncate_diff`). It answers two questions on a **real multi-file PR**:

1. **How many tokens does compaction save?** (RAW vs AFTER vs the old BEFORE)
2. **Is the compressed diff still understandable to a review LLM?**

## What it compares

| Label  | Behaviour |
|--------|-----------|
| RAW    | untouched diff — a naive prompt |
| BEFORE | pre-unidiff `_sanitize_diff`: blind character-cut at the char cap. Reproduced **in-script** (`baseline_sanitize`) so it runs on any branch, before or after the change |
| AFTER  | current `src.layer3_ai._sanitize_diff`: unidiff compaction + hunk-boundary truncation |

Totals are also split **source vs test**, because a test-adequacy gate cares
most about the *test* diffs surviving truncation.

## Telling an intelligent shrink from a dumb one

Token count **alone cannot** judge shrink quality — the dumbest shrink
(delete everything) wins on tokens and is useless. So the benchmark measures
**two axes**:

1. **Size** — total tokens (must fit the budget).
2. **Signal retention** — `% of changed (+/-) lines kept`, split by role.
   Changed lines are the actual signal; context lines are filler.

A shrink is **intelligent** when, at the same-or-lower token cost, it keeps
**more TEST-file signal** — because the tool's job is judging test adequacy.
The concrete rule: **TEST kept% ≥ SOURCE kept%** (protect tests first), ideally
≥ the BEFORE baseline. The `--assert-intelligent` flag turns this into a
regression gate (exit non-zero when TEST kept% < SOURCE kept%):

```bash
python benchmarks/benchmark_compaction.py --assert-intelligent
```

The **ground-truth verdict** check (Part A of the LLM step below) is the final
proof: with real coverage ~100%, the correct verdict is roughly *pass*; an
intelligent shrink keeps enough test signal for the model to land it, a dumb
one starves the model into a wrong/low-confidence guess.

## Files

- `benchmark_compaction.py` — the harness.
- `fixtures/matecat_pr_diffs.json` — a real 29-file PR (MateCat `develop...HEAD`)
  captured as GitHub-style per-file patches (hunks only, no `---/+++` header,
  exactly what GitHub's PR-files API returns in `patch`). Committed so the
  benchmark runs standalone with **no external checkout**.
- `fixtures/matecat_cov.xml.gz` — the Clover coverage report for the **same**
  PR (the artifact test-guard's Layer 1 consumes). The harness derives per-file
  *changed-line* coverage from it (added executable lines ∩ Clover, covered =
  `count>0`) for the eval prompt's Coverage Summary. gzipped (~200 KB); the
  harness reads `.gz` directly. Override with `--coverage path/to.xml[.gz]`.

## Run it

```bash
# from repo root; token counts use tiktoken if installed, else chars//3
pip install tiktoken            # optional but recommended (real GPT tokens)

python benchmarks/benchmark_compaction.py                 # table + totals
python benchmarks/benchmark_compaction.py --emit-samples  # also write sample files
```

`--emit-samples` writes to `$TMPDIR/tg-bench/` (override with `--out-dir`):

- `compressed_sample.txt` — the biggest file's compacted diff, for eyeballing.
- `eval_prompt.txt` — a full Layer-3 prompt (system + the largest source/test
  pair) used for the LLM-understandability step below.

### Regenerate the diff set from a live repo (instead of the fixture)

```bash
python benchmarks/benchmark_compaction.py --repo /path/to/checkout --base develop
```

## Reproduce the "before vs after" comparison

The harness already prints BEFORE and AFTER side by side in **one run** — no
branch switching needed, because `baseline_sanitize` reproduces the old
char-cut behaviour. To sanity-check against the *actual* historical code,
`git switch main` and re-run; the RAW and BEFORE columns must match.

## LLM-understandability step (the "send it to an agent" test)

1. Generate the prompt: `python benchmarks/benchmark_compaction.py --emit-samples`
2. Hand `eval_prompt.txt` to a capable review agent and ask it to:
   - **Part A** — perform the test-adequacy review the `===SYSTEM===` section
     specifies (verdict / confidence / per-file reasons). Tests whether the
     compressed diff carries enough signal to do the job.
   - **Part B** — score understandability 1–10, confirm the diff is still a
     valid unified diff (well-formed `@@`, no mid-line cuts), and flag any
     change it could not follow because context/hunks were trimmed.

## Reference results (fixture, tiktoken o200k_base)

**Signal retention** (the quality axis — % of changed +/- lines kept):

| role | dumb (uniform cap) | intelligent shrink | old char-cut baseline |
|------|-------------------|--------------------|-----------------------|
| test | 64% | **92%** | 70% |
| source | 68% | 73% | 70% |
| `--assert-intelligent` gate | FAIL | **PASS** | — |

The intelligent shrink (adaptive context ladder + larger test-file cap) lifts
test-signal retention from 64% → 92% and flips the regression gate green,
while source is sacrificed first (as intended). It spends a few more tokens
than the dumb cap — each file still fits its per-call budget; the goal is
fitting the 8k cap with the *right* content, not minimising tokens.

Agent evaluation of the pre-shrink prompt scored **6/10** understandability
and flagged the core defect: truncation dropped *test-file* hunks (~54%) more
than *source* (~35%) — backwards for a test-adequacy gate. The intelligent
shrink addresses exactly that (test file now drops ~2 of 24 hunks, source ~6
of 20). Re-run the LLM step to confirm on your own changes.
