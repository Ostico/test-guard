# Layer 3: Token Budget Fix + 413 Model Fallback

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two bugs causing Layer 3 AI to fail with 413 errors on every batch in CI: incorrect token budget constants and missing model fallback on 413 after truncation retry.

**Architecture:** Bug 1 fixes the token budget constants so requests fit within GitHub Models' real 8000-token input limit. Bug 2 extends the model fallback chain (currently 403-only) to also trigger on 413 after truncation retry fails, so gpt-4.1-nano is attempted before giving up.

**Tech Stack:** Python 3.12, pytest, openai SDK, unittest.mock

---

## File Structure

| File | Role |
|---|---|
| `src/layer3_ai.py` | Production code — constants (lines 377–386), `_call_ai_for_batch()` (lines 561–602), `run_layer3()` model loop (lines 772–821) |
| `tests/test_layer3_ai.py` | Test file — existing classes `TestEstimateTokens`, `TestBatchFiles`, `TestCallAiForBatch`, `TestRunLayer3Batching` |

Both bugs are in `src/layer3_ai.py`. All new tests go into `tests/test_layer3_ai.py`.

---

## Context for Implementers

### Current Constants (lines 377–386)

```python
_CHARS_PER_TOKEN = 4
_INPUT_TOKEN_LIMIT = 8192
_SYSTEM_OVERHEAD_TOKENS = 700
_SAFETY_FACTOR = 0.80
_USER_PROMPT_TOKEN_BUDGET = int(
    (_INPUT_TOKEN_LIMIT - _SYSTEM_OVERHEAD_TOKENS) * _SAFETY_FACTOR
)  # ≈ 5993 tokens
_FILE_ENTRY_OVERHEAD_TOKENS = 25
_BATCH_OVERHEAD_TOKENS = 30
_RETRY_MAX_DIFF_CHARS = 3000
```

### Problem

GitHub Models free tier hard-limits gpt-4.1-mini to **8000 input tokens** (not 8192). The `_CHARS_PER_TOKEN = 4` underestimates real tokenization for diffs (~3–3.5 chars/token). Combined, requests land at ~7700–8200 real tokens and get 413'd. The retry path truncates to 3000 chars but still fails because multiple diffs + overhead exceed the limit. After both attempts fail, the code gives up — it does **not** try gpt-4.1-nano because `_is_model_forbidden()` only matches 403.

### Test Helper

Tests use `_make_api_error(status_code, message)` at line 33 to create mock `APIStatusError` instances:

```python
def _make_api_error(status_code: int, message: str = "error") -> APIStatusError:
    mock_response = MagicMock()
    mock_response.status_code = status_code
    return APIStatusError(message=message, response=mock_response, body=None)
```

### Test runner command

```bash
cd /home/hashashiyyin/tools/working_dir/test-guard
source .venv/bin/activate && python -m pytest tests/test_layer3_ai.py -v
```

To run a specific test class or method:

```bash
source .venv/bin/activate && python -m pytest tests/test_layer3_ai.py::TestClassName::test_method -v
```

---

## Task 1: Fix Token Budget Constants (Bug 1)

**Files:**
- Modify: `src/layer3_ai.py:377-386` (constants block)
- Test: `tests/test_layer3_ai.py` (existing `TestEstimateTokens` class, new `TestTokenBudgetConstants` class)

### Rationale for New Values

| Constant | Old | New | Why |
|---|---|---|---|
| `_INPUT_TOKEN_LIMIT` | 8192 | 8000 | GitHub Models free-tier hard limit confirmed in 413 error message |
| `_CHARS_PER_TOKEN` | 4 | 3 | Diffs tokenize at ~3–3.5 chars/token (code + symbols); 3 is conservative |
| `_SYSTEM_OVERHEAD_TOKENS` | 700 | 800 | System prompt is 2421 chars ≈ 807 tokens at 3 chars/token; round up |
| `_SAFETY_FACTOR` | 0.80 | 0.85 | Conservative enough with the corrected base values; 0.80 was compensating for the wrong base |

New budget: `int((8000 - 800) * 0.85)` = **6120 tokens** → ~18360 chars max user prompt (down from ~23972).

- [ ] **Step 1: Add `TestTokenBudgetConstants` test class**

Add this test class at the end of `tests/test_layer3_ai.py`, just before the final `# BUG 1+2` comment block (before line 1518). These tests pin the corrected constant values:

```python
class TestTokenBudgetConstants:
    """Pin the token budget constants to their corrected values."""

    def test_input_token_limit(self):
        assert layer3_ai._INPUT_TOKEN_LIMIT == 8000

    def test_chars_per_token(self):
        assert layer3_ai._CHARS_PER_TOKEN == 3

    def test_system_overhead_tokens(self):
        assert layer3_ai._SYSTEM_OVERHEAD_TOKENS == 800

    def test_safety_factor(self):
        assert layer3_ai._SAFETY_FACTOR == 0.85

    def test_user_prompt_token_budget(self):
        expected = int((8000 - 800) * 0.85)  # 6120
        assert layer3_ai._USER_PROMPT_TOKEN_BUDGET == expected

    def test_budget_leaves_headroom_for_system_prompt(self):
        # Total tokens used = system overhead + user budget must be < input limit
        total = layer3_ai._SYSTEM_OVERHEAD_TOKENS + layer3_ai._USER_PROMPT_TOKEN_BUDGET
        assert total < layer3_ai._INPUT_TOKEN_LIMIT
```

- [ ] **Step 2: Run new tests — verify they FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/test_layer3_ai.py::TestTokenBudgetConstants -v
```

Expected: 5 of 6 tests fail (old values don't match). `test_budget_leaves_headroom_for_system_prompt` might pass with old values too since it's a structural check.

- [ ] **Step 3: Update constants in `src/layer3_ai.py`**

Replace the constants block at lines 377–383 of `src/layer3_ai.py`. Change from:

```python
_CHARS_PER_TOKEN = 4
_INPUT_TOKEN_LIMIT = 8192
_SYSTEM_OVERHEAD_TOKENS = 700   # system prompt + JSON schema overhead
_SAFETY_FACTOR = 0.80
_USER_PROMPT_TOKEN_BUDGET = int(
    (_INPUT_TOKEN_LIMIT - _SYSTEM_OVERHEAD_TOKENS) * _SAFETY_FACTOR
)  # ≈ 5993 tokens
```

To:

```python
_CHARS_PER_TOKEN = 3
_INPUT_TOKEN_LIMIT = 8000
_SYSTEM_OVERHEAD_TOKENS = 800   # system prompt + JSON schema overhead
_SAFETY_FACTOR = 0.85
_USER_PROMPT_TOKEN_BUDGET = int(
    (_INPUT_TOKEN_LIMIT - _SYSTEM_OVERHEAD_TOKENS) * _SAFETY_FACTOR
)  # = 6120 tokens
```

- [ ] **Step 4: Update the docstring in `_estimate_tokens()`**

The docstring at line 390 says "~4 chars per token". Update it to match:

Change from:

```python
def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token, minimum 1.

    The 4-char heuristic is a well-known approximation for English/code text
    with GPT-family tokenizers. It intentionally underestimates to leave
    headroom; the _SAFETY_FACTOR on the budget absorbs the error.
    """
    return len(text) // _CHARS_PER_TOKEN + 1
```

To:

```python
def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~3 chars per token, minimum 1.

    Code diffs tokenize at ~3-3.5 characters per token with GPT-family
    tokenizers (symbols and punctuation inflate token count vs. prose).
    The _SAFETY_FACTOR on the budget provides additional headroom.
    """
    return len(text) // _CHARS_PER_TOKEN + 1
```

- [ ] **Step 5: Update existing `TestEstimateTokens` expectations**

The existing test `test_known_lengths` in class `TestEstimateTokens` asserts `_estimate_tokens("a" * 100) == 26` (i.e., `100 // 4 + 1`). With `_CHARS_PER_TOKEN = 3`, this becomes `100 // 3 + 1 = 34`. Find and update that test:

Change from:

```python
    def test_known_lengths(self):
        assert _estimate_tokens("a" * 100) == 26  # 100/4 + 1
```

To:

```python
    def test_known_lengths(self):
        assert _estimate_tokens("a" * 100) == 34  # 100/3 + 1
```

Also check all tests in `TestEstimateTokens` and `TestBatchFiles` that depend on the old chars-per-token ratio. If any hardcode token expectations based on `len // 4 + 1`, update them to `len // 3 + 1`. Specifically search for any assertion that computes `// 4` and correct to `// 3`.

- [ ] **Step 6: Run ALL tests — verify they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_layer3_ai.py -v
```

Expected: ALL tests pass. If any `TestBatchFiles` tests fail due to changed budget math (batches split differently now), update their `token_budget` parameter or expected batch counts to match the new constants. Tests that pass explicit `token_budget` arguments should be unaffected since they don't use the default.

- [ ] **Step 7: Commit**

```bash
git add src/layer3_ai.py tests/test_layer3_ai.py
git commit -m "fix: correct token budget constants for GitHub Models 8000-token limit

_INPUT_TOKEN_LIMIT 8192→8000 (real free-tier cap),
_CHARS_PER_TOKEN 4→3 (code diffs tokenize denser),
_SYSTEM_OVERHEAD_TOKENS 700→800 (measured from prompt),
_SAFETY_FACTOR 0.80→0.85 (rebalanced with correct base).

New budget: 6120 tokens (~18K chars) vs old 5993 (~24K chars).
Fixes 413 'Request body too large' on every CI batch."
```

---

## Task 2: Add 413 Model Fallback (Bug 2)

**Files:**
- Modify: `src/layer3_ai.py:561-602` (`_call_ai_for_batch`), `src/layer3_ai.py:780-812` (`run_layer3` model loop)
- Test: `tests/test_layer3_ai.py` (new tests in `TestCallAiForBatch` and `TestRunLayer3Batching`)

### Design

Currently, the model loop in `run_layer3()` (line 808) only advances `current_model_idx` on 403. After a 413 where both attempts fail, `_call_ai_for_batch()` returns `(None, exc)` and the loop hits the `else: break` at line 812 — it never tries gpt-4.1-nano.

**Fix approach:** Make `_call_ai_for_batch()` return a signal distinguishing "retryable size error (both attempts failed)" from "non-retryable error". Then the model loop in `run_layer3()` can advance `current_model_idx` on BOTH 403 and exhausted-413, giving gpt-4.1-nano a chance.

Concretely:
1. Add a new helper `_is_size_error_after_retry(exc)` that checks if the exception from `_call_ai_for_batch` is a size error (meaning both normal and truncated attempts failed).
2. In the `run_layer3()` model loop, after `_is_model_forbidden(exc)`, add an `elif _is_retryable_size_error(exc)` branch that also advances `current_model_idx`.

We reuse the existing `_is_retryable_size_error()` since when it's returned from `_call_ai_for_batch`, it already means both attempts failed. The 413 from the retry attempt IS the exc that gets returned.

- [ ] **Step 1: Add test for 413 fallback in `TestCallAiForBatch`**

Add this test to `TestCallAiForBatch` (after `test_includes_all_test_diffs_in_batch_prompt`, around line 1363):

```python
    @patch("src.layer3_ai._call_github_models")
    def test_413_retry_returns_size_error_for_caller(self, mock_call: MagicMock):
        """When both normal and truncated attempts fail with 413,
        the returned exception should be a retryable size error
        so the caller can try a different model."""
        error_413 = _make_api_error(413, "Request body too large for model")
        mock_call.side_effect = [error_413, error_413]
        raw, exc = _call_ai_for_batch(
            batch_files=["src/a.py"],
            source_diffs={"src/a.py": "x" * 20_000},
            test_diffs={},
            coverage_details=None,
            coverage_threshold=80.0,
            matched_tests={"src/a.py": None},
            model="openai/gpt-4.1-mini",
            system_prompt="system",
            token="ghp_fake",
        )
        assert raw is None
        assert exc is not None
        assert _is_retryable_size_error(exc) is True
```

- [ ] **Step 2: Run the test — verify it passes (already true)**

```bash
source .venv/bin/activate && python -m pytest tests/test_layer3_ai.py::TestCallAiForBatch::test_413_retry_returns_size_error_for_caller -v
```

Expected: PASS. This test verifies the precondition — `_call_ai_for_batch()` already returns the 413 exception. This is a characterization test confirming the contract we depend on.

- [ ] **Step 3: Add test for 413-triggers-model-fallback in `TestRunLayer3Batching`**

Add this test to `TestRunLayer3Batching` (after `test_remaining_batches_skip_when_models_exhausted`, around line 1515):

```python
    @patch("src.layer3_ai._call_github_models")
    def test_413_triggers_model_fallback(self, mock_call: MagicMock):
        """When gpt-4.1-mini returns 413 on both normal and truncated attempts,
        the model loop should escalate to gpt-4.1-nano (just like 403)."""
        error_413 = _make_api_error(413, "Request body too large for model")
        mock_call.side_effect = [
            # First call: gpt-4.1-mini, normal attempt → 413
            error_413,
            # Second call: gpt-4.1-mini, truncated retry → 413 again
            error_413,
            # Third call: gpt-4.1-nano → success
            json.dumps({
                "verdict": "pass",
                "confidence": 0.9,
                "files": [{"file": "src/big.py", "verdict": "pass", "reason": "OK"}],
            }),
        ]
        result = run_layer3(
            source_diffs={"src/big.py": "x" * 20_000},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/big.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-4.1-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.PASS
        assert mock_call.call_count == 3
        # Verify model escalation: mini → mini (retry) → nano
        assert mock_call.call_args_list[0][0][0] == "openai/gpt-4.1-mini"
        assert mock_call.call_args_list[1][0][0] == "openai/gpt-4.1-mini"
        assert mock_call.call_args_list[2][0][0] == "openai/gpt-4.1-nano"
```

- [ ] **Step 4: Add test for 413-all-models-exhausted**

Add this test immediately after the previous one:

```python
    @patch("src.layer3_ai._call_github_models")
    def test_413_all_models_exhausted_returns_skip(self, mock_call: MagicMock):
        """When all models fail with 413 (both attempts each), result is SKIP."""
        error_413 = _make_api_error(413, "Request body too large for model")
        # gpt-4.1-mini: 2 attempts (normal + retry), gpt-4.1-nano: 2 attempts
        mock_call.side_effect = [error_413, error_413, error_413, error_413]
        result = run_layer3(
            source_diffs={"src/big.py": "x" * 20_000},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/big.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-4.1-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.SKIP
        assert mock_call.call_count == 4
```

- [ ] **Step 5: Add test for 413-no-fallback-for-custom-model**

```python
    @patch("src.layer3_ai._call_github_models")
    def test_413_no_fallback_for_custom_model(self, mock_call: MagicMock):
        """Custom models have no fallback chain — 413 means SKIP immediately."""
        error_413 = _make_api_error(413, "Request body too large for model")
        mock_call.side_effect = [error_413, error_413]
        result = run_layer3(
            source_diffs={"src/big.py": "x" * 20_000},
            deleted_files=set(),
            test_diffs={"tests/test_stuff.py": "+ def test(): ..."},
            l2_matched_tests={"src/big.py": None},
            coverage_details=None,
            coverage_threshold=80.0,
            model="openai/gpt-5-mini",
            token="ghp_fake",
            confidence_threshold=0.7,
        )
        assert result.verdict == Verdict.SKIP
        assert mock_call.call_count == 2  # normal + retry, no fallback
```

- [ ] **Step 6: Run new tests — verify they FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/test_layer3_ai.py::TestRunLayer3Batching::test_413_triggers_model_fallback tests/test_layer3_ai.py::TestRunLayer3Batching::test_413_all_models_exhausted_returns_skip tests/test_layer3_ai.py::TestRunLayer3Batching::test_413_no_fallback_for_custom_model -v
```

Expected: `test_413_triggers_model_fallback` FAILS (only 2 calls instead of 3, no nano attempt). The other two may fail or pass depending on current behavior — the key one is the first.

- [ ] **Step 7: Modify the model loop in `run_layer3()`**

In `src/layer3_ai.py`, in the `run_layer3()` function, the model loop currently reads (lines 808–812):

```python
                    elif exc and _is_model_forbidden(exc):
                        # 403: this model is not enabled — try the next one.
                        current_model_idx += 1
                    else:
                        break
```

Change to:

```python
                    elif exc and _is_model_forbidden(exc):
                        # 403: this model is not enabled — try the next one.
                        current_model_idx += 1
                    elif exc and _is_retryable_size_error(exc):
                        # 413 after truncation retry: this model can't handle
                        # the payload even truncated — try a smaller model.
                        current_model_idx += 1
                    else:
                        break
```

This is the ONLY production code change for Bug 2. The `_call_ai_for_batch()` already does the truncation retry internally and returns the 413 exception when both attempts fail. We just need the model loop to recognize that exception and escalate.

**Important:** Note that `_call_ai_for_batch` is called inside the `while current_model_idx < len(models)` loop, which is itself inside the `for batch in batches` loop. The `elif` is added to the inner while loop, at the same level as the existing `_is_model_forbidden` check. The `exc` variable at that point is the exception returned from `_call_ai_for_batch`, which — when both 413 attempts fail — will be the 413 `APIStatusError` from the retry.

- [ ] **Step 8: Run the new tests — verify they PASS**

```bash
source .venv/bin/activate && python -m pytest tests/test_layer3_ai.py::TestRunLayer3Batching::test_413_triggers_model_fallback tests/test_layer3_ai.py::TestRunLayer3Batching::test_413_all_models_exhausted_returns_skip tests/test_layer3_ai.py::TestRunLayer3Batching::test_413_no_fallback_for_custom_model tests/test_layer3_ai.py::TestCallAiForBatch::test_413_retry_returns_size_error_for_caller -v
```

Expected: ALL PASS.

- [ ] **Step 9: Run FULL test suite — verify no regressions**

```bash
source .venv/bin/activate && python -m pytest tests/test_layer3_ai.py -v
```

Expected: ALL tests pass. The existing 403 fallback tests should be unaffected since we only added a new `elif` branch.

- [ ] **Step 10: Commit**

```bash
git add src/layer3_ai.py tests/test_layer3_ai.py
git commit -m "fix: escalate to fallback model on 413 after truncation retry fails

When _call_ai_for_batch returns a 413 (both normal and truncated
attempts failed), the model loop now advances to the next model
in the fallback chain — same as 403. This gives gpt-4.1-nano a
chance before giving up and returning SKIP.

Combined with the token budget fix, this provides two layers of
defense: better budgeting prevents most 413s, and fallback handles
the rest."
```

---

## Task 3: Final Verification

- [ ] **Step 1: Run full test suite from project root**

```bash
cd /home/hashashiyyin/tools/working_dir/test-guard
source .venv/bin/activate && python -m pytest tests/ -v
```

Expected: ALL tests pass.

- [ ] **Step 2: Run linter**

```bash
source .venv/bin/activate && python -m ruff check src/layer3_ai.py tests/test_layer3_ai.py
```

Expected: No errors (or only pre-existing ones unrelated to our changes).

- [ ] **Step 3: Verify the computed budget value**

```bash
source .venv/bin/activate && python -c "from src.layer3_ai import _USER_PROMPT_TOKEN_BUDGET, _CHARS_PER_TOKEN; print(f'Budget: {_USER_PROMPT_TOKEN_BUDGET} tokens, ~{_USER_PROMPT_TOKEN_BUDGET * _CHARS_PER_TOKEN} chars')"
```

Expected output: `Budget: 6120 tokens, ~18360 chars`
