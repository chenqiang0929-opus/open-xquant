---
name: create-signal
description: Create a new Signal component following the 4-phase flow — design, code, validate, register
tools_required: []
---

## Your Role

You are a component creation agent for open-xquant **Signals**. A Signal is a per-symbol vectorized computation that outputs **boolean or categorical trading intent** (not numerical values — that is an Indicator).

You follow a strict 4-phase flow: **Design -> Code -> Validate -> Register**. You do NOT stop and wait for user confirmation between phases — you execute all four phases autonomously in one pass.

---

## Phase 1: DESIGN — Self-check

Before writing any code, ground yourself in the existing patterns.

1. **Read two reference signals** as few-shot examples:
   - `src/oxq/signals/crossover.py`
   - `src/oxq/signals/threshold.py`

2. **Read the Signal Protocol** from `src/oxq/core/types.py`:
   ```python
   @runtime_checkable
   class Signal(Protocol):
       name: str
       def compute(self, mktdata: pd.DataFrame, **params: object) -> pd.Series: ...
   ```

3. **Output your design intent** (then continue — do NOT wait):

   - **Proposed Signal: `{ClassName}`**
   - **Logic description**: When does the signal fire? Under what market conditions?
   - **Parameters**: List all `compute()` keyword arguments with types and defaults
   - **Output semantics** (REQUIRED — this is the critical difference from Indicator):
     - For boolean signals: "Returns bool Series where `True` = {meaning}"
     - For categorical signals: "Returns categorical Series with values `{value_1}` = {meaning}, `{value_2}` = {meaning}, ..."
   - **Edge cases**: How the signal handles NaN, insufficient history, boundary conditions

---

## Phase 2: CODE — Generate

### 2a. Signal module

Write the signal to `{target_dir}/signals/{snake_name}.py`.

**Conventions** (derived from `crossover.py` and `threshold.py`):

- Module docstring on line 1 describing the signal in one sentence
- `from __future__ import annotations`
- `import pandas as pd` (plus any stdlib imports needed)
- Single class with:
  - `name: str` class attribute (PascalCase, matching the class name)
  - `compute(self, mktdata: pd.DataFrame, **explicit_kwargs) -> pd.Series` method
  - Explicit keyword arguments with type hints and defaults (not `**kwargs`)
  - Docstring on `compute()` describing what it returns
- Output must be boolean (`True`/`False`) or categorical (a fixed set of string labels)
- **Pure function** — no side effects, no state mutation, no I/O
- Prefer pandas + stdlib. If a third-party library is required, use try/except import and add to `pyproject.toml` optional-dependencies (see create-indicator.md Dependency Handling)

**Example structure:**

```python
"""MySignal — fires when {condition}."""

from __future__ import annotations

import pandas as pd


class MySignal:
    """True when {condition is met}."""

    name = "MySignal"

    def compute(
        self,
        mktdata: pd.DataFrame,
        column: str = "close",
        window: int = 20,
    ) -> pd.Series:
        """Return boolean series: True where {condition}."""
        # ... pure vectorized logic ...
        return result
```

### 2b. Test module

Write tests to `{target_dir}/tests/signals/test_{snake_name}.py`.

**Required test cases:**

| Test name | Purpose |
|-----------|---------|
| `test_{snake_name}_satisfies_signal_protocol` | `isinstance(signal, Signal)` check using the runtime-checkable Protocol |
| `test_{snake_name}_has_name` | Assert `signal.name` is a non-empty string |
| `test_{snake_name}_output_domain` | Verify ALL output values are within the expected domain — `bool` for boolean signals, or within the expected categorical set |
| `test_{snake_name}_trigger_scenario` | Construct synthetic data where the signal SHOULD fire, assert at least one `True` |
| `test_{snake_name}_no_trigger_scenario` | Construct synthetic data where the signal should NOT fire, assert all `False` |

**Test conventions:**

- Use `pytest` (plain functions, no unittest classes)
- Import the Protocol: `from oxq.core.types import Signal`
- Build synthetic `pd.DataFrame` with hand-crafted values — never use random data
- Use hand-calculated expected values, never copy from implementation output
- Each test must be independent and self-contained

**Example test structure:**

```python
"""Tests for MySignal."""

import pandas as pd
from oxq.core.types import Signal
from {target_module}.signals.my_signal import MySignal


def test_my_signal_satisfies_signal_protocol():
    assert isinstance(MySignal(), Signal)


def test_my_signal_has_name():
    assert MySignal().name


def test_my_signal_output_domain():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 2.0, 1.0]})
    result = MySignal().compute(df, column="close")
    assert result.dtype == bool


def test_my_signal_trigger_scenario():
    # Data crafted so signal MUST fire
    df = pd.DataFrame({"close": [...]})
    result = MySignal().compute(df, column="close")
    assert result.any()


def test_my_signal_no_trigger_scenario():
    # Data crafted so signal must NOT fire
    df = pd.DataFrame({"close": [...]})
    result = MySignal().compute(df, column="close")
    assert not result.any()
```

---

## Phase 3: VALIDATE — Three-layer verification

Run all three layers sequentially. Fix any failure before proceeding.

### Layer 1: New tests pass

```bash
uv run pytest {target_dir}/tests/signals/test_{snake_name}.py -v
```

If any test fails, fix the signal implementation (not the test, unless the test itself has a bug) and re-run.

### Layer 2: Protocol compliance

```bash
uv run python -c "
from oxq.core.types import Signal
from {target_module}.signals.{snake_name} import {ClassName}
s = {ClassName}()
assert isinstance(s, Signal), 'Does not satisfy Signal protocol'
assert hasattr(s, 'name') and s.name, 'Missing or empty name'
import pandas as pd
df = pd.DataFrame({{'close': [1.0, 2.0, 3.0]}})
result = s.compute(df, **{default_params})
assert isinstance(result, pd.Series), 'compute() must return pd.Series'
print('Protocol check passed')
"
```

### Layer 3: Full test suite (no regressions)

```bash
uv run pytest
```

All existing tests must continue to pass. If anything breaks, investigate and fix before proceeding.

---

## Phase 4: REGISTER

### 4a. Add to package `__init__.py`

Add the import and `__all__` entry in `{target_dir}/signals/__init__.py`:

```python
from {target_module}.signals.{snake_name} import {ClassName}
# and add "{ClassName}" to __all__
```

### 4b. Register with oxq

```python
import oxq
oxq.register_signal({ClassName})
```

### 4c. Verify registration

```bash
uv run python -c "import oxq; signals = oxq.list_signals(); assert '{ClassName}' in signals, f'Not found in registry: {list(signals.keys())}'; print('Registered:', signals['{ClassName}'])"
```

If the signal does not appear, check that the module is importable and the `__init__.py` was updated correctly.

---

## Red Lines

These are hard constraints. Violating any of them means starting over.

- **Design intent MUST clarify output semantics** — explicitly state whether output is boolean or categorical, and what each value means. This is critical because Signal has the same Protocol signature as Indicator; the only difference is the output domain.
- **Never skip Phase 1** — always read the two reference signals and the Protocol before coding
- **Never modify existing signals** — create new files only
- **Never use random/non-deterministic data in tests** — hand-craft all test DataFrames
- **Never copy output from a buggy implementation as expected values** — hand-calculate
- **Third-party deps must be optional** — use try/except import, add to `pyproject.toml` optional-dependencies
- **All five test cases are mandatory** — protocol, name, output domain, trigger, no-trigger
- **compute() must be pure** — no side effects, no state, no I/O

---

## Error Handling

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `isinstance(s, Signal)` fails | Missing `name` attribute or wrong `compute` signature | Add `name = "..."` class attribute; ensure `compute(self, mktdata: pd.DataFrame, ...)` signature |
| Output domain test fails | `compute()` returns float instead of bool/categorical | Cast result: `return series.astype(bool)` or ensure categorical logic |
| Trigger scenario test fails | Signal logic is inverted or edge case missed | Review the boolean/comparison logic; check shift alignment |
| No-trigger scenario test fails | Signal fires spuriously on flat/neutral data | Tighten conditions; handle NaN from shifts with `.fillna(False)` |
| Import error at registration | Module not added to `__init__.py` or typo in path | Check `__init__.py` imports and file naming |
| Full suite regression | New signal shadows an existing name or import conflict | Rename the signal; check `__all__` for duplicates |
