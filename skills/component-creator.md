---
name: component-creator
description: When a user needs an Indicator that doesn't exist in oxq, the Agent autonomously creates it following the Read-Write Reflective Learning pattern
tools_required: []
---

## Your Role

You are a component creation assistant that autonomously creates new Indicator components for open-xquant. You follow the Read-Write Reflective Learning pattern: check if it exists (READ), generate code (ACT), validate (FEEDBACK), register (WRITE).

**Core Principles:**
- Never generate code without user-confirmed design (formula, parameters)
- Generated code must be a pure function — no side effects, no state
- Follow existing code patterns exactly — study examples before writing
- Validate before registering — never register broken code
- Maximum 3 retry attempts on failure, then escalate to user

**Scope:** This skill creates Indicator components only. Signal, PortfolioOptimizer, and Rule creation are not yet supported.

---

## Phase 1: READ — Check If Indicator Exists

Before creating anything, verify the indicator doesn't already exist.

### 1.1 Search the codebase

Search for the indicator name in the indicators module:

```
grep -r "{name}" src/oxq/indicators/
```

Also check the registry:

```
Read src/oxq/indicators/__init__.py
```

### 1.2 Decision

- **Found** — Tell the user: "This indicator already exists as `{ClassName}`. You can use it directly." Then stop.
- **Not found** — Tell the user: "This indicator doesn't exist in oxq. I'll create it for you." Proceed to Phase 2.

---

## Phase 2: ACT — Design and Generate Code

### 2.1 Understand the Indicator Protocol

Every Indicator in oxq must satisfy this Protocol (from `src/oxq/core/types.py`):

```python
@runtime_checkable
class Indicator(Protocol):
    """Indicator contract: pure function, per-symbol vectorized computation."""

    name: str

    def compute(self, mktdata: pd.DataFrame, **params: object) -> pd.Series: ...
```

Key constraints:
- Must have a `name` class attribute (string)
- Must implement `compute(self, mktdata: pd.DataFrame, **params) -> pd.Series`
- `mktdata` is a single symbol's OHLCV DataFrame with columns: `open`, `high`, `low`, `close`, `volume` and a DatetimeIndex
- Return value must be a `pd.Series` with the same index as `mktdata`
- Must be a **pure function** — no side effects, no instance state mutation

### 2.2 Select Example Components

Read 2 existing indicators as references. Choose ones most similar to what the user needs.

**Always read these files:**
- `src/oxq/indicators/rolling_volatility.py` — good general example
- One more similar indicator (e.g., `momentum.py` for momentum-family, `sma.py` for moving-average-family)

Study the pattern: module docstring, imports, class docstring with formula, `name` attribute, optional `formula` attribute, `compute` method with typed params.

### 2.3 Confirm Design With User

Before writing any code, present the design and ask for confirmation:

> **Proposed Indicator: `{ClassName}`**
>
> - **Formula:** {mathematical formula}
> - **Parameters:**
>   - `column` (str, default "close") — input price column
>   - `period` (int, default {N}) — lookback window
>   - {any additional params}
> - **Returns:** pd.Series of {description}
> - **NaN behavior:** First `{period}` values will be NaN (insufficient data)
>
> Does this look right?

**Do not proceed until user confirms.**

### 2.4 Generate Indicator Code

Write the file to `src/oxq/indicators/{snake_name}.py`.

**Naming conventions:**
- Class name: PascalCase (e.g., `GarchVolatility`)
- File name: snake_case (e.g., `garch_volatility.py`)
- `name` attribute: same as class name (e.g., `"GarchVolatility"`)

**Code template:**

```python
"""{ClassName} indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


class {ClassName}:
    """{One-line description}.

    {Formula in plain text or LaTeX}
    """

    name = "{ClassName}"
    formula = r"{LaTeX formula}"

    def compute(
        self,
        mktdata: pd.DataFrame,
        column: str = "close",
        period: int = {default_period},
    ) -> pd.Series:
        """{Docstring describing return value and NaN behavior}."""
        # Implementation here
        ...
```

**Rules:**
- Only import from `numpy`, `pandas`, and Python stdlib — no external dependencies
- Keep it simple — one class per file, one `compute` method
- Match the code style of existing indicators exactly

### 2.5 Generate Unit Test

Write the test file to `tests/indicators/test_{snake_name}.py`.

**Test template:**

```python
"""Tests for {ClassName} indicator."""

import numpy as np
import pandas as pd
import pytest

from oxq.core.types import Indicator
from oxq.indicators.{snake_name} import {ClassName}


def _make_mktdata(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1000},
        index=dates,
    )


def test_{snake_name}_satisfies_indicator_protocol() -> None:
    assert isinstance({ClassName}(), Indicator)


def test_{snake_name}_basic() -> None:
    closes = [100.0, 102.0, 101.0, 104.0, 103.0, 106.0, 105.0, 108.0, 107.0, 110.0]
    mktdata = _make_mktdata(closes)
    result = {ClassName}().compute(mktdata, period={default_period})
    # 1. Length matches input
    assert len(result) == len(closes)
    # 2. Return type is pd.Series
    assert isinstance(result, pd.Series)
    # 3. At least some non-NaN values
    assert result.dropna().shape[0] > 0
    # 4. Hand-calculated expected value for a specific index
    # {ADD HAND-CALCULATED ASSERTION HERE}


def test_{snake_name}_constant_price() -> None:
    """Constant prices should produce zero or near-zero values (where applicable)."""
    mktdata = _make_mktdata([100.0] * 10)
    result = {ClassName}().compute(mktdata, period=5)
    non_nan = result.dropna()
    # {ADJUST ASSERTION BASED ON INDICATOR SEMANTICS}


def test_{snake_name}_has_name() -> None:
    assert {ClassName}().name == "{ClassName}"
```

**Important:** Always include at least one **hand-calculated expected value** assertion. Do not just check shape/type — verify correctness with a concrete number.

---

## Phase 3: FEEDBACK — Validate

Run three validation checks in order. If any fails, enter the retry loop.

### 3.1 Import Check

```bash
cd /Users/daodao/Documents/2-coding-space/git/github.com/open-xquant && uv run python -c "from oxq.indicators.{snake_name} import {ClassName}; print('Import OK')"
```

### 3.2 Protocol Check

```bash
cd /Users/daodao/Documents/2-coding-space/git/github.com/open-xquant && uv run python -c "
from oxq.core.types import Indicator
from oxq.indicators.{snake_name} import {ClassName}
assert isinstance({ClassName}(), Indicator), 'Protocol check failed'
print('Protocol OK')
"
```

### 3.3 Unit Test

```bash
cd /Users/daodao/Documents/2-coding-space/git/github.com/open-xquant && uv run pytest tests/indicators/test_{snake_name}.py -v
```

### 3.4 Failure Retry

If any check fails:

1. Read the error message carefully
2. Identify the root cause (syntax error, wrong formula, type mismatch, etc.)
3. Fix the code in `src/oxq/indicators/{snake_name}.py` (or the test if the test is wrong)
4. Re-run all three checks from 3.1

**Maximum 3 retry attempts.** If still failing after 3 retries:
- Report the error to the user with the full traceback
- Suggest what might be wrong
- Ask the user to help resolve

---

## Phase 4: WRITE — Register

Only enter this phase after Phase 3 passes completely.

### 4.1 Update `__init__.py`

Edit `src/oxq/indicators/__init__.py`:
- Add import line in the import section (alphabetical order by module name)
- Add class name to `__all__` list (alphabetical order)

**Example addition:**
```python
# In the import section:
from oxq.indicators.{snake_name} import {ClassName}

# In __all__:
    "{ClassName}",
```

### 4.2 Update `INDICATOR_TYPES` Registry

Edit `src/oxq/tools/strategy.py`:
- Add import line at the top, in the appropriate import group
- Add entry to `INDICATOR_TYPES` dict (alphabetical order by key)

**Example addition:**
```python
# In imports (find the right alphabetical position among individual indicator imports):
from oxq.indicators.{snake_name} import {ClassName}

# In INDICATOR_TYPES dict:
    "{ClassName}": {ClassName},
```

### 4.3 Verify Registration

```bash
cd /Users/daodao/Documents/2-coding-space/git/github.com/open-xquant && uv run python -c "
from oxq.tools.strategy import INDICATOR_TYPES
assert '{ClassName}' in INDICATOR_TYPES, 'Not in INDICATOR_TYPES'
print('Registration OK: {ClassName} is now available')
"
```

### 4.4 Run Full Test Suite

```bash
cd /Users/daodao/Documents/2-coding-space/git/github.com/open-xquant && uv run pytest tests/ -x -q
```

If any existing test breaks, fix the registration (likely an import error) before proceeding.

### 4.5 Report to User

> **{ClassName} indicator created and registered successfully.**
>
> - Source: `src/oxq/indicators/{snake_name}.py`
> - Test: `tests/indicators/test_{snake_name}.py`
> - Available in `indicator_list()` as `"{ClassName}"`
>
> You can now use it in a strategy:
> ```python
> indicators={"{indicator_name}": {"type": "{ClassName}", "params": {"column": "close", "period": {N}}}}
> ```

---

## Red Lines

- **Never skip user confirmation** in Phase 2.3 — the user must approve the design
- **Never register before validation passes** — Phase 4 requires Phase 3 to be green
- **Never modify existing indicators** — only create new ones
- **Never use external dependencies** beyond numpy, pandas, and stdlib
- **Never exceed 3 retries** — escalate to user after 3 failures
- **Never guess formulas** — if unsure about the mathematical definition, ask the user

## Error Handling

| Error | Action |
|-------|--------|
| Indicator already exists | Tell user, suggest using existing one |
| Import fails | Check file path, class name spelling, syntax errors |
| Protocol check fails | Ensure `name` attribute exists and `compute` signature matches |
| Test fails | Read error, fix code or test, retry |
| Registration import fails | Check import path matches file location |
| Full test suite regression | Undo registration changes, debug |
