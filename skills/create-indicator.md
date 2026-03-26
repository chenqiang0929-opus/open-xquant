---
name: create-indicator
description: Create a new Indicator component following the 4-phase flow — design, code, validate, register
tools_required: []
---

## Your Role

Component creation agent for Indicators. Follow the 4-phase flow exactly.

**Core principles:**

- Generated code must be a pure function — no side effects, no state
- Follow existing code patterns exactly — study examples before writing
- Maximum 3 retry attempts on failure, then escalate

---

## Phase 1: DESIGN — Self-check

1. Read 2 existing indicators as few-shot references:
   - Always read `src/oxq/indicators/rolling_volatility.py`
   - Choose one more similar indicator (e.g., `momentum.py` for momentum-family, `sma.py` for moving-average-family)
2. Read the Protocol from `src/oxq/core/types.py` — Indicator requires `name: str` and `compute(self, mktdata: pd.DataFrame, **params) -> pd.Series`
3. Output design intent (**DO NOT wait for confirmation**, continue autonomously):

> **Proposed Indicator: `{ClassName}`**
> - Formula (mathematical)
> - Parameters with defaults
> - Return value description
> - NaN behavior (first N values)

This serves as an audit record, not a checkpoint.

---

## Phase 2: CODE — Generate

### Indicator file

Write indicator to `{target_dir}/indicators/{snake_name}.py`.

Follow this template (match existing code style exactly):

```python
"""<Descriptive name> indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


class {ClassName}:
    """<One-line description>.

    <LaTeX or text formula>
    """

    name = "{ClassName}"
    formula = r"<LaTeX formula>"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", <other params>,
    ) -> pd.Series:
        """<Return description>."""
        # Pure computation — no side effects
        ...
```

Rules:
- Prefer numpy, pandas, stdlib. If a third-party library is required, see **Dependency Handling** below
- Match existing code style exactly (docstrings, type hints, spacing)

### Composite Indicator Pattern

When the user needs an indicator that combines multiple base indicators (e.g., risk-adjusted momentum = Momentum / RollingVolatility), create a **self-contained** indicator that internally computes its dependencies:

```python
"""Risk-adjusted momentum indicator."""

from __future__ import annotations

import pandas as pd

from oxq.indicators.momentum import Momentum
from oxq.indicators.rolling_volatility import RollingVolatility


class RiskAdjustedMomentum:
    """Momentum normalized by rolling volatility.

    RiskAdjMom = Momentum_N / RollingVol_N
    """

    name = "RiskAdjustedMomentum"
    formula = r"RiskAdjMom = \frac{Mom_N}{\sigma_N}"

    def compute(
        self,
        mktdata: pd.DataFrame,
        column: str = "close",
        momentum_period: int = 20,
        vol_period: int = 20,
    ) -> pd.Series:
        """Return momentum / volatility ratio."""
        mom = Momentum().compute(mktdata, column=column, period=momentum_period)
        vol = RollingVolatility().compute(mktdata, column=column, period=vol_period)
        return mom / vol
```

**Key rules for composite indicators:**
- Import and instantiate base indicators **inside the module** — do NOT assume they exist in mktdata columns
- The `compute()` method must be self-contained: given raw mktdata (OHLCV), return the final result
- Do NOT use `Ratio` or other column-referencing indicators — those require the Engine's dependency chain
- Expose each base indicator's parameters separately (e.g., `momentum_period`, `vol_period`)

### Dependency Handling

If the indicator requires a third-party library (e.g., `arch` for GARCH, `scipy` for special functions):

1. **Check `pyproject.toml` `[project.optional-dependencies]`** — does a group for this library already exist?
2. **If not, add a new optional group:**
   ```toml
   [project.optional-dependencies]
   arch = ["arch>=7.0"]
   ```
3. **Use try/except import in the component code:**
   ```python
   try:
       from arch import arch_model
   except ImportError as e:
       raise ImportError(
           "arch is required for GarchVolatility. Install with: pip install open-xquant[arch]"
       ) from e
   ```
4. **Install the dep before validation:** `uv pip install -e ".[arch]"` (or the appropriate group name)
5. **Note the dependency in the design intent output** (Phase 1)

### Test file

Write test to `{target_dir}/tests/indicators/test_{snake_name}.py`.

The test file **must** include all 5 test cases:

```python
"""Tests for {ClassName} indicator."""

import numpy as np
import pandas as pd
import pytest

from oxq.core.types import Indicator
from {module}.indicators.{snake_name} import {ClassName}


def _make_mktdata(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1000},
        index=dates,
    )


def test_{snake_name}_satisfies_indicator_protocol() -> None:
    assert isinstance({ClassName}(), Indicator)


def test_{snake_name}_basic() -> None:
    # Length matches input, return type is pd.Series, some non-NaN values
    ...


def test_{snake_name}_hand_calculated() -> None:
    # At least one hand-calculated expected value assertion
    # NEVER copy values from code output — compute by hand
    ...


def test_{snake_name}_constant_price() -> None:
    # Constant prices edge case
    ...


def test_{snake_name}_has_name() -> None:
    assert {ClassName}().name == "{ClassName}"
```

### Naming conventions

- Class: PascalCase (e.g., `GarchVolatility`)
- File: snake_case (e.g., `garch_volatility.py`)
- `name` attr: same as class name

---

## Phase 3: VALIDATE — Three-layer verification

Run each check in order. All three must pass before proceeding to Phase 4.

**Layer 1 — Import check:**
```bash
uv run python -c "from {module}.indicators.{snake_name} import {ClassName}; print('Import OK')"
```

**Layer 2 — Protocol check:**
```bash
uv run python -c "from oxq.core.types import Indicator; from {module}.indicators.{snake_name} import {ClassName}; assert isinstance({ClassName}(), Indicator); print('Protocol OK')"
```

**Layer 3 — Unit test:**
```bash
uv run pytest {target_dir}/tests/indicators/test_{snake_name}.py -v
```

**On failure:** read the error, fix the code or test, retry. Maximum 3 retries total across all layers. After 3 failed retries: report the error with full traceback, suggest what might be wrong, and ask the user for help.

---

## Phase 4: REGISTER — Dynamic registration

1. Register and verify:
```bash
uv run python -c "import oxq; from {module}.indicators.{snake_name} import {ClassName}; oxq.register_indicator({ClassName}); assert '{ClassName}' in oxq.list_indicators(); print('Registration OK')"
```

2. Report success:

> **Created Indicator: `{ClassName}`**
> - Source: `{target_dir}/indicators/{snake_name}.py`
> - Test: `{target_dir}/tests/indicators/test_{snake_name}.py`
> - Usage:
>   ```python
>   import oxq
>   from {module}.indicators.{snake_name} import {ClassName}
>   oxq.register_indicator({ClassName})
>   result = {ClassName}().compute(mktdata)
>   ```

---

## Red Lines

- **Never skip design output (Phase 1)** — it is the audit record
- **Never register before validation passes** — Phase 4 requires Phase 3 green
- **Never modify existing indicators** — only create new ones
- **Third-party deps must be optional** — use try/except import, add to `pyproject.toml` optional-dependencies
- **Never exceed 3 retries** — escalate to user
- **Never guess formulas** — if unsure, ask the user

---

## Error Handling Table

| Error | Action |
|-------|--------|
| Import fails | Check file path, class name, syntax errors |
| Protocol check fails | Ensure `name` attr exists and `compute` signature matches `(self, mktdata: pd.DataFrame, **params) -> pd.Series` |
| Test fails | Read error message, fix code or test, retry |
| Registration fails | Check import path, verify `oxq.register_indicator` is callable |
