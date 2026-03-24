---
name: create-portfolio-optimizer
description: Create a new PortfolioOptimizer component following the 4-phase flow — design, code, validate, register
tools_required: []
---

## Your Role

Component creation agent for PortfolioOptimizers. Follow the 4-phase flow exactly.

**Core principles:**

- Weights MUST always sum to 1.0 (including CASH if needed) — this is a hard invariant
- Empty or invalid input MUST return `{"CASH": 1.0}` — never return an empty dict
- Follow existing code patterns exactly — study examples before writing
- Maximum 3 retry attempts on failure, then escalate

---

## Phase 1: DESIGN — Self-check

1. Read 2 existing optimizers as few-shot references:
   - Always read `EqualWeightOptimizer` and `TopNRankingOptimizer` from `src/oxq/portfolio/optimizers.py`
   - If the new optimizer is risk-based, also read `RiskParityOptimizer` from the same file
2. Read the Protocol from `src/oxq/core/types.py` — PortfolioOptimizer requires `name: str` and `optimize(self, signals: dict[str, pd.DataFrame], indicators: dict[str, pd.DataFrame]) -> dict[str, float]`
3. Output design intent (**DO NOT wait for confirmation**, continue autonomously):

> **Proposed Optimizer: `{ClassName}`**
> - Allocation logic (how weights are calculated)
> - Parameters (constructor args with defaults)
> - What inputs it uses (signals, indicators, or both)
> - Fallback behavior (what to return when no valid inputs)

This serves as an audit record, not a checkpoint.

---

## Phase 2: CODE — Generate

### Optimizer file

Write optimizer to `{target_dir}/portfolio/{snake_name}.py`.

Follow this template (match existing code style exactly):

```python
"""<Descriptive name> portfolio optimizer."""

from __future__ import annotations

import pandas as pd


class {ClassName}:
    """<One-line description>.

    <Allocation logic explanation>
    """

    name: str = "{ClassName}"

    def __init__(self, <params>) -> None:
        self.<param> = <param>

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """Return target weights summing to 1.0."""
        if not signals:
            return {"CASH": 1.0}

        # Compute raw weights
        ...

        # Normalize so weights sum to 1.0 (use CASH for remainder)
        ...

        return weights
```

Rules:
- Only import numpy, pandas, stdlib
- Match existing code style exactly (docstrings, type hints, spacing)
- Every code path must return weights that sum to 1.0
- Every code path with no valid data must return `{"CASH": 1.0}`

### Test file

Write test to `{target_dir}/tests/portfolio/test_{snake_name}.py`.

The test file **must** include all 6 test cases:

```python
"""Tests for {ClassName} portfolio optimizer."""

import pandas as pd
import pytest

from oxq.core.types import PortfolioOptimizer
from {module}.portfolio.{snake_name} import {ClassName}


def _make_signals(symbols: list[str]) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=5)
    return {
        sym: pd.DataFrame(
            {"signal": [1.0] * 5},
            index=dates,
        )
        for sym in symbols
    }


def _make_indicators(data: dict[str, dict[str, list[float]]]) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=5)
    return {
        sym: pd.DataFrame(cols, index=dates)
        for sym, cols in data.items()
    }


def test_{snake_name}_satisfies_protocol() -> None:
    # Note: if constructor has required args, pass them here
    assert isinstance({ClassName}(<required_args>), PortfolioOptimizer)


def test_{snake_name}_weights_sum_to_one() -> None:
    optimizer = {ClassName}(<args>)
    signals = _make_signals(["AAPL", "GOOGL", "MSFT"])
    indicators = _make_indicators(...)
    weights = optimizer.optimize(signals, indicators)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_{snake_name}_multi_symbol() -> None:
    # At least 3 symbols input
    optimizer = {ClassName}(<args>)
    signals = _make_signals(["AAPL", "GOOGL", "MSFT"])
    indicators = _make_indicators(...)
    weights = optimizer.optimize(signals, indicators)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert len(weights) >= 1


def test_{snake_name}_empty_signals() -> None:
    optimizer = {ClassName}(<args>)
    weights = optimizer.optimize({}, {})
    assert weights == {"CASH": 1.0}


def test_{snake_name}_hand_calculated() -> None:
    # Hand-calculated weight for a specific scenario
    # NEVER copy values from code output — compute by hand
    ...
    assert weights["AAPL"] == pytest.approx(<hand_calculated_value>)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_{snake_name}_has_name() -> None:
    assert {ClassName}(<required_args>).name == "{ClassName}"
```

### Naming conventions

- Class: PascalCase (e.g., `MomentumWeightOptimizer`)
- File: snake_case (e.g., `momentum_weight_optimizer.py`)
- `name` attr: same as class name

---

## Phase 3: VALIDATE — Three-layer verification

Run each check in order. All three must pass before proceeding to Phase 4.

**Layer 1 — Import check:**
```bash
uv run python -c "from {module}.portfolio.{snake_name} import {ClassName}; print('Import OK')"
```

**Layer 2 — Protocol check:**
```bash
uv run python -c "from oxq.core.types import PortfolioOptimizer; from {module}.portfolio.{snake_name} import {ClassName}; assert isinstance({ClassName}(<required_args>), PortfolioOptimizer); print('Protocol OK')"
```

**Layer 3 — Unit test:**
```bash
uv run pytest {target_dir}/tests/portfolio/test_{snake_name}.py -v
```

**On failure:** read the error, fix the code or test, retry. Maximum 3 retries total across all layers. After 3 failed retries: report the error with full traceback, suggest what might be wrong, and ask the user for help.

---

## Phase 4: REGISTER — Dynamic registration

1. Register and verify:
```bash
uv run python -c "import oxq; from {module}.portfolio.{snake_name} import {ClassName}; oxq.register_portfolio_optimizer({ClassName}); assert '{ClassName}' in oxq.list_portfolio_optimizers(); print('Registration OK')"
```

2. Report success:

> **Created PortfolioOptimizer: `{ClassName}`**
> - Source: `{target_dir}/portfolio/{snake_name}.py`
> - Test: `{target_dir}/tests/portfolio/test_{snake_name}.py`
> - Usage:
>   ```python
>   import oxq
>   from {module}.portfolio.{snake_name} import {ClassName}
>   oxq.register_portfolio_optimizer({ClassName})
>   weights = {ClassName}(<args>).optimize(signals, indicators)
>   ```

---

## Red Lines

- **Never skip design output (Phase 1)** — it is the audit record
- **Never register before validation passes** — Phase 4 requires Phase 3 green
- **Never modify existing optimizers** — only create new ones
- **Never use external dependencies** beyond numpy, pandas, stdlib
- **Never exceed 3 retries** — escalate to user
- **Never guess allocation logic** — if unsure, ask the user
- **Weights MUST sum to 1.0** — this is a hard invariant; every code path must guarantee it
- **Empty/invalid input MUST return `{"CASH": 1.0}`** — never return an empty dict

---

## Error Handling Table

| Error | Action |
|-------|--------|
| Import fails | Check file path, class name, syntax errors |
| Protocol check fails | Ensure `name` attr exists and `optimize` signature matches `(self, signals: dict[str, pd.DataFrame], indicators: dict[str, pd.DataFrame]) -> dict[str, float]` |
| Test fails | Read error message, fix code or test, retry |
| Registration fails | Check import path, verify `oxq.register_portfolio_optimizer` is callable |
| Weights don't sum to 1.0 | Fix allocation logic — ensure normalization step or CASH remainder covers the gap |
