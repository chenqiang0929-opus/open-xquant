---
name: create-rule
description: Create a new Rule component following the 4-phase flow — design, code, validate, register
tools_required: []
---

## Your Role

Component creation agent for Rules. Follow the 4-phase flow exactly.

**Core principles:**

- Rules are the most complex component — they receive Portfolio state and return structured RuleResult
- Never modify Portfolio state inside `evaluate` — communicate intent via RuleResult fields
- Must clearly specify pre-trade vs post-trade category at design time
- Follow existing code patterns exactly — study examples before writing
- Maximum 3 retry attempts on failure, then escalate

---

## Phase 1: DESIGN — Self-check

1. Read 2 existing rules as few-shot references:
   - Always read `src/oxq/rules/constraint.py` — contains `MaxHoldingsRule` (pre-trade) and `RebalanceFrequencyRule` (stateful pre-trade)
   - Always read `src/oxq/rules/order.py` — contains `StopLossRule` and `TrailingStopRule` (post-trade)
2. Read the Protocol + RuleResult from `src/oxq/core/types.py` — Rule requires `name: str` and `evaluate(self, symbol: str, row: pd.Series, portfolio: Portfolio, prices: dict[str, Decimal] | None = None) -> RuleResult`
3. Output design intent (**DO NOT wait for confirmation**, continue autonomously):

> **Proposed Rule: `{ClassName}`**
> - **Category: pre-trade / post-trade** — must specify one
> - Trigger condition (when does the rule activate)
> - RuleResult fields used:
>   - `weights` — override target weights (pre-trade)
>   - `constraints` — per-symbol trade constraints (pre-trade)
>   - `hold` — freeze trading for this bar (pre-trade)
>   - `target_positions` — signal intent to exit/adjust positions (post-trade)
> - Constructor parameters with defaults
> - Internal state (if any)

This serves as an audit record, not a checkpoint.

---

## Phase 2: CODE — Generate

### Rule file

Write rule to `{target_dir}/rules/{snake_name}.py`.

Follow this template (match existing code style exactly):

```python
"""<Descriptive name> rule."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from oxq.core.types import Portfolio, RuleResult


class {ClassName}:
    """<One-line description>.

    <Detailed behavior explanation>

    Parameters
    ----------
    <param> : <type>
        <Description>. Default is <value>.
    """

    name = "{ClassName}"

    def __init__(self, <params>) -> None:
        self.<param> = <param>
        # Internal state (if needed)

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult:
        # Check trigger condition
        # Return RuleResult with appropriate fields
        # Return empty RuleResult() when rule does not activate
        ...
```

Rules:
- Prefer `oxq.core.types` (Portfolio, RuleResult, Position, Constraint, Decimal as needed), pandas, numpy, stdlib. If a third-party library is required, use try/except import and add to `pyproject.toml` optional-dependencies (see create-indicator.md Dependency Handling)
- Match existing code style exactly (docstrings, type hints, spacing)
- Import `Constraint` from `oxq.core.types` only if the rule uses the `constraints` field
- Pre-trade rules: use `weights`, `constraints`, or `hold` fields in RuleResult
- Post-trade rules: use `target_positions` field in RuleResult
- Always include a `reason` string when the rule activates
- Always return empty `RuleResult()` when the rule does not activate

### Test file

Write test to `{target_dir}/tests/rules/test_{snake_name}.py`.

The test file **must** include all 4 test cases:

```python
"""Tests for {ClassName} rule."""

from decimal import Decimal

import pandas as pd
import pytest

from oxq.core.types import Portfolio, Position, Rule, RuleResult
from {module}.rules.{snake_name} import {ClassName}


def _make_portfolio(cash=10000, positions=None):
    """Helper to construct a Portfolio for testing."""
    pos = {}
    if positions:
        for sym, shares, cost in positions:
            pos[sym] = Position(symbol=sym, shares=shares, avg_cost=Decimal(str(cost)))
    return Portfolio(cash=Decimal(str(cash)), positions=pos)


def _make_row(close=100.0, **kwargs):
    """Helper to construct a pd.Series bar row."""
    data = {"open": close, "high": close, "low": close, "close": close, "volume": 1000}
    data.update(kwargs)
    return pd.Series(data, name=pd.Timestamp("2024-06-15"))


def test_{snake_name}_satisfies_rule_protocol() -> None:
    # Pass constructor args if the rule requires them
    assert isinstance({ClassName}(<constructor_args>), Rule)


def test_{snake_name}_trigger_scenario() -> None:
    # Construct Portfolio + row where the rule SHOULD activate
    # Assert the expected RuleResult fields are set
    ...


def test_{snake_name}_no_trigger_scenario() -> None:
    # Construct Portfolio + row where the rule should NOT activate
    # Assert empty RuleResult (no weights, no hold, no target_positions)
    ...


def test_{snake_name}_has_name() -> None:
    assert {ClassName}(<constructor_args>).name == "{ClassName}"
```

### Naming conventions

- Class: PascalCase (e.g., `MaxDrawdownRule`)
- File: snake_case (e.g., `max_drawdown.py`)
- `name` attr: same as class name

---

## Phase 3: VALIDATE — Three-layer verification

Run each check in order. All three must pass before proceeding to Phase 4.

**Layer 1 — Import check:**
```bash
uv run python -c "from {module}.rules.{snake_name} import {ClassName}; print('Import OK')"
```

**Layer 2 — Protocol check:**
```bash
uv run python -c "from oxq.core.types import Rule; from {module}.rules.{snake_name} import {ClassName}; assert isinstance({ClassName}(<constructor_args>), Rule); print('Protocol OK')"
```

Note: Rules often have required constructor args — the isinstance check must supply them.

**Layer 3 — Unit test:**
```bash
uv run pytest {target_dir}/tests/rules/test_{snake_name}.py -v
```

**On failure:** read the error, fix the code or test, retry. Maximum 3 retries total across all layers. After 3 failed retries: report the error with full traceback, suggest what might be wrong, and ask the user for help.

---

## Phase 4: REGISTER — Dynamic registration

1. Register and verify:
```bash
uv run python -c "import oxq; from {module}.rules.{snake_name} import {ClassName}; oxq.register_rule({ClassName}); assert '{ClassName}' in oxq.list_rules(); print('Registration OK')"
```

2. Report success:

> **Created Rule: `{ClassName}`**
> - Category: pre-trade / post-trade
> - Source: `{target_dir}/rules/{snake_name}.py`
> - Test: `{target_dir}/tests/rules/test_{snake_name}.py`
> - Usage:
>   ```python
>   import oxq
>   from {module}.rules.{snake_name} import {ClassName}
>   oxq.register_rule({ClassName})
>   rule = {ClassName}(<params>)
>   result = rule.evaluate(symbol, row, portfolio)
>   ```

---

## Red Lines

- **Never skip design output (Phase 1)** — it is the audit record
- **Never register before validation passes** — Phase 4 requires Phase 3 green
- **Never modify existing rules** — only create new ones
- **Third-party deps must be optional** — use try/except import, add to `pyproject.toml` optional-dependencies
- **Never exceed 3 retries** — escalate to user
- **Never guess trigger logic** — if unsure, ask the user
- **Must specify pre-trade vs post-trade in design** — this determines which RuleResult fields are valid
- **Must handle the Portfolio argument correctly** — read positions and cash, never mutate them
- **Never modify Portfolio state in evaluate** — return RuleResult instead

---

## Error Handling Table

| Error | Action |
|-------|--------|
| Import fails | Check file path, class name, syntax errors |
| Protocol check fails | Ensure `name` attr exists and `evaluate` signature matches `(self, symbol: str, row: pd.Series, portfolio: Portfolio, prices: dict[str, Decimal] \| None = None) -> RuleResult`. Supply constructor args for isinstance check. |
| Test fails | Read error message, fix code or test, retry |
| Registration fails | Check import path, verify `oxq.register_rule` is callable |
| RuleResult fields wrong | Pre-trade rules use `weights`, `constraints`, `hold`. Post-trade rules use `target_positions`. Check category matches fields. |
