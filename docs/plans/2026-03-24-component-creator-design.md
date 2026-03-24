# Component Creator System Design (v2)

## Overview

Expand open-xquant's component-creator from an Indicator-only MVP into a full component creation system supporting 4 strategy logic component types. Primary consumer is autonomous Agents (e.g., Quangent), with open-xquant serving two roles:

- **Constraints** — Protocol definitions, register validation, test requirements
- **Few-shot** — Built-in components as reference implementations

## Scope

4 component types (strategy logic only):

| Type | Protocol | Method |
|------|----------|--------|
| Indicator | `Indicator` | `compute(mktdata) -> Series` |
| Signal | `Signal` | `compute(mktdata) -> Series` |
| PortfolioOptimizer | `PortfolioOptimizer` | `optimize(signals, indicators) -> dict[str, float]` |
| Rule | `Rule` | `evaluate(symbol, row, portfolio, prices) -> RuleResult` |

Out of scope for v1: UniverseProvider, MarketDataProvider (infrastructure components, low creation frequency, external API dependencies).

## Two-Layer Registration Mechanism

### Layer 1: Runtime Dynamic Registration (research phase)

```python
# src/oxq/core/registry.py

# Register API — each does Protocol isinstance check, raises TypeError on failure
oxq.register_indicator(cls) -> None
oxq.register_signal(cls) -> None
oxq.register_portfolio_optimizer(cls) -> None
oxq.register_rule(cls) -> None

# Query API — returns merged view (built-in + entry points + dynamic)
oxq.list_indicators() -> dict[str, type]
oxq.list_signals() -> dict[str, type]
oxq.list_portfolio_optimizers() -> dict[str, type]
oxq.list_rules() -> dict[str, type]
```

Internal: all `register_xxx` share a private `_register(cls, protocol, registry_dict)` implementation.

### Layer 2: Entry Points (post-approval persistence)

```toml
# Consumer's pyproject.toml
[project.entry-points."oxq.indicators"]
GarchVolatility = "myproject.indicators.garch:GarchVolatility"

[project.entry-points."oxq.signals"]
RegimeSwitch = "myproject.signals.regime:RegimeSwitch"

[project.entry-points."oxq.portfolio_optimizers"]
BlackLitterman = "myproject.optimizers.bl:BlackLitterman"

[project.entry-points."oxq.rules"]
SectorLimit = "myproject.rules.sector:SectorLimit"
```

Entry points groups: `oxq.indicators`, `oxq.signals`, `oxq.portfolio_optimizers`, `oxq.rules`.

### Loading & Conflict Resolution

- Entry points loaded at `core/registry.py` module initialization via `importlib.metadata.entry_points()`
- Single entry point load failure: log warning, continue (no blocking)
- `register_xxx()` callable at any time, appends to same dict
- `list_xxx()` always returns merged view
- Name conflict: last-write-wins, no error

### Registry Migration

Move registry dicts from `tools/strategy.py` to `core/registry.py`. `tools/strategy.py` becomes a consumer that imports from `core/registry.py`.

## Two-Layer Skill Structure

```
component-creator (navigation layer)
  |-- READ: query registry -> exists? return usage -> not found? route to sub-skill
  |-- /create-indicator
  |-- /create-signal
  |-- /create-portfolio-optimizer
  +-- /create-rule
```

### Navigation Layer (component-creator.md)

Responsibilities:

1. **Receive request** — Agent describes what component it needs
2. **Determine type** — Indicator / Signal / PortfolioOptimizer / Rule
3. **Query registry** — call `list_xxx()` to check existence
4. **Route:**
   - Exists -> return component name and usage, done
   - Not found -> route to corresponding sub-skill

Does NOT: write code, run tests, register components, or do anything beyond intent recognition and routing.

### Sub-Skill Common Flow (4 phases)

**Phase 1: DESIGN — Self-check**
- Read 2 existing same-type components as few-shot reference
- Output design intent: name, formula/logic, parameters, return value, boundary behavior
- No external confirmation required — continue autonomously
- Design intent preserved as audit record

**Phase 2: CODE — Generate**
- Component code: written to consumer project directory (not `src/oxq/`)
- Test code: written to consumer project test directory
- Dependencies: prefer numpy, pandas, stdlib, and oxq types. Third-party libs allowed as optional deps (try/except import + `pyproject.toml` optional-dependencies group)
- Code style: follow oxq built-in component conventions

**Phase 3: VALIDATE — Three-layer verification**
1. Import check
2. Protocol `isinstance` check
3. Unit test (with hand-calculated / scenario assertions)
- Max 3 retries on failure, then escalate

**Phase 4: REGISTER — Dynamic registration**
- Call `oxq.register_xxx(cls)`
- Verify registration (query back from registry)
- Output result report

Key: skill specifies flow and constraints, NOT where code lives. Use placeholders (`{target_dir}`) for output paths — directory structure is the consumer's concern.

## Per-Type Differences

### Indicator
- Test: hand-calculated value assertions, constant-price boundary test
- Few-shot: `rolling_volatility.py` + one similar indicator
- Most mature, closest to existing MVP

### Signal
- Test: output value domain check (bool or finite categorical), logic correctness via known trigger/no-trigger scenario
- Few-shot: `crossover.py` + `threshold.py`
- Note: same Protocol signature as Indicator — design intent must clarify output semantics

### PortfolioOptimizer
- Test: **mandatory** weights sum = 1.0 assertion, multi-symbol input construction, empty-signal boundary
- Few-shot: `EqualWeightOptimizer` + `TopNRankingOptimizer`

### Rule
- Test: scenario tests with constructed Portfolio state (e.g., "when holdings exceed limit, should return hold=True")
- Few-shot: `MaxHoldingsRule` (pre-trade) + `StopLossRule` (post-trade)
- Most complex: design intent must specify pre-trade vs post-trade

## Deliverables

| # | File | Description |
|---|------|-------------|
| 1 | `src/oxq/core/registry.py` | 4 registry dicts + register/list API + entry points loading |
| 2 | `skills/component-creator.md` | Rewrite as navigation layer (READ + route) |
| 3 | `skills/create-indicator.md` | Adapted from MVP to new flow |
| 4 | `skills/create-signal.md` | New |
| 5 | `skills/create-portfolio-optimizer.md` | New |
| 6 | `skills/create-rule.md` | New |
| 7 | `tools/strategy.py` | Import registry from `core/registry.py` |

## History

- v1 (MVP): Indicator-only, direct write to `src/oxq/`, user confirmation required
- v2 (this design): 4 component types, plugin registration mechanism, Agent-first autonomous flow
