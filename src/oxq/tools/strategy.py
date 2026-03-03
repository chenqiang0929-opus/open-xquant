"""Strategy tools — create, compose, and inspect strategies."""

from __future__ import annotations

from typing import Any

from oxq.core.strategy import Strategy
from oxq.indicators.sma import SMA
from oxq.rules.entry import EntryRule, FullPositionEntryRule, TargetValueEntryRule
from oxq.rules.exit import ExitRule
from oxq.signals.crossover import Crossover
from oxq.tools import session
from oxq.tools.registry import registry
from oxq.universe.static import StaticUniverse

# ---------------------------------------------------------------------------
# Type registries (string → class mapping)
# ---------------------------------------------------------------------------

INDICATOR_TYPES: dict[str, type] = {"SMA": SMA}
SIGNAL_TYPES: dict[str, type] = {"Crossover": Crossover}
RULE_TYPES: dict[str, type] = {
    "EntryRule": EntryRule,
    "TargetValueEntryRule": TargetValueEntryRule,
    "FullPositionEntryRule": FullPositionEntryRule,
    "ExitRule": ExitRule,
}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@registry.tool(
    name="strategy_create",
    description="Create a new strategy with hypothesis and objectives",
)
def strategy_create(
    name: str,
    hypothesis: str,
    objectives: dict[str, dict[str, float]] | None = None,
    benchmarks: list[str] | None = None,
) -> dict[str, Any]:
    """Create a new empty strategy and store it in the session."""
    if not hypothesis:
        return {"error": "hypothesis must not be empty"}
    if not objectives:
        return {"error": "objectives must not be empty"}

    strategy = Strategy(
        name=name,
        hypothesis=hypothesis,
        objectives=objectives,
        benchmarks=benchmarks or [],
        universe=StaticUniverse(()),
        indicators={},
        signals={},
        entry_rules=[],
        exit_rules=[],
    )
    session._strategies[name] = strategy
    session._save()
    return {
        "name": name,
        "hypothesis": hypothesis,
        "objectives": objectives,
        "benchmarks": benchmarks or [],
    }


@registry.tool(
    name="strategy_add_indicator",
    description="Add an indicator (e.g. SMA) to a strategy",
)
def strategy_add_indicator(
    strategy: str,
    name: str,
    type: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add an indicator to an existing strategy."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    cls = INDICATOR_TYPES.get(type)
    if cls is None:
        return {"error": f"Unknown indicator type '{type}'. Available: {sorted(INDICATOR_TYPES)}"}

    strat.indicators[name] = (cls(), params or {})
    session._save()
    return {
        "strategy": strategy,
        "indicator": name,
        "type": type,
        "params": params or {},
    }


@registry.tool(
    name="strategy_add_signal",
    description="Add a signal (e.g. Crossover) to a strategy",
)
def strategy_add_signal(
    strategy: str,
    name: str,
    type: str,
    inputs: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add a signal to an existing strategy."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    cls = SIGNAL_TYPES.get(type)
    if cls is None:
        return {"error": f"Unknown signal type '{type}'. Available: {sorted(SIGNAL_TYPES)}"}

    merged = {**(params or {}), **(inputs or {})}
    strat.signals[name] = (cls(), merged)
    session._save()
    return {
        "strategy": strategy,
        "signal": name,
        "type": type,
        "inputs": inputs or {},
    }


@registry.tool(
    name="strategy_add_rule",
    description="Add an entry or exit rule to a strategy",
)
def strategy_add_rule(
    strategy: str,
    name: str,
    type: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add a rule to an existing strategy."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    cls = RULE_TYPES.get(type)
    if cls is None:
        return {"error": f"Unknown rule type '{type}'. Available: {sorted(RULE_TYPES)}"}

    rule_params = params or {}
    try:
        rule = cls(**rule_params)
    except TypeError as e:
        return {"error": f"Invalid params for {type}: {e}"}

    if type in ("EntryRule", "TargetValueEntryRule", "FullPositionEntryRule"):
        strat.entry_rules.append(rule)
    elif type == "ExitRule":
        strat.exit_rules.append(rule)

    session._save()
    return {
        "strategy": strategy,
        "rule": name,
        "type": type,
        "params": rule_params,
    }


@registry.tool(
    name="strategy_inspect",
    description="Inspect a strategy definition (indicators, signals, rules)",
)
def strategy_inspect(strategy: str) -> dict[str, Any]:
    """Return the full strategy definition as a dict."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    return {
        "name": strat.name,
        "hypothesis": strat.hypothesis,
        "objectives": strat.objectives,
        "benchmarks": strat.benchmarks,
        "universe": list(strat.universe.symbols) if hasattr(strat.universe, "symbols") else [],
        "indicators": {
            k: {"type": v[0].__class__.__name__, "params": v[1]}
            for k, v in strat.indicators.items()
        },
        "signals": {
            k: {"type": v[0].__class__.__name__, "params": v[1]}
            for k, v in strat.signals.items()
        },
        "entry_rules": [
            {"type": r.__class__.__name__, "name": r.name}
            for r in strat.entry_rules
        ],
        "exit_rules": [
            {"type": r.__class__.__name__, "name": r.name}
            for r in strat.exit_rules
        ],
    }
