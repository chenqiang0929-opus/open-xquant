"""Strategy tools — create, compose, and inspect strategies."""

from __future__ import annotations

import inspect
from typing import Any

from oxq.core.strategy import Strategy
from oxq.indicators.builtin import (
    ADX,
    AROON,
    ATR,
    CCI,
    DEMA,
    EMA,
    MFI,
    OBV,
    PPO,
    ROC,
    RSI,
    TEMA,
    VWAP,
    WMA,
    BollingerLower,
    BollingerUpper,
    MACDHistogram,
    MACDLine,
    MACDSignal,
    StochK,
)
from oxq.indicators.log_return import LogReturn
from oxq.indicators.momentum import Momentum
from oxq.indicators.nday_return import NdayReturn
from oxq.indicators.ratio import Ratio
from oxq.indicators.rolling_mdd import RollingMDD
from oxq.indicators.rolling_volatility import RollingVolatility
from oxq.indicators.sma import SMA
from oxq.rules.entry import EntryRule, FullPositionEntryRule, TargetValueEntryRule
from oxq.rules.exit import ExitRule
from oxq.rules.rebalance import RebalanceRule
from oxq.signals.crossover import Crossover
from oxq.signals.top_n_ranking import TopNRanking
from oxq.tools import session
from oxq.tools.registry import registry
from oxq.universe.static import StaticUniverse

# ---------------------------------------------------------------------------
# Type registries (string → class mapping)
# ---------------------------------------------------------------------------

INDICATOR_TYPES: dict[str, type] = {
    "ADX": ADX,
    "AROON": AROON,
    "ATR": ATR,
    "BollingerLower": BollingerLower,
    "BollingerUpper": BollingerUpper,
    "CCI": CCI,
    "DEMA": DEMA,
    "EMA": EMA,
    "LogReturn": LogReturn,
    "MACDHistogram": MACDHistogram,
    "MACDLine": MACDLine,
    "MACDSignal": MACDSignal,
    "MFI": MFI,
    "Momentum": Momentum,
    "NdayReturn": NdayReturn,
    "OBV": OBV,
    "PPO": PPO,
    "ROC": ROC,
    "RSI": RSI,
    "Ratio": Ratio,
    "RollingMDD": RollingMDD,
    "RollingVolatility": RollingVolatility,
    "SMA": SMA,
    "StochK": StochK,
    "TEMA": TEMA,
    "VWAP": VWAP,
    "WMA": WMA,
}
SIGNAL_TYPES: dict[str, type] = {"Crossover": Crossover, "TopNRanking": TopNRanking}
RULE_TYPES: dict[str, type] = {
    "EntryRule": EntryRule,
    "TargetValueEntryRule": TargetValueEntryRule,
    "FullPositionEntryRule": FullPositionEntryRule,
    "ExitRule": ExitRule,
    "RebalanceRule": RebalanceRule,
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
    elif type == "RebalanceRule":
        strat.rebalance_rules.append(rule)

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
        "rebalance_rules": [
            {"type": r.__class__.__name__, "name": r.name}
            for r in strat.rebalance_rules
        ],
    }


# ---------------------------------------------------------------------------
# Indicator metadata tools
# ---------------------------------------------------------------------------


@registry.tool(
    name="indicator_describe",
    description="Describe an indicator: show its LaTeX formula, parameters, category, and dependencies",
)
def indicator_describe(type: str) -> dict[str, Any]:
    """Return indicator metadata including LaTeX formula."""
    cls = INDICATOR_TYPES.get(type)
    if cls is None:
        return {"error": f"Unknown indicator '{type}'. Available: {sorted(INDICATOR_TYPES)}"}

    sig = inspect.signature(cls().compute)
    params = {
        k: str(v.default) if v.default is not v.empty else "(required)"
        for k, v in sig.parameters.items()
        if k not in ("self", "mktdata")
    }

    return {
        "name": getattr(cls, "name", type),
        "formula": getattr(cls, "formula", ""),
        "description": (cls.__doc__ or "").strip(),
        "params": params,
        "depends_on": list(getattr(cls, "depends_on", ())),
    }


@registry.tool(
    name="indicator_list",
    description="List all available indicator types with their formulas",
)
def indicator_list() -> dict[str, Any]:
    """Return all indicator types with names, formulas, and descriptions."""
    return {
        "indicators": [
            {
                "name": name,
                "formula": getattr(cls, "formula", ""),
                "description": (cls.__doc__ or "").split("\n")[0].strip(),
            }
            for name, cls in sorted(INDICATOR_TYPES.items())
        ],
    }
