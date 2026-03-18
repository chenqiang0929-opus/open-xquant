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
from oxq.portfolio.optimizers import EqualWeightOptimizer
from oxq.rules.exit import ExitRule
from oxq.rules.order import StopLossRule, TakeProfitRule, TrailingStopRule
from oxq.rules.risk import DailyLossLimitRisk, MaxDrawdownRisk
from oxq.signals.comparison import Comparison
from oxq.signals.composite import Composite
from oxq.signals.crossover import Crossover
from oxq.signals.equal_weight import EqualWeight
from oxq.signals.formula import Formula
from oxq.signals.peak import Peak
from oxq.signals.risk_parity import RiskParity
from oxq.signals.threshold import Threshold
from oxq.signals.timestamp import Timestamp
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
SIGNAL_TYPES: dict[str, type] = {
    "Comparison": Comparison,
    "Composite": Composite,
    "Crossover": Crossover,
    "EqualWeight": EqualWeight,
    "Formula": Formula,
    "Peak": Peak,
    "RiskParity": RiskParity,
    "Threshold": Threshold,
    "Timestamp": Timestamp,
    "TopNRanking": TopNRanking,
}
RULE_TYPES: dict[str, type] = {
    "ExitRule": ExitRule,
    "StopLossRule": StopLossRule,
    "TakeProfitRule": TakeProfitRule,
    "TrailingStopRule": TrailingStopRule,
    "MaxDrawdownRisk": MaxDrawdownRisk,
    "DailyLossLimitRisk": DailyLossLimitRisk,
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
        signals={},
        portfolio=EqualWeightOptimizer(),
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
    name="strategy_list",
    description="List all strategies created in the current session",
)
def strategy_list() -> dict[str, Any]:
    """Return names and summaries of all strategies in session state."""
    return {
        "strategies": [
            {
                "name": name,
                "hypothesis": strat.hypothesis,
                "signals": len(strat.signals),
                "portfolio": strat.portfolio.name,
            }
            for name, strat in sorted(session._strategies.items())
        ],
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

    # Store indicator info in signal's required_indicators when signal is added.
    # For now, store indicators in a _pending_indicators dict on the strategy.
    if not hasattr(strat, "_pending_indicators"):
        strat._pending_indicators = {}
    strat._pending_indicators[name] = (cls(), params or {})
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
    description="Add a rule (entry, exit, order, rebalance, or risk) to a strategy",
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

    # Rules are stored externally; they are passed to Engine.run() at runtime.
    # Store them in a _pending_rules list on the strategy for tool convenience.
    if not hasattr(strat, "_pending_rules"):
        strat._pending_rules = []
    strat._pending_rules.append(rule)

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

    pending_indicators = getattr(strat, "_pending_indicators", {})
    pending_rules = getattr(strat, "_pending_rules", [])
    return {
        "name": strat.name,
        "hypothesis": strat.hypothesis,
        "objectives": strat.objectives,
        "benchmarks": strat.benchmarks,
        "universe": list(strat.universe.symbols) if hasattr(strat.universe, "symbols") else [],
        "indicators": {
            k: {"type": v[0].__class__.__name__, "params": v[1]}
            for k, v in pending_indicators.items()
        },
        "signals": {
            k: {"type": v[0].__class__.__name__, "params": v[1]}
            for k, v in strat.signals.items()
        },
        "portfolio": strat.portfolio.name,
        "rules": [
            {"type": r.__class__.__name__, "name": r.name}
            for r in pending_rules
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


# ---------------------------------------------------------------------------
# Signal metadata tools
# ---------------------------------------------------------------------------


@registry.tool(
    name="signal_describe",
    description="Describe a signal type: show its parameters and usage",
)
def signal_describe(type: str) -> dict[str, Any]:
    """Return signal metadata including parameters."""
    cls = SIGNAL_TYPES.get(type)
    if cls is None:
        return {"error": f"Unknown signal '{type}'. Available: {sorted(SIGNAL_TYPES)}"}

    sig = inspect.signature(cls().compute)
    params = {
        k: str(v.default) if v.default is not v.empty else "(required)"
        for k, v in sig.parameters.items()
        if k not in ("self", "mktdata")
    }

    return {
        "name": getattr(cls, "name", type),
        "description": (cls.__doc__ or "").strip(),
        "params": params,
    }


@registry.tool(
    name="signal_list",
    description="List all available signal types with descriptions",
)
def signal_list() -> dict[str, Any]:
    """Return all signal types with names and descriptions."""
    return {
        "signals": [
            {
                "name": name,
                "description": (cls.__doc__ or "").split("\n")[0].strip(),
            }
            for name, cls in sorted(SIGNAL_TYPES.items())
        ],
    }


# ---------------------------------------------------------------------------
# Rule metadata tools
# ---------------------------------------------------------------------------


@registry.tool(
    name="rule_describe",
    description="Describe a rule type: show its parameters and usage",
)
def rule_describe(type: str) -> dict[str, Any]:
    """Return rule metadata including constructor parameters."""
    cls = RULE_TYPES.get(type)
    if cls is None:
        return {"error": f"Unknown rule '{type}'. Available: {sorted(RULE_TYPES)}"}

    sig = inspect.signature(cls.__init__)
    params = {
        k: str(v.default) if v.default is not v.empty else "(required)"
        for k, v in sig.parameters.items()
        if k != "self"
    }

    return {
        "name": getattr(cls, "name", type),
        "description": (cls.__doc__ or "").strip(),
        "params": params,
    }


@registry.tool(
    name="rule_list",
    description="List all available rule types with descriptions",
)
def rule_list() -> dict[str, Any]:
    """Return all rule types with names and descriptions."""
    return {
        "rules": [
            {
                "name": name,
                "description": (cls.__doc__ or "").split("\n")[0].strip(),
            }
            for name, cls in sorted(RULE_TYPES.items())
        ],
    }
