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
from oxq.indicators.annualized_volatility import AnnualizedVolatility
from oxq.indicators.hurst_exponent import HurstExponent
from oxq.indicators.ichimoku import (
    IchimokuChikou,
    IchimokuKijun,
    IchimokuSenkouA,
    IchimokuSenkouB,
    IchimokuTenkan,
)
from oxq.indicators.log_return import LogReturn
from oxq.indicators.momentum import Momentum
from oxq.indicators.nday_return import NdayReturn
from oxq.indicators.power_ratio import PowerRatio
from oxq.indicators.ratio import Ratio
from oxq.indicators.rolling_mdd import RollingMDD
from oxq.indicators.rolling_volatility import RollingVolatility
from oxq.indicators.simple_momentum import SimpleMomentum
from oxq.indicators.sma import SMA
from oxq.portfolio.optimizers import (
    EqualWeightOptimizer,
    KellyOptimizer,
    PctEquityOptimizer,
    RiskParityOptimizer,
    TopNRankingOptimizer,
)
from oxq.rules.constraint import BlacklistRule, MaxHoldingsRule, RebalanceFrequencyRule
from oxq.rules.exit import ExitRule
from oxq.rules.order import StopLossRule, TakeProfitRule, TrailingStopRule
from oxq.rules.risk import DailyLossLimitRisk, MaxDrawdownRisk
from oxq.signals.comparison import Comparison
from oxq.signals.composite import Composite
from oxq.signals.crossover import Crossover
from oxq.signals.formula import Formula
from oxq.signals.peak import Peak
from oxq.signals.threshold import Threshold
from oxq.signals.timestamp import Timestamp
from oxq.tools import session
from oxq.tools.registry import registry
from oxq.universe.static import StaticUniverse

# ---------------------------------------------------------------------------
# Type registries (string → class mapping)
# ---------------------------------------------------------------------------

INDICATOR_TYPES: dict[str, type] = {
    "ADX": ADX,
    "AnnualizedVolatility": AnnualizedVolatility,
    "AROON": AROON,
    "ATR": ATR,
    "BollingerLower": BollingerLower,
    "BollingerUpper": BollingerUpper,
    "CCI": CCI,
    "DEMA": DEMA,
    "EMA": EMA,
    "HurstExponent": HurstExponent,
    "IchimokuChikou": IchimokuChikou,
    "IchimokuKijun": IchimokuKijun,
    "IchimokuSenkouA": IchimokuSenkouA,
    "IchimokuSenkouB": IchimokuSenkouB,
    "IchimokuTenkan": IchimokuTenkan,
    "LogReturn": LogReturn,
    "MACDHistogram": MACDHistogram,
    "MACDLine": MACDLine,
    "MACDSignal": MACDSignal,
    "MFI": MFI,
    "Momentum": Momentum,
    "NdayReturn": NdayReturn,
    "OBV": OBV,
    "PPO": PPO,
    "PowerRatio": PowerRatio,
    "ROC": ROC,
    "RSI": RSI,
    "Ratio": Ratio,
    "RollingMDD": RollingMDD,
    "RollingVolatility": RollingVolatility,
    "SMA": SMA,
    "SimpleMomentum": SimpleMomentum,
    "StochK": StochK,
    "TEMA": TEMA,
    "VWAP": VWAP,
    "WMA": WMA,
}
SIGNAL_TYPES: dict[str, type] = {
    "Comparison": Comparison,
    "Composite": Composite,
    "Crossover": Crossover,
    "Formula": Formula,
    "Peak": Peak,
    "Threshold": Threshold,
    "Timestamp": Timestamp,
}
RULE_TYPES: dict[str, type] = {
    "BlacklistRule": BlacklistRule,
    "DailyLossLimitRisk": DailyLossLimitRisk,
    "ExitRule": ExitRule,
    "MaxDrawdownRisk": MaxDrawdownRisk,
    "MaxHoldingsRule": MaxHoldingsRule,
    "RebalanceFrequencyRule": RebalanceFrequencyRule,
    "StopLossRule": StopLossRule,
    "TakeProfitRule": TakeProfitRule,
    "TrailingStopRule": TrailingStopRule,
}
PORTFOLIO_TYPES: dict[str, type] = {
    "EqualWeight": EqualWeightOptimizer,
    "RiskParity": RiskParityOptimizer,
    "Kelly": KellyOptimizer,
    "TopNRanking": TopNRankingOptimizer,
    "PctEquity": PctEquityOptimizer,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_required_indicators(
    indicators: dict[str, dict[str, Any]],
) -> dict[str, tuple] | str:
    """Convert a tool-level indicators dict to required_indicators format.

    Returns a dict of (instance, params) tuples, or an error string.
    """
    result: dict[str, tuple] = {}
    for ind_name, ind_def in indicators.items():
        ind_type = ind_def.get("type")
        if not ind_type:
            return f"Indicator '{ind_name}' missing 'type' field"
        cls = INDICATOR_TYPES.get(ind_type)
        if cls is None:
            return f"Unknown indicator type '{ind_type}'. Available: {sorted(INDICATOR_TYPES)}"
        result[ind_name] = (cls(), ind_def.get("params", {}))
    return result


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
    name="strategy_add_signal",
    description=(
        "Add a signal (e.g. Crossover) to a strategy, "
        "along with the indicators it depends on"
    ),
)
def strategy_add_signal(
    strategy: str,
    name: str,
    type: str,
    params: dict[str, Any] | None = None,
    indicators: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Add a signal with its required indicators to an existing strategy."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    cls = SIGNAL_TYPES.get(type)
    if cls is None:
        return {"error": f"Unknown signal type '{type}'. Available: {sorted(SIGNAL_TYPES)}"}

    signal = cls()

    if indicators:
        built = _build_required_indicators(indicators)
        if isinstance(built, str):
            return {"error": built}
        signal.required_indicators = built

    strat.signals[name] = (signal, params or {})
    session._save()
    return {
        "strategy": strategy,
        "signal": name,
        "type": type,
        "params": params or {},
        "indicators": list((indicators or {}).keys()),
    }


@registry.tool(
    name="strategy_add_rule",
    description=(
        "Add a rule (exit, order, constraint, or risk) to a strategy, "
        "along with the indicators it depends on"
    ),
)
def strategy_add_rule(
    strategy: str,
    name: str,
    type: str,
    params: dict[str, Any] | None = None,
    indicators: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Add a rule with its required indicators to an existing strategy."""
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

    if indicators:
        built = _build_required_indicators(indicators)
        if isinstance(built, str):
            return {"error": built}
        rule.required_indicators = built

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

    pending_rules = getattr(strat, "_pending_rules", [])

    # Collect indicators from all components
    def _fmt_indicators(obj: object) -> dict[str, Any]:
        return {
            k: {"type": v[0].__class__.__name__, "params": v[1]}
            for k, v in getattr(obj, "required_indicators", {}).items()
        }

    signals_info = {}
    for k, (sig, params) in strat.signals.items():
        info: dict[str, Any] = {"type": sig.__class__.__name__, "params": params}
        ind = _fmt_indicators(sig)
        if ind:
            info["indicators"] = ind
        signals_info[k] = info

    portfolio_info: dict[str, Any] = {"type": strat.portfolio.name}
    # Show portfolio constructor params
    port_params = {
        k: v for k, v in vars(strat.portfolio).items()
        if not k.startswith("_") and k != "name"
    }
    if port_params:
        portfolio_info["params"] = port_params
    port_ind = _fmt_indicators(strat.portfolio)
    if port_ind:
        portfolio_info["indicators"] = port_ind

    rules_info = []
    for r in pending_rules:
        rule_item: dict[str, Any] = {"type": r.__class__.__name__, "name": r.name}
        # Show rule constructor params
        rule_params = {
            k: v for k, v in vars(r).items()
            if not k.startswith("_") and k != "name"
        }
        if rule_params:
            rule_item["params"] = rule_params
        r_ind = _fmt_indicators(r)
        if r_ind:
            rule_item["indicators"] = r_ind
        rules_info.append(rule_item)

    return {
        "name": strat.name,
        "hypothesis": strat.hypothesis,
        "objectives": strat.objectives,
        "benchmarks": strat.benchmarks,
        "universe": list(strat.universe.symbols) if hasattr(strat.universe, "symbols") else [],
        "signals": signals_info,
        "portfolio": portfolio_info,
        "rules": rules_info,
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


# ---------------------------------------------------------------------------
# Universe tools (strategy-level)
# ---------------------------------------------------------------------------


@registry.tool(
    name="strategy_set_universe",
    description="Set the universe (symbol pool) on a strategy. type='static' for a fixed list, type='filter' for condition-based screening.",
)
def strategy_set_universe(
    strategy: str,
    type: str,
    symbols: list[str] | None = None,
    filters: list[dict[str, Any]] | None = None,
    name: str = "",
) -> dict[str, Any]:
    """Bind a universe to an existing strategy."""
    from oxq.universe import Filter, FilterUniverse

    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    if type == "static":
        if not symbols:
            return {"error": "type='static' requires 'symbols' list."}
        strat.universe = StaticUniverse(symbols=tuple(symbols), name=name)
        session._save()
        return {
            "strategy": strategy,
            "universe_type": "static",
            "symbols": symbols,
        }

    if type == "filter":
        if not symbols:
            return {"error": "type='filter' requires 'symbols' (base pool) list."}
        if not filters:
            return {"error": "type='filter' requires 'filters' list."}
        filter_objs = tuple(
            Filter(column=f["column"], op=f["op"], value=f["value"]) for f in filters
        )
        strat.universe = FilterUniverse(
            base=tuple(symbols), filters=filter_objs, mktdata={}, name=name,
        )
        session._save()
        return {
            "strategy": strategy,
            "universe_type": "filter",
            "base_symbols": symbols,
            "filters": filters,
        }

    return {"error": f"Unknown type '{type}'. Use 'static' or 'filter'."}


# ---------------------------------------------------------------------------
# Portfolio optimizer tools
# ---------------------------------------------------------------------------


@registry.tool(
    name="strategy_set_portfolio",
    description="Set the portfolio optimizer on a strategy (e.g. EqualWeight, RiskParity, Kelly, TopNRanking, PctEquity)",
)
def strategy_set_portfolio(
    strategy: str,
    type: str,
    params: dict[str, Any] | None = None,
    indicators: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Set the portfolio optimizer with its required indicators."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    cls = PORTFOLIO_TYPES.get(type)
    if cls is None:
        return {"error": f"Unknown portfolio type '{type}'. Available: {sorted(PORTFOLIO_TYPES)}"}

    try:
        optimizer = cls(**(params or {}))
    except TypeError as e:
        return {"error": f"Invalid params for {type}: {e}"}

    if indicators:
        built = _build_required_indicators(indicators)
        if isinstance(built, str):
            return {"error": built}
        optimizer.required_indicators = built

    strat.portfolio = optimizer
    session._save()
    return {
        "strategy": strategy,
        "portfolio": type,
        "params": params or {},
        "indicators": list((indicators or {}).keys()),
    }


@registry.tool(
    name="portfolio_list",
    description="List all available portfolio optimizer types",
)
def portfolio_list() -> dict[str, Any]:
    """Return all portfolio optimizer types with descriptions."""
    return {
        "optimizers": [
            {
                "name": name,
                "description": (cls.__doc__ or "").split("\n")[0].strip(),
            }
            for name, cls in sorted(PORTFOLIO_TYPES.items())
        ],
    }


@registry.tool(
    name="portfolio_describe",
    description="Describe a portfolio optimizer: show its parameters and usage",
)
def portfolio_describe(type: str) -> dict[str, Any]:
    """Return portfolio optimizer metadata including constructor parameters."""
    cls = PORTFOLIO_TYPES.get(type)
    if cls is None:
        return {"error": f"Unknown portfolio optimizer '{type}'. Available: {sorted(PORTFOLIO_TYPES)}"}

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
