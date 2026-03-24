"""Component registry — single source of truth for Indicators, Signals, Optimizers, and Rules.

All built-in components are registered at import time via ``_load_builtins()``.
Third-party packages can register additional components via the public
``register_indicator``, ``register_signal``, ``register_portfolio_optimizer``,
and ``register_rule`` functions.
"""

from __future__ import annotations

from typing import Any

from oxq.core.types import Indicator, PortfolioOptimizer, Rule, Signal

# ---------------------------------------------------------------------------
# Private registry dicts: name -> class
# ---------------------------------------------------------------------------

_INDICATOR_REGISTRY: dict[str, type] = {}
_SIGNAL_REGISTRY: dict[str, type] = {}
_PORTFOLIO_OPTIMIZER_REGISTRY: dict[str, type] = {}
_RULE_REGISTRY: dict[str, type] = {}


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _register(cls: type, protocol: type, registry: dict[str, type]) -> None:
    """Validate *cls* against *protocol* and store it in *registry*.

    Instantiates ``cls()`` and checks ``isinstance(instance, protocol)``.
    If the class requires constructor arguments (raises ``TypeError``),
    falls back to a structural check on the class itself.
    """
    try:
        instance = cls()
        if not isinstance(instance, protocol):
            msg = f"{cls.__name__} does not satisfy {protocol.__name__} protocol"
            raise TypeError(msg)
    except TypeError as exc:
        # Re-raise if the error came from our own protocol check
        if "does not satisfy" in str(exc):
            raise
        # Constructor requires arguments — check class-level attributes instead
        _check_class_structure(cls, protocol)

    name: str = getattr(cls, "name", cls.__name__)
    registry[name] = cls


def _check_class_structure(cls: type, protocol: type) -> None:
    """Verify *cls* has the methods/attributes declared by *protocol*."""
    # Collect required members from the protocol (skip dunder and private)
    hints: dict[str, Any] = {}
    for base in protocol.__mro__:
        hints.update(getattr(base, "__annotations__", {}))

    for attr_name in hints:
        if attr_name.startswith("_"):
            continue
        if not hasattr(cls, attr_name):
            msg = f"{cls.__name__} does not satisfy {protocol.__name__} protocol"
            raise TypeError(msg)

    # Check callable members from the protocol (e.g. compute, optimize, evaluate)
    for member_name in dir(protocol):
        if member_name.startswith("_"):
            continue
        proto_member = getattr(protocol, member_name, None)
        if callable(proto_member) and not isinstance(proto_member, type):
            if not hasattr(cls, member_name) or not callable(
                getattr(cls, member_name, None)
            ):
                msg = (
                    f"{cls.__name__} does not satisfy {protocol.__name__} protocol"
                )
                raise TypeError(msg)


# ---------------------------------------------------------------------------
# Public register functions
# ---------------------------------------------------------------------------

def register_indicator(cls: type) -> None:
    """Register an Indicator class."""
    _register(cls, Indicator, _INDICATOR_REGISTRY)


def register_signal(cls: type) -> None:
    """Register a Signal class."""
    _register(cls, Signal, _SIGNAL_REGISTRY)


def register_portfolio_optimizer(cls: type) -> None:
    """Register a PortfolioOptimizer class."""
    _register(cls, PortfolioOptimizer, _PORTFOLIO_OPTIMIZER_REGISTRY)


def register_rule(cls: type) -> None:
    """Register a Rule class."""
    _register(cls, Rule, _RULE_REGISTRY)


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------

def list_indicators() -> dict[str, type]:
    """Return a copy of the indicator registry."""
    return dict(_INDICATOR_REGISTRY)


def list_signals() -> dict[str, type]:
    """Return a copy of the signal registry."""
    return dict(_SIGNAL_REGISTRY)


def list_portfolio_optimizers() -> dict[str, type]:
    """Return a copy of the portfolio optimizer registry."""
    return dict(_PORTFOLIO_OPTIMIZER_REGISTRY)


def list_rules() -> dict[str, type]:
    """Return a copy of the rule registry."""
    return dict(_RULE_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in registration
# ---------------------------------------------------------------------------

def _load_builtins() -> None:
    """Import and register all built-in components."""

    # -- Indicators ----------------------------------------------------------
    from oxq.indicators import (
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
        AnnualizedVolatility,
        BollingerLower,
        BollingerUpper,
        HurstExponent,
        IchimokuChikou,
        IchimokuKijun,
        IchimokuSenkouA,
        IchimokuSenkouB,
        IchimokuTenkan,
        LogReturn,
        MACDHistogram,
        MACDLine,
        MACDSignal,
        Momentum,
        NdayReturn,
        PowerRatio,
        Ratio,
        RollingMDD,
        RollingVolatility,
        SMA,
        SimpleMomentum,
        StochK,
    )

    for cls in (
        ADX, AROON, ATR, CCI, DEMA, EMA, MFI, OBV, PPO, ROC, RSI, TEMA,
        VWAP, WMA, AnnualizedVolatility, BollingerLower, BollingerUpper,
        HurstExponent, IchimokuChikou, IchimokuKijun, IchimokuSenkouA,
        IchimokuSenkouB, IchimokuTenkan, LogReturn, MACDHistogram, MACDLine,
        MACDSignal, Momentum, NdayReturn, PowerRatio, Ratio, RollingMDD,
        RollingVolatility, SMA, SimpleMomentum, StochK,
    ):
        _register(cls, Indicator, _INDICATOR_REGISTRY)

    # -- Signals -------------------------------------------------------------
    from oxq.signals import (
        Comparison,
        Composite,
        Crossover,
        Formula,
        Peak,
        Threshold,
        Timestamp,
    )

    for cls in (Comparison, Composite, Crossover, Formula, Peak, Threshold, Timestamp):
        _register(cls, Signal, _SIGNAL_REGISTRY)

    # -- Portfolio Optimizers ------------------------------------------------
    from oxq.portfolio.optimizers import (
        EqualWeightOptimizer,
        KellyOptimizer,
        PctEquityOptimizer,
        RiskParityOptimizer,
        TopNRankingOptimizer,
    )

    for cls in (
        EqualWeightOptimizer,
        RiskParityOptimizer,
        KellyOptimizer,
        TopNRankingOptimizer,
        PctEquityOptimizer,
    ):
        _register(cls, PortfolioOptimizer, _PORTFOLIO_OPTIMIZER_REGISTRY)

    # -- Rules ---------------------------------------------------------------
    from oxq.rules import (
        BlacklistRule,
        DailyLossLimitRisk,
        ExitRule,
        MaxDrawdownRisk,
        MaxHoldingsRule,
        RebalanceFrequencyRule,
        StopLossRule,
        TakeProfitRule,
        TrailingStopRule,
    )

    for cls in (
        BlacklistRule,
        DailyLossLimitRisk,
        ExitRule,
        MaxDrawdownRisk,
        MaxHoldingsRule,
        RebalanceFrequencyRule,
        StopLossRule,
        TakeProfitRule,
        TrailingStopRule,
    ):
        _register(cls, Rule, _RULE_REGISTRY)


# Register built-ins at module load time.
_load_builtins()
