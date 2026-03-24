"""Tests for oxq.core.registry — register/list API and built-in loading."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from oxq.core.types import Portfolio, RuleResult


# ---------------------------------------------------------------------------
# Dummy classes that satisfy the Protocols
# ---------------------------------------------------------------------------

class _DummyIndicator:
    name = "DummyInd"

    def compute(self, mktdata: pd.DataFrame, **params: object) -> pd.Series:
        return pd.Series(dtype=float)


class _DummySignal:
    name = "DummySig"

    def compute(self, mktdata: pd.DataFrame, **params: object) -> pd.Series:
        return pd.Series(dtype=float)


class _DummyOptimizer:
    name = "DummyOpt"

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        return {"CASH": 1.0}


class _DummyRule:
    name = "DummyRule"

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult:
        return RuleResult()


class _Invalid:
    """Does not satisfy any component protocol."""
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuiltinsLoaded:
    """Verify that _load_builtins populated each registry on import."""

    def test_indicators_loaded(self) -> None:
        from oxq.core.registry import list_indicators

        indicators = list_indicators()
        # Spot-check a handful of built-in indicators
        for name in ("SMA", "EMA", "RSI", "ATR", "BollingerUpper", "HurstExponent"):
            assert name in indicators, f"{name} missing from indicator registry"

    def test_signals_loaded(self) -> None:
        from oxq.core.registry import list_signals

        signals = list_signals()
        for name in ("Crossover", "Threshold", "Formula", "Composite"):
            assert name in signals, f"{name} missing from signal registry"

    def test_portfolio_optimizers_loaded(self) -> None:
        from oxq.core.registry import list_portfolio_optimizers

        optimizers = list_portfolio_optimizers()
        for name in ("EqualWeight", "RiskParity", "Kelly", "TopNRanking", "PctEquity"):
            assert name in optimizers, f"{name} missing from optimizer registry"

    def test_rules_loaded(self) -> None:
        from oxq.core.registry import list_rules

        rules = list_rules()
        for name in (
            "BlacklistRule",
            "StopLossRule",
            "TakeProfitRule",
            "TrailingStopRule",
            "MaxDrawdownRisk",
            "DailyLossLimitRisk",
            "ExitRule",
            "MaxHoldingsRule",
            "RebalanceFrequencyRule",
        ):
            assert name in rules, f"{name} missing from rule registry"


class TestRegisterAPI:
    """Test the public register_xxx / list_xxx functions."""

    def test_register_indicator(self) -> None:
        from oxq.core.registry import list_indicators, register_indicator

        register_indicator(_DummyIndicator)
        assert "DummyInd" in list_indicators()
        assert list_indicators()["DummyInd"] is _DummyIndicator

    def test_register_signal(self) -> None:
        from oxq.core.registry import list_signals, register_signal

        register_signal(_DummySignal)
        assert "DummySig" in list_signals()
        assert list_signals()["DummySig"] is _DummySignal

    def test_register_portfolio_optimizer(self) -> None:
        from oxq.core.registry import (
            list_portfolio_optimizers,
            register_portfolio_optimizer,
        )

        register_portfolio_optimizer(_DummyOptimizer)
        assert "DummyOpt" in list_portfolio_optimizers()
        assert list_portfolio_optimizers()["DummyOpt"] is _DummyOptimizer

    def test_register_rule(self) -> None:
        from oxq.core.registry import list_rules, register_rule

        register_rule(_DummyRule)
        assert "DummyRule" in list_rules()
        assert list_rules()["DummyRule"] is _DummyRule

    def test_register_invalid_raises_type_error(self) -> None:
        from oxq.core.registry import register_indicator

        with pytest.raises(TypeError, match="does not satisfy"):
            register_indicator(_Invalid)

    def test_last_write_wins(self) -> None:
        from oxq.core.registry import list_indicators, register_indicator

        class _AnotherInd:
            name = "DummyInd"  # same name as _DummyIndicator

            def compute(self, mktdata: pd.DataFrame, **params: object) -> pd.Series:
                return pd.Series(dtype=float)

        register_indicator(_DummyIndicator)
        register_indicator(_AnotherInd)
        assert list_indicators()["DummyInd"] is _AnotherInd

    def test_list_returns_copy(self) -> None:
        from oxq.core.registry import list_indicators

        d = list_indicators()
        d["GARBAGE"] = object()  # type: ignore[assignment]
        assert "GARBAGE" not in list_indicators()
