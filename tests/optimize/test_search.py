"""Tests for GridSearch — parameter optimization and helpers."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from oxq.core.engine import Engine
from oxq.core.strategy import Strategy
from oxq.core.types import Portfolio
from oxq.indicators.sma import SMA
from oxq.optimize.paramset import ParameterSet
from oxq.optimize.search import (
    METRIC_DIRECTIONS,
    GridSearch,
    SearchResult,
    TrialResult,
    _apply_params,
    _apply_rule_params,
    _extract_metric,
    _resolve_direction,
)
from oxq.portfolio.analytics import RunResult
from oxq.rules.entry import EntryRule, SizedEntryRule
from oxq.rules.exit import ExitRule
from oxq.rules.order import StopLossRule, TakeProfitRule, TrailingStopRule
from oxq.rules.rebalance import RebalanceRule
from oxq.rules.risk import DailyLossLimitRisk, MaxDrawdownRisk
from oxq.signals.crossover import Crossover
from oxq.trade.sim_broker import SimBroker
from oxq.universe.static import StaticUniverse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeMarketDataProvider:
    """In-memory market data provider for testing."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        df = self._data[symbol]
        return df[(df.index >= start) & (df.index <= end)]

    def get_latest(self, symbol: str) -> pd.Series:
        return self._data[symbol].iloc[-1]


def _make_result(values: list[float]) -> RunResult:
    """Build a RunResult from a sequence of portfolio values."""
    dates = pd.bdate_range("2024-01-01", periods=len(values))
    equity_curve = [(d, v) for d, v in zip(dates, values)]
    return RunResult(
        portfolio=Portfolio(cash=Decimal(str(values[-1])) if values else Decimal("0")),
        trades=[],
        equity_curve=equity_curve,
        mktdata={},
    )


def _make_trending_data(n: int = 120) -> dict[str, pd.DataFrame]:
    """Trending price data: down → up → down for SMA crossover signals."""
    dates = pd.bdate_range("2024-01-01", periods=n)
    closes: list[float] = []
    for i in range(50):
        closes.append(200 - i * 2)       # 200 → 102
    for i in range(40):
        closes.append(102 + i * 2)       # 102 → 180
    for i in range(n - 90):
        closes.append(180 - i * 2)       # 180 → ...
    return {
        "AAPL": pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1 for c in closes],
                "low": [c - 1 for c in closes],
                "close": closes,
                "volume": [1_000_000] * n,
            },
            index=dates,
        ),
    }


def _make_strategy(fast: int = 10, slow: int = 50) -> Strategy:
    return Strategy(
        name="test_sma",
        hypothesis="SMA crossover",
        universe=StaticUniverse(("AAPL",)),
        indicators={
            "sma_fast": (SMA(), {"period": fast}),
            "sma_slow": (SMA(), {"period": slow}),
        },
        signals={
            "sma_cross": (Crossover(), {"fast": "sma_fast", "slow": "sma_slow"}),
        },
        entry_rules=[EntryRule(signal="sma_cross", shares=100)],
        exit_rules=[ExitRule(fast="sma_fast", slow="sma_slow")],
    )


# ---------------------------------------------------------------------------
# _resolve_direction
# ---------------------------------------------------------------------------


def test_resolve_direction_explicit() -> None:
    assert _resolve_direction("sharpe_ratio", "maximize") == "maximize"
    assert _resolve_direction("sharpe_ratio", "minimize") == "minimize"


def test_resolve_direction_from_registry() -> None:
    assert _resolve_direction("sharpe_ratio", None) == "maximize"
    assert _resolve_direction("annualized_volatility", None) == "minimize"
    assert _resolve_direction("max_drawdown", None) == "maximize"


def test_resolve_direction_unknown_defaults_maximize() -> None:
    assert _resolve_direction("unknown_metric", None) == "maximize"


def test_resolve_direction_callable_defaults_maximize() -> None:
    assert _resolve_direction(lambda r: 0.0, None) == "maximize"


def test_resolve_direction_invalid_raises() -> None:
    with pytest.raises(ValueError, match="metric_direction"):
        _resolve_direction("sharpe_ratio", "invalid")


# ---------------------------------------------------------------------------
# _extract_metric
# ---------------------------------------------------------------------------


def test_extract_metric_by_name() -> None:
    result = _make_result([100, 110, 115])
    val = _extract_metric(result, "total_return")
    assert val == pytest.approx(0.15, rel=1e-4)


def test_extract_metric_by_callable() -> None:
    result = _make_result([100, 110, 115])
    val = _extract_metric(result, lambda r: r.total_return() * 2)
    assert val == pytest.approx(0.30, rel=1e-4)


def test_extract_metric_unknown_raises() -> None:
    result = _make_result([100, 110])
    with pytest.raises(ValueError, match="Unknown metric"):
        _extract_metric(result, "nonexistent_metric")


# ---------------------------------------------------------------------------
# _apply_params
# ---------------------------------------------------------------------------


def test_apply_params_modifies_indicator() -> None:
    strategy = _make_strategy(fast=10, slow=50)
    params = {"sma_fast": {"period": 20}}
    new = _apply_params(strategy, params)

    # New strategy has updated param
    assert new.indicators["sma_fast"][1]["period"] == 20
    # Unmodified indicator is unchanged
    assert new.indicators["sma_slow"][1]["period"] == 50
    # Original is not mutated
    assert strategy.indicators["sma_fast"][1]["period"] == 10


def test_apply_params_modifies_signal() -> None:
    strategy = _make_strategy()
    params = {"sma_cross": {"fast": "sma_new", "slow": "sma_slow"}}
    new = _apply_params(strategy, params)
    assert new.signals["sma_cross"][1]["fast"] == "sma_new"


def test_apply_params_preserves_name_and_universe() -> None:
    strategy = _make_strategy()
    new = _apply_params(strategy, {})
    assert new.name == strategy.name
    assert new.universe is strategy.universe


def test_apply_params_does_not_mutate_original() -> None:
    strategy = _make_strategy(fast=10, slow=50)
    params = {"sma_fast": {"period": 99}}
    _apply_params(strategy, params)
    assert strategy.indicators["sma_fast"][1]["period"] == 10


# ---------------------------------------------------------------------------
# _apply_params — rule support
# ---------------------------------------------------------------------------


def _make_strategy_with_rules() -> Strategy:
    return Strategy(
        name="test_with_rules",
        hypothesis="SMA crossover with risk management",
        universe=StaticUniverse(("AAPL",)),
        indicators={
            "sma_fast": (SMA(), {"period": 10}),
            "sma_slow": (SMA(), {"period": 50}),
        },
        signals={
            "sma_cross": (Crossover(), {"fast": "sma_fast", "slow": "sma_slow"}),
        },
        entry_rules=[EntryRule(signal="sma_cross", shares=100)],
        exit_rules=[ExitRule(fast="sma_fast", slow="sma_slow")],
        order_rules=[
            StopLossRule(threshold=0.05),
            TakeProfitRule(threshold=0.15),
        ],
        risk_rules=[MaxDrawdownRisk(max_drawdown=0.15)],
    )


def test_apply_params_modifies_order_rule() -> None:
    """_apply_params can override StopLossRule.threshold via rule name."""
    strategy = _make_strategy_with_rules()
    params = {"StopLossRule": {"threshold": 0.10}}
    new = _apply_params(strategy, params)

    stop_rule = [r for r in new.order_rules if r.name == "StopLossRule"][0]
    assert stop_rule.threshold == 0.10

    # TakeProfitRule unchanged
    tp_rule = [r for r in new.order_rules if r.name == "TakeProfitRule"][0]
    assert tp_rule.threshold == 0.15


def test_apply_params_modifies_risk_rule() -> None:
    strategy = _make_strategy_with_rules()
    params = {"MaxDrawdownRisk": {"max_drawdown": 0.25}}
    new = _apply_params(strategy, params)

    risk_rule = new.risk_rules[0]
    assert risk_rule.max_drawdown == 0.25


def test_apply_params_does_not_mutate_original_rules() -> None:
    strategy = _make_strategy_with_rules()
    params = {"StopLossRule": {"threshold": 0.99}}
    _apply_params(strategy, params)

    # Original is unchanged
    stop_rule = [r for r in strategy.order_rules if r.name == "StopLossRule"][0]
    assert stop_rule.threshold == 0.05


def test_apply_params_mixed_indicators_and_rules() -> None:
    """Params can target both indicators and rules in one call."""
    strategy = _make_strategy_with_rules()
    params = {
        "sma_fast": {"period": 20},
        "StopLossRule": {"threshold": 0.08},
        "TakeProfitRule": {"threshold": 0.20},
    }
    new = _apply_params(strategy, params)

    assert new.indicators["sma_fast"][1]["period"] == 20
    stop = [r for r in new.order_rules if r.name == "StopLossRule"][0]
    assert stop.threshold == 0.08
    tp = [r for r in new.order_rules if r.name == "TakeProfitRule"][0]
    assert tp.threshold == 0.20


# ---------------------------------------------------------------------------
# _apply_rule_params
# ---------------------------------------------------------------------------


def test_apply_rule_params_basic() -> None:
    rules = [StopLossRule(threshold=0.05), TakeProfitRule(threshold=0.15)]
    params = {"StopLossRule": {"threshold": 0.10}}
    new_rules = _apply_rule_params(rules, params)

    assert len(new_rules) == 2
    assert new_rules[0].threshold == 0.10
    assert new_rules[1].threshold == 0.15
    # Original unchanged
    assert rules[0].threshold == 0.05


def test_apply_rule_params_empty_params() -> None:
    rules = [StopLossRule(threshold=0.05)]
    new_rules = _apply_rule_params(rules, {})
    assert new_rules[0].threshold == 0.05


def test_apply_rule_params_empty_rules() -> None:
    new_rules = _apply_rule_params([], {"StopLossRule": {"threshold": 0.10}})
    assert new_rules == []


def test_apply_rule_params_multiple_attrs_on_same_rule() -> None:
    """Override multiple attributes on the same rule instance."""
    rules = [SizedEntryRule(signal="sig", shares=100, max_position=500)]
    params = {"SizedEntryRule": {"shares": 200, "max_position": 1000}}
    new_rules = _apply_rule_params(rules, params)
    assert new_rules[0].shares == 200
    assert new_rules[0].max_position == 1000
    # Original unchanged
    assert rules[0].shares == 100


def test_apply_rule_params_rule_without_name_attr() -> None:
    """Rules without a name attribute are deep-copied unchanged."""

    class _NoNameRule:
        def __init__(self, x: int) -> None:
            self.x = x

    rules = [_NoNameRule(x=5)]
    new_rules = _apply_rule_params(rules, {"_NoNameRule": {"x": 99}})
    # No name attribute → no match → unchanged
    assert new_rules[0].x == 5


def test_apply_params_modifies_trailing_stop_rule() -> None:
    strategy = Strategy(
        name="test",
        hypothesis="test",
        universe=StaticUniverse(("AAPL",)),
        indicators={"sma_fast": (SMA(), {"period": 10}), "sma_slow": (SMA(), {"period": 50})},
        signals={"sma_cross": (Crossover(), {"fast": "sma_fast", "slow": "sma_slow"})},
        entry_rules=[EntryRule(signal="sma_cross", shares=100)],
        exit_rules=[ExitRule(fast="sma_fast", slow="sma_slow")],
        order_rules=[TrailingStopRule(trail_pct=0.05)],
    )
    new = _apply_params(strategy, {"TrailingStopRule": {"trail_pct": 0.08}})
    assert new.order_rules[0].trail_pct == 0.08
    assert strategy.order_rules[0].trail_pct == 0.05


def test_apply_params_modifies_entry_rule() -> None:
    strategy = _make_strategy_with_rules()
    params = {"EntryRule": {"shares": 200}}
    new = _apply_params(strategy, params)
    assert new.entry_rules[0].shares == 200
    assert strategy.entry_rules[0].shares == 100


def test_apply_params_modifies_rebalance_rule() -> None:
    strategy = Strategy(
        name="test",
        hypothesis="test",
        universe=StaticUniverse(("AAPL",)),
        indicators={"sma_fast": (SMA(), {"period": 10}), "sma_slow": (SMA(), {"period": 50})},
        signals={"sma_cross": (Crossover(), {"fast": "sma_fast", "slow": "sma_slow"})},
        entry_rules=[],
        exit_rules=[],
        rebalance_rules=[RebalanceRule(weight_col="tw", frequency=10)],
    )
    new = _apply_params(strategy, {"RebalanceRule": {"frequency": 20}})
    assert new.rebalance_rules[0].frequency == 20
    assert strategy.rebalance_rules[0].frequency == 10


def test_apply_params_modifies_daily_loss_limit_risk() -> None:
    strategy = Strategy(
        name="test",
        hypothesis="test",
        universe=StaticUniverse(("AAPL",)),
        indicators={"sma_fast": (SMA(), {"period": 10}), "sma_slow": (SMA(), {"period": 50})},
        signals={"sma_cross": (Crossover(), {"fast": "sma_fast", "slow": "sma_slow"})},
        entry_rules=[EntryRule(signal="sma_cross", shares=100)],
        exit_rules=[ExitRule(fast="sma_fast", slow="sma_slow")],
        risk_rules=[DailyLossLimitRisk(max_daily_loss=0.03)],
    )
    new = _apply_params(strategy, {"DailyLossLimitRisk": {"max_daily_loss": 0.05}})
    assert new.risk_rules[0].max_daily_loss == 0.05
    assert strategy.risk_rules[0].max_daily_loss == 0.03


def test_apply_params_unmatched_rule_names_are_ignored() -> None:
    """Params for names not in the strategy are silently ignored."""
    strategy = _make_strategy_with_rules()
    params = {"NonexistentRule": {"threshold": 0.99}}
    new = _apply_params(strategy, params)
    # All rules unchanged
    for orig, copy in zip(strategy.order_rules, new.order_rules):
        assert orig.threshold == copy.threshold


# ---------------------------------------------------------------------------
# SearchResult — edge cases
# ---------------------------------------------------------------------------


def test_search_result_top_n_larger_than_results() -> None:
    """top_n(n) with n > len returns all results."""
    trials = [_make_trial(0.5), _make_trial(0.8)]
    sr = SearchResult(
        all_results=trials,
        paramset=ParameterSet("test"),
        metric="sharpe_ratio",
        metric_direction="maximize",
    )
    top5 = sr.top_n(5)
    assert len(top5) == 2
    assert top5[0].metric_value == 0.8


def test_search_result_to_dataframe_with_rule_params() -> None:
    """to_dataframe flattens rule params as component.param columns."""
    trials = [
        TrialResult(
            params={
                "sma_fast": {"period": 10},
                "StopLossRule": {"threshold": 0.05},
            },
            metric_value=0.5,
            run_result=_make_result([100, 105, 110]),
        ),
    ]
    sr = SearchResult(
        all_results=trials,
        paramset=ParameterSet("test"),
        metric="sharpe_ratio",
        metric_direction="maximize",
    )
    df = sr.to_dataframe()
    assert "sma_fast.period" in df.columns
    assert "StopLossRule.threshold" in df.columns
    assert df["StopLossRule.threshold"].iloc[0] == 0.05


# ---------------------------------------------------------------------------
# METRIC_DIRECTIONS registry
# ---------------------------------------------------------------------------


def test_metric_directions_has_standard_metrics() -> None:
    assert "sharpe_ratio" in METRIC_DIRECTIONS
    assert "total_return" in METRIC_DIRECTIONS
    assert "max_drawdown" in METRIC_DIRECTIONS
    assert "annualized_volatility" in METRIC_DIRECTIONS


def test_max_drawdown_direction_is_maximize() -> None:
    """max_drawdown is negative; maximize = least bad."""
    assert METRIC_DIRECTIONS["max_drawdown"] == "maximize"


# ---------------------------------------------------------------------------
# TrialResult / SearchResult
# ---------------------------------------------------------------------------


def _make_trial(metric_value: float, values: list[float] | None = None) -> TrialResult:
    if values is None:
        values = [100, 100 + metric_value * 100]
    return TrialResult(
        params={"sma": {"period": int(abs(metric_value) * 100)}},
        metric_value=metric_value,
        run_result=_make_result(values),
    )


def test_search_result_best_maximize() -> None:
    trials = [_make_trial(0.5), _make_trial(1.2), _make_trial(0.8)]
    sr = SearchResult(
        all_results=trials,
        paramset=ParameterSet("test"),
        metric="sharpe_ratio",
        metric_direction="maximize",
    )
    assert sr.best.metric_value == 1.2


def test_search_result_best_minimize() -> None:
    trials = [_make_trial(0.5), _make_trial(0.1), _make_trial(0.8)]
    sr = SearchResult(
        all_results=trials,
        paramset=ParameterSet("test"),
        metric="annualized_volatility",
        metric_direction="minimize",
    )
    assert sr.best.metric_value == 0.1


def test_search_result_top_n() -> None:
    trials = [_make_trial(v) for v in [0.1, 0.5, 0.3, 0.9, 0.7]]
    sr = SearchResult(
        all_results=trials,
        paramset=ParameterSet("test"),
        metric="sharpe_ratio",
        metric_direction="maximize",
    )
    top3 = sr.top_n(3)
    assert len(top3) == 3
    assert [t.metric_value for t in top3] == [0.9, 0.7, 0.5]


def test_search_result_to_dataframe() -> None:
    trials = [
        TrialResult(
            params={"sma_fast": {"period": 10}, "sma_slow": {"period": 50}},
            metric_value=0.5,
            run_result=_make_result([100, 105, 110]),
        ),
    ]
    sr = SearchResult(
        all_results=trials,
        paramset=ParameterSet("test"),
        metric="sharpe_ratio",
        metric_direction="maximize",
    )
    df = sr.to_dataframe()
    assert len(df) == 1
    assert "sma_fast.period" in df.columns
    assert "sma_slow.period" in df.columns
    assert "metric_value" in df.columns
    assert "total_return" in df.columns
    assert "sharpe_ratio" in df.columns
    assert df["sma_fast.period"].iloc[0] == 10


# ---------------------------------------------------------------------------
# GridSearch.run — integration
# ---------------------------------------------------------------------------


def test_grid_search_runs_all_combos() -> None:
    """GridSearch evaluates every valid parameter combination."""
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)

    ps = ParameterSet("test")
    ps.add("sma_fast", "period", values=[5, 10])
    ps.add("sma_slow", "period", values=[30, 50])
    ps.add_constraint("sma_fast.period < sma_slow.period")

    strategy = _make_strategy()
    gs = GridSearch(ps)
    result = gs.run(
        strategy=strategy,
        market=market,
        broker_factory=SimBroker,
        start="2024-01-01",
        end="2024-12-31",
        metric="sharpe_ratio",
    )

    # 2 × 2 = 4, all valid (5<30, 5<50, 10<30, 10<50)
    assert len(result.all_results) == 4
    assert result.metric == "sharpe_ratio"
    assert result.metric_direction == "maximize"

    # Best should have the highest sharpe
    best = result.best
    for trial in result.all_results:
        assert best.metric_value >= trial.metric_value


def test_grid_search_custom_metric() -> None:
    """GridSearch with a callable metric works."""
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)

    ps = ParameterSet("test")
    ps.add("sma_fast", "period", values=[5, 10])

    strategy = _make_strategy()
    gs = GridSearch(ps)
    result = gs.run(
        strategy=strategy,
        market=market,
        broker_factory=SimBroker,
        start="2024-01-01",
        end="2024-12-31",
        metric=lambda r: r.total_return() + r.sharpe_ratio(),
    )

    assert len(result.all_results) == 2
    assert result.metric == "<custom>"


def test_grid_search_minimize_direction() -> None:
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)

    ps = ParameterSet("test")
    ps.add("sma_fast", "period", values=[5, 10])

    strategy = _make_strategy()
    gs = GridSearch(ps)
    result = gs.run(
        strategy=strategy,
        market=market,
        broker_factory=SimBroker,
        start="2024-01-01",
        end="2024-12-31",
        metric="annualized_volatility",
    )

    assert result.metric_direction == "minimize"
    best = result.best
    for trial in result.all_results:
        assert best.metric_value <= trial.metric_value


def test_grid_search_with_rule_params() -> None:
    """GridSearch can optimize rule parameters like StopLossRule.threshold."""
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)

    ps = ParameterSet("rule_tuning")
    ps.add("StopLossRule", "threshold", values=[0.03, 0.05, 0.10])

    strategy = Strategy(
        name="test_rule_opt",
        hypothesis="Optimize stop loss threshold",
        universe=StaticUniverse(("AAPL",)),
        indicators={
            "sma_fast": (SMA(), {"period": 10}),
            "sma_slow": (SMA(), {"period": 50}),
        },
        signals={
            "sma_cross": (Crossover(), {"fast": "sma_fast", "slow": "sma_slow"}),
        },
        entry_rules=[EntryRule(signal="sma_cross", shares=100)],
        exit_rules=[ExitRule(fast="sma_fast", slow="sma_slow")],
        order_rules=[StopLossRule(threshold=0.05)],
    )

    gs = GridSearch(ps)
    result = gs.run(
        strategy=strategy,
        market=market,
        broker_factory=SimBroker,
        start="2024-01-01",
        end="2024-12-31",
        metric="sharpe_ratio",
    )

    assert len(result.all_results) == 3
    # Verify each trial used a different threshold
    thresholds = [t.params["StopLossRule"]["threshold"] for t in result.all_results]
    assert set(thresholds) == {0.03, 0.05, 0.10}


def test_grid_search_mixed_indicator_and_rule_params() -> None:
    """GridSearch can optimize indicators and rules simultaneously."""
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)

    ps = ParameterSet("mixed")
    ps.add("sma_fast", "period", values=[5, 10])
    ps.add("StopLossRule", "threshold", values=[0.05, 0.10])

    strategy = Strategy(
        name="test_mixed_opt",
        hypothesis="Optimize SMA + stop loss",
        universe=StaticUniverse(("AAPL",)),
        indicators={
            "sma_fast": (SMA(), {"period": 10}),
            "sma_slow": (SMA(), {"period": 50}),
        },
        signals={
            "sma_cross": (Crossover(), {"fast": "sma_fast", "slow": "sma_slow"}),
        },
        entry_rules=[EntryRule(signal="sma_cross", shares=100)],
        exit_rules=[ExitRule(fast="sma_fast", slow="sma_slow")],
        order_rules=[StopLossRule(threshold=0.05)],
    )

    gs = GridSearch(ps)
    result = gs.run(
        strategy=strategy,
        market=market,
        broker_factory=SimBroker,
        start="2024-01-01",
        end="2024-12-31",
        metric="sharpe_ratio",
    )

    # 2 fast periods × 2 thresholds = 4 combos
    assert len(result.all_results) == 4
