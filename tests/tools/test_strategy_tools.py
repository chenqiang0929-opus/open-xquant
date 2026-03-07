"""Tests for strategy tools."""

from __future__ import annotations

import pytest

from oxq.tools import session
from oxq.tools.strategy import (
    INDICATOR_TYPES,
    RULE_TYPES,
    indicator_describe,
    indicator_list,
    strategy_add_indicator,
    strategy_add_rule,
    strategy_add_signal,
    strategy_create,
    strategy_inspect,
)


@pytest.fixture(autouse=True)
def _reset_session():
    """Reset session state before each test."""
    session.clear()


# ---------------------------------------------------------------------------
# strategy_create
# ---------------------------------------------------------------------------


def test_strategy_create() -> None:
    result = strategy_create(
        name="test",
        hypothesis="SMA crossover predicts returns",
        objectives={"total_return": {"min": 0.05}},
        benchmarks=["SPY"],
    )
    assert result["name"] == "test"
    assert result["hypothesis"] == "SMA crossover predicts returns"
    assert result["objectives"] == {"total_return": {"min": 0.05}}
    assert result["benchmarks"] == ["SPY"]
    assert "test" in session._strategies


def test_strategy_create_missing_hypothesis() -> None:
    result = strategy_create(name="bad", hypothesis="", objectives={"r": {"min": 0.0}})
    assert "error" in result


def test_strategy_create_missing_objectives() -> None:
    result = strategy_create(name="bad", hypothesis="test hypothesis")
    assert "error" in result


# ---------------------------------------------------------------------------
# strategy_add_indicator
# ---------------------------------------------------------------------------


def test_strategy_add_indicator() -> None:
    strategy_create(
        name="s1",
        hypothesis="test",
        objectives={"total_return": {"min": 0.0}},
    )
    result = strategy_add_indicator(
        strategy="s1",
        name="sma_10",
        type="SMA",
        params={"column": "close", "period": 10},
    )
    assert result["indicator"] == "sma_10"
    assert result["type"] == "SMA"
    strat = session._strategies["s1"]
    assert "sma_10" in strat.indicators
    assert strat.indicators["sma_10"][1] == {"column": "close", "period": 10}


def test_strategy_add_indicator_unknown_type() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    result = strategy_add_indicator(strategy="s1", name="x", type="FooBar")
    assert "error" in result
    assert "FooBar" in result["error"]


def test_strategy_add_indicator_not_found() -> None:
    result = strategy_add_indicator(strategy="missing", name="x", type="SMA")
    assert "error" in result


# ---------------------------------------------------------------------------
# strategy_add_signal
# ---------------------------------------------------------------------------


def test_strategy_add_signal() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    result = strategy_add_signal(
        strategy="s1",
        name="cross",
        type="Crossover",
        inputs={"fast": "sma_10", "slow": "sma_50"},
    )
    assert result["signal"] == "cross"
    assert result["type"] == "Crossover"
    strat = session._strategies["s1"]
    assert "cross" in strat.signals
    assert strat.signals["cross"][1]["fast"] == "sma_10"


def test_strategy_add_signal_unknown_type() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    result = strategy_add_signal(strategy="s1", name="x", type="MACD")
    assert "error" in result


# ---------------------------------------------------------------------------
# strategy_add_rule
# ---------------------------------------------------------------------------


def test_strategy_add_rule_entry() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    result = strategy_add_rule(
        strategy="s1",
        name="buy_on_cross",
        type="EntryRule",
        params={"signal": "cross", "shares": 100},
    )
    assert result["type"] == "EntryRule"
    strat = session._strategies["s1"]
    assert len(strat.entry_rules) == 1
    assert strat.entry_rules[0].signal == "cross"
    assert strat.entry_rules[0].shares == 100


def test_strategy_add_rule_exit() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    result = strategy_add_rule(
        strategy="s1",
        name="sell_on_cross",
        type="ExitRule",
        params={"fast": "sma_10", "slow": "sma_50"},
    )
    assert result["type"] == "ExitRule"
    strat = session._strategies["s1"]
    assert len(strat.exit_rules) == 1
    assert strat.exit_rules[0].fast == "sma_10"


def test_strategy_add_rule_unknown_type() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    result = strategy_add_rule(strategy="s1", name="x", type="StopLoss")
    assert "error" in result


def test_strategy_add_rule_invalid_params() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    result = strategy_add_rule(
        strategy="s1", name="bad", type="EntryRule", params={"bad_key": 1},
    )
    assert "error" in result


# ---------------------------------------------------------------------------
# strategy_inspect
# ---------------------------------------------------------------------------


def test_strategy_inspect() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_indicator(strategy="s1", name="sma_10", type="SMA", params={"period": 10})
    strategy_add_signal(
        strategy="s1", name="cross", type="Crossover",
        inputs={"fast": "sma_10", "slow": "sma_50"},
    )
    strategy_add_rule(
        strategy="s1", name="buy", type="EntryRule",
        params={"signal": "cross", "shares": 100},
    )

    result = strategy_inspect("s1")
    assert result["name"] == "s1"
    assert "sma_10" in result["indicators"]
    assert result["indicators"]["sma_10"]["type"] == "SMA"
    assert "cross" in result["signals"]
    assert len(result["entry_rules"]) == 1


def test_strategy_inspect_not_found() -> None:
    result = strategy_inspect("missing")
    assert "error" in result


# ---------------------------------------------------------------------------
# indicator_describe
# ---------------------------------------------------------------------------


def test_indicator_describe_rsi() -> None:
    result = indicator_describe(type="RSI")
    assert result["name"] == "RSI"
    assert "frac" in result["formula"]
    assert result["params"]["period"] == "14"
    assert result["params"]["column"] == "close"
    assert result["depends_on"] == []


def test_indicator_describe_macd_signal_depends_on() -> None:
    result = indicator_describe(type="MACDSignal")
    assert result["name"] == "MACDSignal"
    assert "macd" in result["depends_on"]


def test_indicator_describe_unknown() -> None:
    result = indicator_describe(type="FooBar")
    assert "error" in result
    assert "FooBar" in result["error"]


def test_indicator_describe_all_have_formula() -> None:
    """Every indicator returned by describe should have a non-empty formula."""
    for name in INDICATOR_TYPES:
        result = indicator_describe(type=name)
        assert "error" not in result, f"{name} returned error"
        assert len(result["formula"]) > 0, f"{name} has empty formula"


# ---------------------------------------------------------------------------
# indicator_list
# ---------------------------------------------------------------------------


def test_indicator_list_count() -> None:
    result = indicator_list()
    assert len(result["indicators"]) == 27


def test_indicator_list_structure() -> None:
    result = indicator_list()
    for item in result["indicators"]:
        assert "name" in item
        assert "formula" in item
        assert "description" in item
        assert len(item["formula"]) > 0, f"{item['name']} has empty formula"


def test_indicator_list_sorted() -> None:
    result = indicator_list()
    names = [i["name"] for i in result["indicators"]]
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# strategy_add_rule — order / risk / sized entry rules
# ---------------------------------------------------------------------------


def test_strategy_add_rule_stop_loss() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_rule(
        strategy="s1",
        name="sl",
        type="StopLossRule",
        params={"threshold": 0.05},
    )
    strat = session._strategies["s1"]
    assert len(strat.order_rules) == 1
    assert strat.order_rules[0].threshold == 0.05


def test_strategy_add_rule_take_profit() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_rule(
        strategy="s1",
        name="tp",
        type="TakeProfitRule",
        params={"threshold": 0.15},
    )
    strat = session._strategies["s1"]
    assert len(strat.order_rules) == 1


def test_strategy_add_rule_trailing_stop() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_rule(
        strategy="s1",
        name="ts",
        type="TrailingStopRule",
        params={"trail_pct": 0.05},
    )
    strat = session._strategies["s1"]
    assert len(strat.order_rules) == 1


def test_strategy_add_rule_max_drawdown_risk() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_rule(
        strategy="s1",
        name="mdd",
        type="MaxDrawdownRisk",
        params={"max_drawdown": 0.15},
    )
    strat = session._strategies["s1"]
    assert len(strat.risk_rules) == 1


def test_strategy_add_rule_daily_loss_limit_risk() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_rule(
        strategy="s1",
        name="dll",
        type="DailyLossLimitRisk",
        params={"max_daily_loss": 0.03},
    )
    strat = session._strategies["s1"]
    assert len(strat.risk_rules) == 1


def test_strategy_add_rule_sized_entry() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_rule(
        strategy="s1",
        name="sized",
        type="SizedEntryRule",
        params={"signal": "sig", "shares": 100, "max_position": 500},
    )
    strat = session._strategies["s1"]
    assert len(strat.entry_rules) == 1


# ---------------------------------------------------------------------------
# strategy_inspect — order_rules and risk_rules fields
# ---------------------------------------------------------------------------


def test_strategy_inspect_shows_order_and_risk_rules() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_rule(
        strategy="s1",
        name="sl",
        type="StopLossRule",
        params={"threshold": 0.05},
    )
    strategy_add_rule(
        strategy="s1",
        name="mdd",
        type="MaxDrawdownRisk",
        params={"max_drawdown": 0.15},
    )
    result = strategy_inspect("s1")
    assert len(result["order_rules"]) == 1
    assert len(result["risk_rules"]) == 1


# ---------------------------------------------------------------------------
# RULE_TYPES registry completeness
# ---------------------------------------------------------------------------


def test_rule_types_registry_completeness() -> None:
    expected = {
        "EntryRule",
        "TargetValueEntryRule",
        "FullPositionEntryRule",
        "SizedEntryRule",
        "ExitRule",
        "StopLossRule",
        "TakeProfitRule",
        "TrailingStopRule",
        "RebalanceRule",
        "MaxDrawdownRisk",
        "DailyLossLimitRisk",
    }
    assert set(RULE_TYPES.keys()) == expected
