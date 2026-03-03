"""Tests for strategy tools."""

from __future__ import annotations

import pytest

from oxq.tools import session
from oxq.tools.strategy import (
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
