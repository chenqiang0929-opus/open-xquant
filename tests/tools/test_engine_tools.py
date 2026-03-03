"""Tests for engine tools."""

from __future__ import annotations

import pandas as pd
import pytest

from oxq.tools import session
from oxq.tools.engine import engine_results, engine_run, engine_trade_list
from oxq.tools.strategy import (
    strategy_add_indicator,
    strategy_add_rule,
    strategy_add_signal,
    strategy_create,
)


@pytest.fixture(autouse=True)
def _reset_session():
    """Reset session state before each test."""
    session.clear()


@pytest.fixture()
def sample_data_dir(tmp_path):
    """Create 120-bar trending data as AAPL.parquet."""
    n = 120
    dates = pd.bdate_range("2024-01-01", periods=n)
    closes: list[float] = []
    for i in range(50):
        closes.append(200 - i * 2)       # 200 → 102
    for i in range(40):
        closes.append(102 + i * 2)       # 102 → 180
    for i in range(30):
        closes.append(180 - i * 2)       # 180 → 122

    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n,
        },
        index=dates,
    )
    df.to_parquet(tmp_path / "AAPL.parquet")
    return tmp_path


def _build_full_strategy() -> None:
    """Build a complete SMA crossover strategy via tools."""
    strategy_create(
        name="sma_cross",
        hypothesis="SMA10 crossing above SMA50 predicts positive returns",
        objectives={
            "total_return": {"min": -0.5},
            "max_drawdown": {"max": 0.0},
        },
    )
    strategy_add_indicator(
        strategy="sma_cross", name="sma_10", type="SMA",
        params={"column": "close", "period": 10},
    )
    strategy_add_indicator(
        strategy="sma_cross", name="sma_50", type="SMA",
        params={"column": "close", "period": 50},
    )
    strategy_add_signal(
        strategy="sma_cross", name="cross_up", type="Crossover",
        inputs={"fast": "sma_10", "slow": "sma_50"},
    )
    strategy_add_rule(
        strategy="sma_cross", name="buy", type="EntryRule",
        params={"signal": "cross_up", "shares": 100},
    )
    strategy_add_rule(
        strategy="sma_cross", name="sell", type="ExitRule",
        params={"fast": "sma_10", "slow": "sma_50"},
    )


# ---------------------------------------------------------------------------
# engine_run
# ---------------------------------------------------------------------------


def test_engine_run(sample_data_dir) -> None:
    _build_full_strategy()
    result = engine_run(
        strategy="sma_cross",
        symbols=["AAPL"],
        start="2024-01-01",
        end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    assert "run_id" in result
    assert "error" not in result
    assert result["total_trades"] > 0
    assert result["equity_curve_length"] == 120
    assert result["portfolio"]["cash"] != 100_000.0 or len(result["portfolio"]["positions"]) > 0


def test_engine_run_missing_strategy() -> None:
    result = engine_run(
        strategy="nonexistent", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
    )
    assert "error" in result


def test_engine_run_through_indicator(sample_data_dir) -> None:
    _build_full_strategy()
    result = engine_run(
        strategy="sma_cross",
        symbols=["AAPL"],
        start="2024-01-01",
        end="2024-12-31",
        run_through="indicator",
        data_dir=str(sample_data_dir),
    )
    assert result["total_trades"] == 0
    assert result["equity_curve_length"] == 0


# ---------------------------------------------------------------------------
# engine_results
# ---------------------------------------------------------------------------


def test_engine_results(sample_data_dir) -> None:
    _build_full_strategy()
    run = engine_run(
        strategy="sma_cross", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    run_id = run["run_id"]

    result = engine_results(run_id)
    assert "metrics" in result
    assert "total_return" in result["metrics"]
    assert "sharpe_ratio" in result["metrics"]
    assert "max_drawdown" in result["metrics"]
    assert isinstance(result["metrics"]["total_return"], float)


def test_engine_results_objectives_check(sample_data_dir) -> None:
    _build_full_strategy()
    run = engine_run(
        strategy="sma_cross", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    run_id = run["run_id"]

    result = engine_results(run_id)
    assert len(result["objectives_check"]) > 0
    for check in result["objectives_check"]:
        assert "metric" in check
        assert "actual" in check
        assert "pass" in check


def test_engine_results_not_found() -> None:
    result = engine_results("nonexistent_123")
    assert "error" in result


# ---------------------------------------------------------------------------
# engine_trade_list
# ---------------------------------------------------------------------------


def test_engine_trade_list(sample_data_dir) -> None:
    _build_full_strategy()
    run = engine_run(
        strategy="sma_cross", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    run_id = run["run_id"]

    result = engine_trade_list(run_id)
    assert result["total_trades"] > 0
    assert len(result["trades"]) == result["total_trades"]
    trade = result["trades"][0]
    assert "symbol" in trade
    assert "side" in trade
    assert "shares" in trade
    assert "price" in trade
    assert "date" in trade
    assert trade["symbol"] == "AAPL"


def test_engine_trade_list_not_found() -> None:
    result = engine_trade_list("nonexistent_123")
    assert "error" in result
