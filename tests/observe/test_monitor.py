"""Tests for StrategyMonitor — rolling metrics and bad period detection."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult


def _make_result(
    values: list[float],
    start: str = "2024-01-01",
    benchmark_prices: dict[str, pd.Series] | None = None,
) -> RunResult:
    """Build a RunResult from portfolio values."""
    dates = pd.bdate_range(start, periods=len(values))
    return RunResult(
        portfolio=Portfolio(cash=Decimal(str(values[-1])) if values else Decimal("0")),
        trades=[],
        equity_curve=[(d, v) for d, v in zip(dates, values)],
        mktdata={},
        benchmark_prices=benchmark_prices or {},
    )


class TestBadPeriod:
    def test_frozen(self) -> None:
        from oxq.observe.monitor import BadPeriod
        from datetime import date
        bp = BadPeriod(start=date(2024, 1, 1), end=date(2024, 2, 1), days=22, avg_sharpe=-0.5)
        with pytest.raises(AttributeError):
            bp.days = 10


class TestRollingSharpe:
    def test_length_matches_equity(self) -> None:
        from oxq.observe.monitor import StrategyMonitor
        np.random.seed(42)
        values = (100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))).tolist()
        values = [100.0] + values
        result = _make_result(values)
        monitor = StrategyMonitor(result, roll_window=20)
        assert len(monitor.rolling_sharpe) == 100

    def test_annualized(self) -> None:
        from oxq.observe.monitor import StrategyMonitor
        np.random.seed(42)
        daily_ret = np.random.normal(0.001, 0.01, 100)
        values = (100 * np.cumprod(1 + daily_ret)).tolist()
        values = [100.0] + values
        result = _make_result(values)
        monitor = StrategyMonitor(result, roll_window=20)
        ret_series = pd.Series(daily_ret)
        last_window = ret_series.iloc[-20:]
        expected = float(last_window.mean() / last_window.std() * np.sqrt(252))
        assert monitor.rolling_sharpe.iloc[-1] == pytest.approx(expected, rel=1e-4)


class TestRollingDrawdown:
    def test_always_non_positive(self) -> None:
        from oxq.observe.monitor import StrategyMonitor
        values = [100, 110, 105, 115, 108, 120, 112, 125]
        result = _make_result(values)
        monitor = StrategyMonitor(result, roll_window=3)
        assert (monitor.rolling_drawdown <= 0).all()

    def test_zero_at_new_high(self) -> None:
        from oxq.observe.monitor import StrategyMonitor
        values = [100.0, 110.0, 120.0, 130.0]
        result = _make_result(values)
        monitor = StrategyMonitor(result, roll_window=3)
        assert (monitor.rolling_drawdown == 0).all()

    def test_calculation(self) -> None:
        from oxq.observe.monitor import StrategyMonitor
        values = [100.0, 110.0, 90.0, 95.0]
        result = _make_result(values)
        monitor = StrategyMonitor(result, roll_window=3)
        assert monitor.rolling_drawdown.iloc[2] == pytest.approx(-20.0 / 110.0, rel=1e-6)
        assert monitor.rolling_drawdown.iloc[3] == pytest.approx(-15.0 / 110.0, rel=1e-6)


class TestRollingExcess:
    def test_none_without_benchmark(self) -> None:
        from oxq.observe.monitor import StrategyMonitor
        result = _make_result([100, 110, 120])
        monitor = StrategyMonitor(result)
        assert monitor.rolling_excess is None

    def test_with_benchmark(self) -> None:
        from oxq.observe.monitor import StrategyMonitor
        dates = pd.bdate_range("2024-01-01", periods=50)
        values = np.linspace(100, 120, 50).tolist()
        bench = pd.Series(np.linspace(100, 110, 50), index=dates)
        result = _make_result(values, benchmark_prices={"BENCH": bench})
        monitor = StrategyMonitor(result, benchmark="BENCH", roll_window=20)
        assert monitor.rolling_excess is not None
        assert len(monitor.rolling_excess) > 0


class TestBadPeriods:
    def test_no_bad_periods_in_bull_market(self) -> None:
        from oxq.observe.monitor import StrategyMonitor
        values = np.linspace(100, 200, 200).tolist()
        result = _make_result(values)
        monitor = StrategyMonitor(result, roll_window=20, min_bad_days=10)
        assert len(monitor.bad_periods) == 0

    def test_detects_bad_period(self) -> None:
        from oxq.observe.monitor import StrategyMonitor
        up1 = np.linspace(100, 120, 30).tolist()
        down = np.linspace(120, 80, 40).tolist()
        up2 = np.linspace(80, 110, 30).tolist()
        values = up1 + down[1:] + up2[1:]
        result = _make_result(values)
        monitor = StrategyMonitor(result, roll_window=10, min_bad_days=5)
        assert len(monitor.bad_periods) >= 1
        bp = monitor.bad_periods[0]
        assert bp.days >= 5
        assert bp.avg_sharpe < 0


class TestSummary:
    def test_summary_keys(self) -> None:
        from oxq.observe.monitor import StrategyMonitor
        values = np.linspace(100, 120, 100).tolist()
        result = _make_result(values)
        monitor = StrategyMonitor(result, roll_window=20)
        s = monitor.summary()
        assert "current_sharpe" in s
        assert "current_drawdown" in s
        assert "current_excess" in s
        assert "n_bad_periods" in s
        assert "status" in s

    def test_healthy_status(self) -> None:
        from oxq.observe.monitor import StrategyMonitor
        np.random.seed(42)
        daily = np.random.normal(0.002, 0.005, 200)
        values = (100 * np.cumprod(1 + daily)).tolist()
        values = [100.0] + values
        result = _make_result(values)
        monitor = StrategyMonitor(result, roll_window=20)
        assert monitor.summary()["status"] == "healthy"

    def test_critical_status(self) -> None:
        from oxq.observe.monitor import StrategyMonitor
        values = np.linspace(100, 50, 100).tolist()
        result = _make_result(values)
        monitor = StrategyMonitor(result, roll_window=20)
        assert monitor.summary()["status"] == "critical"
