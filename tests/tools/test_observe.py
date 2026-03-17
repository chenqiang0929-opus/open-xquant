"""Tests for observe tools — strategy diagnostics MCP tools."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.tools import session


@pytest.fixture(autouse=True)
def _clear_session():
    """Clear session state before each test."""
    session.clear()
    yield
    session.clear()


def _setup_run_result(run_id: str = "test_strat_123", n_days: int = 100) -> str:
    """Create and store a RunResult in session."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    values = (100_000 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n_days))).tolist()

    mktdata = {}
    for sym in ["AAPL", "GOOG"]:
        returns = np.random.normal(0.0005, 0.02, n_days)
        close = 100 * np.cumprod(1 + returns)
        mktdata[sym] = pd.DataFrame({"close": close}, index=dates)

    bench = pd.Series(np.linspace(100, 110, n_days), index=dates)

    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(d, v) for d, v in zip(dates, values)],
        mktdata=mktdata,
        benchmark_prices={"BENCH": bench},
    )
    session._run_results[run_id] = result
    return run_id


class TestMonitorCreate:
    def test_creates_monitor(self):
        from oxq.tools.observe import observe_monitor_create

        run_id = _setup_run_result()
        result = observe_monitor_create(run_id, roll_window=20)
        assert "monitor_id" in result
        assert result["status"] in ("healthy", "warning", "critical")
        assert "current_sharpe" in result
        assert "current_drawdown" in result

    def test_with_benchmark(self):
        from oxq.tools.observe import observe_monitor_create

        run_id = _setup_run_result()
        result = observe_monitor_create(run_id, benchmark="BENCH", roll_window=20)
        assert "monitor_id" in result
        assert result["current_excess"] is not None

    def test_missing_run_id(self):
        from oxq.tools.observe import observe_monitor_create

        result = observe_monitor_create("nonexistent")
        assert "error" in result


class TestMonitorSummary:
    def test_returns_summary(self):
        from oxq.tools.observe import observe_monitor_create, observe_monitor_summary

        run_id = _setup_run_result()
        create_result = observe_monitor_create(run_id, roll_window=20)
        monitor_id = create_result["monitor_id"]
        result = observe_monitor_summary(monitor_id)
        assert "current_sharpe" in result
        assert "bad_periods" in result

    def test_missing_monitor(self):
        from oxq.tools.observe import observe_monitor_summary

        result = observe_monitor_summary("nonexistent")
        assert "error" in result


class TestDetectMarketState:
    def test_detects_states(self):
        from oxq.tools.observe import observe_detect_market_state

        run_id = _setup_run_result()
        result = observe_detect_market_state(run_id, vol_lookback=20)
        assert "detector_id" in result
        assert "thresholds" in result
        assert "vol_median" in result["thresholds"]
        assert "state_counts" in result

    def test_missing_run_id(self):
        from oxq.tools.observe import observe_detect_market_state

        result = observe_detect_market_state("nonexistent")
        assert "error" in result


class TestPerformanceByState:
    def test_returns_performance(self):
        from oxq.tools.observe import (
            observe_detect_market_state,
            observe_performance_by_state,
        )

        run_id = _setup_run_result()
        det_result = observe_detect_market_state(run_id, vol_lookback=20)
        detector_id = det_result["detector_id"]
        result = observe_performance_by_state(detector_id, run_id)
        assert "performance" in result
        # Should have at least one state
        assert len(result["performance"]) >= 1

    def test_missing_detector(self):
        from oxq.tools.observe import observe_performance_by_state

        result = observe_performance_by_state("nonexistent", "nonexistent")
        assert "error" in result


class TestExperimentCreate:
    def test_creates_log(self):
        from oxq.tools.observe import observe_experiment_create

        result = observe_experiment_create(name="test-batch")
        assert "log_id" in result
        assert result["name"] == "test-batch"


class TestExperimentAdd:
    def test_adds_experiment(self):
        from oxq.tools.observe import observe_experiment_add, observe_experiment_create

        log_result = observe_experiment_create(name="test")
        log_id = log_result["log_id"]
        result = observe_experiment_add(
            log_id=log_id,
            name="iter1",
            observation="sharpe drops",
            hypothesis="reduce freq",
            criteria={"sharpe": "above_baseline"},
            result={"best_freq": 63},
            conclusion="rejected",
        )
        assert result["experiment_count"] == 1

    def test_missing_log(self):
        from oxq.tools.observe import observe_experiment_add

        result = observe_experiment_add(
            log_id="nonexistent",
            name="x",
            observation="o",
            hypothesis="h",
            criteria={},
            result={},
            conclusion="confirmed",
        )
        assert "error" in result


class TestExperimentAddFromStrategy:
    def test_extracts_from_strategy(self):
        from oxq.core.strategy import Strategy
        from oxq.tools.observe import (
            observe_experiment_add_from_strategy,
            observe_experiment_create,
        )
        from oxq.universe.static import StaticUniverse

        # Setup strategy in session
        strat = Strategy(
            name="test_strat",
            universe=StaticUniverse(("AAPL",)),
            indicators={},
            signals={},
            entry_rules=[],
            exit_rules=[],
            hypothesis="test hypothesis",
            objectives={"sharpe_ratio": {"min": 1.0}},
        )
        session._strategies["test_strat"] = strat
        run_id = _setup_run_result()

        log_result = observe_experiment_create()
        log_id = log_result["log_id"]
        result = observe_experiment_add_from_strategy(
            log_id=log_id,
            strategy="test_strat",
            run_id=run_id,
            observation="testing",
            conclusion="confirmed",
        )
        assert "experiment" in result
        assert "total_return" in result["experiment"]["result"]


class TestExperimentList:
    def test_lists_experiments(self):
        from oxq.tools.observe import (
            observe_experiment_add,
            observe_experiment_create,
            observe_experiment_list,
        )

        log_result = observe_experiment_create()
        log_id = log_result["log_id"]
        observe_experiment_add(
            log_id=log_id,
            name="e1",
            observation="o",
            hypothesis="h",
            criteria={},
            result={},
            conclusion="confirmed",
        )
        result = observe_experiment_list(log_id)
        assert "experiments" in result
        assert len(result["experiments"]) == 1
        assert "markdown_table" in result
