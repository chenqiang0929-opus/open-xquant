"""Tests for RunResult performance metrics."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult


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


# -- annualized_return --------------------------------------------------------

def test_annualized_return_basic() -> None:
    values = np.linspace(100, 110, 252).tolist()
    result = _make_result(values)
    arr = np.array(values)
    log_ret = np.diff(np.log(arr))
    expected = float(np.mean(log_ret) * 252)
    assert result.annualized_return() == pytest.approx(expected, rel=1e-4)


def test_annualized_return_empty() -> None:
    result = _make_result([])
    assert result.annualized_return() == 0.0


def test_annualized_return_single_point() -> None:
    result = _make_result([100.0])
    assert result.annualized_return() == 0.0


# -- annualized_volatility ----------------------------------------------------

def test_annualized_volatility_basic() -> None:
    values = [100.0, 102.0, 99.0, 103.0, 101.0, 104.0]
    result = _make_result(values)
    arr = np.array(values)
    log_ret = np.diff(np.log(arr))
    expected = float(np.std(log_ret, ddof=1) * np.sqrt(252))
    assert result.annualized_volatility() == pytest.approx(expected, rel=1e-6)


def test_annualized_volatility_empty() -> None:
    result = _make_result([])
    assert result.annualized_volatility() == 0.0


def test_annualized_volatility_constant() -> None:
    result = _make_result([100.0] * 10)
    assert result.annualized_volatility() == 0.0


# -- calmar_ratio --------------------------------------------------------------

def test_calmar_ratio_basic() -> None:
    values = [100.0, 110.0, 105.0, 115.0, 120.0]
    result = _make_result(values)
    ann_ret = result.annualized_return()
    mdd = result.max_drawdown()
    expected = ann_ret / abs(mdd)
    assert result.calmar_ratio() == pytest.approx(expected, rel=1e-6)


def test_calmar_ratio_no_drawdown() -> None:
    result = _make_result([100.0, 110.0, 120.0, 130.0])
    assert result.calmar_ratio() == 0.0


def test_calmar_ratio_empty() -> None:
    result = _make_result([])
    assert result.calmar_ratio() == 0.0


# -- sortino_ratio -------------------------------------------------------------

def test_sortino_ratio_basic() -> None:
    values = [100.0, 102.0, 99.0, 103.0, 97.0, 105.0]
    result = _make_result(values)
    arr = np.array(values)
    log_ret = np.diff(np.log(arr))
    downside = log_ret[log_ret < 0]
    downside_dev = float(np.sqrt(np.mean(downside**2)) * np.sqrt(252))
    ann_ret = float(np.mean(log_ret) * 252)
    expected = ann_ret / downside_dev
    assert result.sortino_ratio() == pytest.approx(expected, rel=1e-4)


def test_sortino_ratio_no_downside() -> None:
    result = _make_result([100.0, 110.0, 120.0, 130.0])
    assert result.sortino_ratio() == 0.0


def test_sortino_ratio_empty() -> None:
    result = _make_result([])
    assert result.sortino_ratio() == 0.0
