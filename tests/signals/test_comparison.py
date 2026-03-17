"""Tests for Comparison signal."""

import pandas as pd

from oxq.signals.comparison import Comparison


def _mktdata() -> dict[str, pd.DataFrame]:
    idx = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame(
        {"sma_fast": [10, 20, 30, 25, 15], "sma_slow": [15, 15, 15, 15, 15]},
        index=idx,
    )
    return {"AAPL": df}


def test_comparison_gt():
    result = Comparison().compute(_mktdata(), left="sma_fast", right="sma_slow", relationship="gt")
    assert list(result["AAPL"]) == [False, True, True, True, False]


def test_comparison_lt():
    result = Comparison().compute(_mktdata(), left="sma_fast", right="sma_slow", relationship="lt")
    assert list(result["AAPL"]) == [True, False, False, False, False]


def test_comparison_eq():
    result = Comparison().compute(_mktdata(), left="sma_fast", right="sma_slow", relationship="eq")
    assert list(result["AAPL"]) == [False, False, False, False, True]


def test_comparison_ne():
    result = Comparison().compute(_mktdata(), left="sma_fast", right="sma_slow", relationship="ne")
    assert list(result["AAPL"]) == [True, True, True, True, False]


def test_comparison_multi_symbol():
    idx = pd.bdate_range("2024-01-01", periods=2)
    mktdata = {
        "A": pd.DataFrame({"x": [10, 20], "y": [15, 15]}, index=idx),
        "B": pd.DataFrame({"x": [20, 10], "y": [15, 15]}, index=idx),
    }
    result = Comparison().compute(mktdata, left="x", right="y", relationship="gt")
    assert list(result["A"]) == [False, True]
    assert list(result["B"]) == [True, False]


def test_comparison_has_name():
    assert Comparison().name == "Comparison"
