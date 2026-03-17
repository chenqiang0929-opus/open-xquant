"""Tests for Formula signal."""

import pandas as pd

from oxq.signals.formula import Formula


def _mktdata() -> dict[str, pd.DataFrame]:
    idx = pd.bdate_range("2024-01-01", periods=4)
    df = pd.DataFrame(
        {"sma_fast": [10, 20, 30, 25], "sma_slow": [15, 15, 15, 15], "rsi": [40, 70, 80, 30]},
        index=idx,
    )
    return {"AAPL": df}


def test_formula_simple():
    result = Formula().compute(_mktdata(), expr="sma_fast > sma_slow")
    assert list(result["AAPL"]) == [False, True, True, True]


def test_formula_compound_and():
    result = Formula().compute(_mktdata(), expr="sma_fast > sma_slow and rsi > 50")
    assert list(result["AAPL"]) == [False, True, True, False]


def test_formula_compound_or():
    result = Formula().compute(_mktdata(), expr="sma_fast < 15 or rsi > 75")
    assert list(result["AAPL"]) == [True, False, True, False]


def test_formula_multi_symbol():
    idx = pd.bdate_range("2024-01-01", periods=2)
    mktdata = {
        "A": pd.DataFrame({"x": [10, 20], "y": [15, 15]}, index=idx),
        "B": pd.DataFrame({"x": [20, 10], "y": [15, 15]}, index=idx),
    }
    result = Formula().compute(mktdata, expr="x > y")
    assert list(result["A"]) == [False, True]
    assert list(result["B"]) == [True, False]


def test_formula_has_name():
    assert Formula().name == "Formula"
