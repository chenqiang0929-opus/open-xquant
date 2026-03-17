"""Tests for Composite signal."""

import pandas as pd

from oxq.signals.composite import Composite


def _mktdata() -> dict[str, pd.DataFrame]:
    idx = pd.bdate_range("2024-01-01", periods=4)
    df = pd.DataFrame(
        {
            "sig_a": [True, True, False, True],
            "sig_b": [True, False, False, True],
            "sig_c": [False, False, True, True],
        },
        index=idx,
    )
    return {"AAPL": df}


def test_composite_and():
    result = Composite().compute(_mktdata(), signals=["sig_a", "sig_b"], logic="and")
    assert list(result["AAPL"]) == [True, False, False, True]


def test_composite_or():
    result = Composite().compute(_mktdata(), signals=["sig_a", "sig_b"], logic="or")
    assert list(result["AAPL"]) == [True, True, False, True]


def test_composite_three_signals_and():
    result = Composite().compute(_mktdata(), signals=["sig_a", "sig_b", "sig_c"], logic="and")
    assert list(result["AAPL"]) == [False, False, False, True]


def test_composite_three_signals_or():
    result = Composite().compute(_mktdata(), signals=["sig_a", "sig_b", "sig_c"], logic="or")
    assert list(result["AAPL"]) == [True, True, True, True]


def test_composite_single_signal():
    result = Composite().compute(_mktdata(), signals=["sig_a"], logic="and")
    assert list(result["AAPL"]) == [True, True, False, True]


def test_composite_empty_signals():
    result = Composite().compute(_mktdata(), signals=[], logic="and")
    assert result == {}


def test_composite_multi_symbol():
    idx = pd.bdate_range("2024-01-01", periods=2)
    mktdata = {
        "A": pd.DataFrame({"x": [True, False], "y": [True, True]}, index=idx),
        "B": pd.DataFrame({"x": [False, True], "y": [True, False]}, index=idx),
    }
    result = Composite().compute(mktdata, signals=["x", "y"], logic="and")
    assert list(result["A"]) == [True, False]
    assert list(result["B"]) == [False, False]


def test_composite_has_name():
    assert Composite().name == "Composite"
