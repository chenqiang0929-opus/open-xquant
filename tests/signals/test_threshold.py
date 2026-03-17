"""Tests for Threshold signal."""

import pandas as pd

from oxq.signals.threshold import Threshold


def _mktdata(values: list[float]) -> dict[str, pd.DataFrame]:
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    df = pd.DataFrame({"close": values, "rsi": values}, index=idx)
    return {"AAPL": df}


def test_threshold_gt():
    sig = Threshold()
    result = sig.compute(_mktdata([10, 30, 70, 80, 50]), column="rsi", threshold=60.0, relationship="gt")
    expected = [False, False, True, True, False]
    assert list(result["AAPL"]) == expected


def test_threshold_lt():
    sig = Threshold()
    result = sig.compute(_mktdata([10, 30, 70, 80, 50]), column="rsi", threshold=40.0, relationship="lt")
    expected = [True, True, False, False, False]
    assert list(result["AAPL"]) == expected


def test_threshold_gte():
    sig = Threshold()
    result = sig.compute(_mktdata([10, 60, 70]), column="rsi", threshold=60.0, relationship="gte")
    expected = [False, True, True]
    assert list(result["AAPL"]) == expected


def test_threshold_lte():
    sig = Threshold()
    result = sig.compute(_mktdata([10, 60, 70]), column="rsi", threshold=60.0, relationship="lte")
    expected = [True, True, False]
    assert list(result["AAPL"]) == expected


def test_threshold_multi_symbol():
    idx = pd.bdate_range("2024-01-01", periods=3)
    mktdata = {
        "A": pd.DataFrame({"rsi": [10, 80, 50]}, index=idx),
        "B": pd.DataFrame({"rsi": [90, 20, 60]}, index=idx),
    }
    result = Threshold().compute(mktdata, column="rsi", threshold=50.0, relationship="gt")
    assert list(result["A"]) == [False, True, False]
    assert list(result["B"]) == [True, False, True]


def test_threshold_has_name():
    assert Threshold().name == "Threshold"
