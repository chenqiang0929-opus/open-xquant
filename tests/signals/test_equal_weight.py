"""Tests for EqualWeight signal."""

import pandas as pd
import pytest

from oxq.core.types import Signal
from oxq.signals.equal_weight import EqualWeight


def _make_mktdata(closes: dict[str, list[float]]) -> dict[str, pd.DataFrame]:
    n = len(next(iter(closes.values())))
    dates = pd.bdate_range("2024-01-01", periods=n)
    result: dict[str, pd.DataFrame] = {}
    for symbol, vals in closes.items():
        result[symbol] = pd.DataFrame({"close": vals}, index=dates)
    return result


def test_equal_weight_satisfies_signal_protocol() -> None:
    assert isinstance(EqualWeight(), Signal)


def test_equal_weight_basic() -> None:
    mktdata = _make_mktdata({"A": [100.0], "B": [200.0], "C": [150.0]})
    result = EqualWeight().compute(mktdata)
    for s in ("A", "B", "C"):
        assert result[s].iloc[0] == pytest.approx(1.0 / 3)


def test_equal_weight_nan_skipped() -> None:
    mktdata = _make_mktdata({"A": [100.0], "B": [float("nan")], "C": [150.0]})
    result = EqualWeight().compute(mktdata)
    assert result["A"].iloc[0] == pytest.approx(0.5)
    assert result["B"].iloc[0] == 0.0
    assert result["C"].iloc[0] == pytest.approx(0.5)


def test_equal_weight_max_weight_cap() -> None:
    mktdata = _make_mktdata({"A": [100.0], "B": [200.0]})
    result = EqualWeight().compute(mktdata, max_weight=0.3)
    # 1/2 = 0.5 > 0.3, both capped at 0.3. Excess to cash.
    assert result["A"].iloc[0] == pytest.approx(0.3)
    assert result["B"].iloc[0] == pytest.approx(0.3)


def test_equal_weight_multi_day() -> None:
    mktdata = _make_mktdata({
        "A": [100.0, float("nan")],
        "B": [200.0, 200.0],
        "C": [150.0, 150.0],
    })
    result = EqualWeight().compute(mktdata)
    # Day 0: 3 valid -> 1/3 each
    assert result["A"].iloc[0] == pytest.approx(1.0 / 3)
    # Day 1: A is NaN -> 2 valid -> 1/2 each
    assert result["A"].iloc[1] == 0.0
    assert result["B"].iloc[1] == pytest.approx(0.5)


def test_equal_weight_all_nan() -> None:
    mktdata = _make_mktdata({"A": [float("nan")], "B": [float("nan")]})
    result = EqualWeight().compute(mktdata)
    assert result["A"].iloc[0] == 0.0
    assert result["B"].iloc[0] == 0.0


def test_equal_weight_empty_mktdata() -> None:
    result = EqualWeight().compute({})
    assert result == {}


def test_equal_weight_has_name() -> None:
    assert EqualWeight().name == "EqualWeight"
