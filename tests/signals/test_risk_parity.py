"""Tests for RiskParity signal."""

import pandas as pd
import pytest

from oxq.core.types import Signal
from oxq.signals.risk_parity import RiskParity


def _make_mktdata(vols: dict[str, list[float]]) -> dict[str, pd.DataFrame]:
    n = len(next(iter(vols.values())))
    dates = pd.bdate_range("2024-01-01", periods=n)
    result: dict[str, pd.DataFrame] = {}
    for symbol, vals in vols.items():
        result[symbol] = pd.DataFrame(
            {"close": [100.0] * n, "vol": vals}, index=dates,
        )
    return result


def test_risk_parity_satisfies_signal_protocol() -> None:
    assert isinstance(RiskParity(), Signal)


def test_risk_parity_basic() -> None:
    # A: vol=0.10 -> inv=10, B: vol=0.20 -> inv=5. Total=15.
    # A=10/15=0.667, B=5/15=0.333
    mktdata = _make_mktdata({"A": [0.10], "B": [0.20]})
    result = RiskParity().compute(mktdata, vol="vol")
    assert result["A"].iloc[0] == pytest.approx(10.0 / 15)
    assert result["B"].iloc[0] == pytest.approx(5.0 / 15)


def test_risk_parity_three_symbols() -> None:
    # A: 0.10 -> inv=10, B: 0.20 -> inv=5, C: 0.25 -> inv=4. Total=19.
    mktdata = _make_mktdata({"A": [0.10], "B": [0.20], "C": [0.25]})
    result = RiskParity().compute(mktdata, vol="vol")
    assert result["A"].iloc[0] == pytest.approx(10.0 / 19)
    assert result["B"].iloc[0] == pytest.approx(5.0 / 19)
    assert result["C"].iloc[0] == pytest.approx(4.0 / 19)


def test_risk_parity_max_weight_cap() -> None:
    # A: vol=0.10 -> inv=10, B: vol=0.20 -> inv=5. Total=15.
    # A=0.667 > 0.5 -> capped at 0.5. B=0.333 stays. Excess to cash.
    mktdata = _make_mktdata({"A": [0.10], "B": [0.20]})
    result = RiskParity().compute(mktdata, vol="vol", max_weight=0.5)
    assert result["A"].iloc[0] == pytest.approx(0.5)
    assert result["B"].iloc[0] == pytest.approx(5.0 / 15)


def test_risk_parity_nan_skipped() -> None:
    mktdata = _make_mktdata({"A": [0.10], "B": [float("nan")]})
    result = RiskParity().compute(mktdata, vol="vol")
    assert result["A"].iloc[0] == pytest.approx(1.0)
    assert result["B"].iloc[0] == 0.0


def test_risk_parity_zero_vol_skipped() -> None:
    mktdata = _make_mktdata({"A": [0.10], "B": [0.0]})
    result = RiskParity().compute(mktdata, vol="vol")
    assert result["A"].iloc[0] == pytest.approx(1.0)
    assert result["B"].iloc[0] == 0.0


def test_risk_parity_negative_vol_skipped() -> None:
    mktdata = _make_mktdata({"A": [0.10], "B": [-0.05]})
    result = RiskParity().compute(mktdata, vol="vol")
    assert result["A"].iloc[0] == pytest.approx(1.0)
    assert result["B"].iloc[0] == 0.0


def test_risk_parity_all_nan() -> None:
    mktdata = _make_mktdata({"A": [float("nan")], "B": [float("nan")]})
    result = RiskParity().compute(mktdata, vol="vol")
    assert result["A"].iloc[0] == 0.0
    assert result["B"].iloc[0] == 0.0


def test_risk_parity_multi_day() -> None:
    mktdata = _make_mktdata({
        "A": [0.10, 0.20],
        "B": [0.20, 0.10],
    })
    result = RiskParity().compute(mktdata, vol="vol")
    # Day 0: A=10/15, B=5/15
    assert result["A"].iloc[0] == pytest.approx(10.0 / 15)
    assert result["B"].iloc[0] == pytest.approx(5.0 / 15)
    # Day 1: A=5/15, B=10/15 (swapped)
    assert result["A"].iloc[1] == pytest.approx(5.0 / 15)
    assert result["B"].iloc[1] == pytest.approx(10.0 / 15)


def test_risk_parity_single_valid() -> None:
    mktdata = _make_mktdata({"A": [0.10], "B": [float("nan")]})
    result = RiskParity().compute(mktdata, vol="vol", max_weight=0.9)
    assert result["A"].iloc[0] == pytest.approx(0.9)


def test_risk_parity_empty_mktdata() -> None:
    result = RiskParity().compute({}, vol="vol")
    assert result == {}


def test_risk_parity_has_name() -> None:
    assert RiskParity().name == "RiskParity"
