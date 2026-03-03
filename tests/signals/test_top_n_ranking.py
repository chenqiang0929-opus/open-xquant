"""Tests for TopNRanking signal."""

import pandas as pd
import pytest

from oxq.core.types import Signal
from oxq.signals.top_n_ranking import TopNRanking


def _make_mktdata(scores: dict[str, list[float]]) -> dict[str, pd.DataFrame]:
    n = len(next(iter(scores.values())))
    dates = pd.bdate_range("2024-01-01", periods=n)
    result: dict[str, pd.DataFrame] = {}
    for symbol, vals in scores.items():
        result[symbol] = pd.DataFrame(
            {"close": [100.0] * n, "score": vals}, index=dates,
        )
    return result


def test_top_n_ranking_satisfies_signal_protocol() -> None:
    assert isinstance(TopNRanking(), Signal)


def test_top_n_ranking_basic() -> None:
    mktdata = _make_mktdata({"A": [0.10], "B": [0.05], "C": [0.15]})
    result = TopNRanking().compute(mktdata, score="score", n=2)
    # Top 2: C(0.15), A(0.10). B excluded.
    # Normalize: C=0.15/(0.15+0.10)=0.6, A=0.4
    assert result["C"].iloc[0] == pytest.approx(0.6)
    assert result["A"].iloc[0] == pytest.approx(0.4)
    assert result["B"].iloc[0] == 0.0


def test_top_n_ranking_filter_negative() -> None:
    mktdata = _make_mktdata({"A": [0.10], "B": [-0.05], "C": [0.15]})
    result = TopNRanking().compute(mktdata, score="score", n=5)
    # B filtered out. A=0.10/(0.10+0.15)=0.4, C=0.6
    assert result["B"].iloc[0] == 0.0
    assert result["A"].iloc[0] == pytest.approx(0.4)
    assert result["C"].iloc[0] == pytest.approx(0.6)


def test_top_n_ranking_max_weight_cap_excess_to_cash() -> None:
    mktdata = _make_mktdata({"A": [0.90], "B": [0.10]})
    result = TopNRanking().compute(mktdata, score="score", n=2, max_weight=0.7)
    # Normalized: A=0.9, B=0.1 -> A capped at 0.7, B stays 0.1
    # Excess 0.2 goes to CASH (not redistributed). Sum = 0.8
    assert result["A"].iloc[0] == pytest.approx(0.7)
    assert result["B"].iloc[0] == pytest.approx(0.1)


def test_top_n_ranking_nan_skipped() -> None:
    mktdata = _make_mktdata({"A": [float("nan")], "B": [0.10]})
    result = TopNRanking().compute(mktdata, score="score", n=5)
    assert result["A"].iloc[0] == 0.0
    assert result["B"].iloc[0] == pytest.approx(1.0)


def test_top_n_ranking_all_negative() -> None:
    mktdata = _make_mktdata({"A": [-0.10], "B": [-0.20]})
    result = TopNRanking().compute(mktdata, score="score", n=5)
    assert result["A"].iloc[0] == 0.0
    assert result["B"].iloc[0] == 0.0


def test_top_n_ranking_multi_day() -> None:
    mktdata = _make_mktdata({
        "A": [0.10, 0.20],
        "B": [0.05, 0.30],
        "C": [0.15, 0.10],
    })
    result = TopNRanking().compute(mktdata, score="score", n=2)
    # Day 0: top2 = C(0.15), A(0.10) -> C=0.6, A=0.4
    assert result["C"].iloc[0] == pytest.approx(0.6)
    assert result["A"].iloc[0] == pytest.approx(0.4)
    # Day 1: top2 = B(0.30), A(0.20) -> B=0.6, A=0.4
    assert result["B"].iloc[1] == pytest.approx(0.6)
    assert result["A"].iloc[1] == pytest.approx(0.4)


def test_top_n_ranking_has_name() -> None:
    assert TopNRanking().name == "TopNRanking"
