"""Tests for Timestamp signal."""

import pandas as pd

from oxq.signals.timestamp import Timestamp


def test_month_start():
    # 2024-01-02 (Tue) is first trading day of Jan
    # 2024-02-01 (Thu) is first trading day of Feb
    idx = pd.bdate_range("2024-01-02", "2024-02-15")
    mktdata = {"A": pd.DataFrame({"close": range(len(idx))}, index=idx)}
    result = Timestamp().compute(mktdata, rule="month_start")
    series = result["A"]
    assert series.iloc[0] is True   # Jan 2 — first bar
    feb1_idx = idx.get_loc(pd.Timestamp("2024-02-01"))
    assert series.iloc[feb1_idx] is True  # Feb 1 — month change
    assert series.iloc[1] is False         # Jan 3 — not month start


def test_month_end():
    idx = pd.bdate_range("2024-01-02", "2024-02-15")
    mktdata = {"A": pd.DataFrame({"close": range(len(idx))}, index=idx)}
    result = Timestamp().compute(mktdata, rule="month_end")
    series = result["A"]
    # 2024-01-31 (Wed) is last trading day of Jan
    jan31_idx = idx.get_loc(pd.Timestamp("2024-01-31"))
    assert series.iloc[jan31_idx] is True
    assert series.iloc[0] is False  # Jan 2 — not month end


def test_quarter_start():
    idx = pd.bdate_range("2024-03-01", "2024-04-15")
    mktdata = {"A": pd.DataFrame({"close": range(len(idx))}, index=idx)}
    result = Timestamp().compute(mktdata, rule="quarter_start")
    series = result["A"]
    # 2024-04-01 is first trading day of Q2
    apr1_idx = idx.get_loc(pd.Timestamp("2024-04-01"))
    assert series.iloc[apr1_idx] is True
    # Most days are not quarter start
    assert series.iloc[1] is False


def test_weekday():
    # weekday:0 = Monday
    idx = pd.bdate_range("2024-01-01", periods=10)
    mktdata = {"A": pd.DataFrame({"close": range(10)}, index=idx)}
    result = Timestamp().compute(mktdata, rule="weekday:0")
    mondays = [idx[i].weekday() == 0 for i in range(len(idx))]
    assert list(result["A"]) == mondays


def test_unknown_rule_all_false():
    idx = pd.bdate_range("2024-01-01", periods=3)
    mktdata = {"A": pd.DataFrame({"close": [1, 2, 3]}, index=idx)}
    result = Timestamp().compute(mktdata, rule="unknown_rule")
    assert list(result["A"]) == [False, False, False]


def test_timestamp_multi_symbol():
    idx = pd.bdate_range("2024-01-01", periods=5)
    mktdata = {
        "A": pd.DataFrame({"close": range(5)}, index=idx),
        "B": pd.DataFrame({"close": range(5)}, index=idx),
    }
    result = Timestamp().compute(mktdata, rule="weekday:0")
    assert list(result["A"]) == list(result["B"])


def test_timestamp_has_name():
    assert Timestamp().name == "Timestamp"
