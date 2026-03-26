"""Tests for time-series factor evaluation tool."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oxq.tools.factor_eval_ts import factor_evaluate_ts


@pytest.fixture()
def data_dir(tmp_path: Path) -> str:
    """Create sample parquet files for 2 symbols."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    for sym in ["AAPL", "GOOG"]:
        prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.01))
        df = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
                "volume": np.random.randint(1000, 10000, 300).astype(float),
            },
            index=dates,
        )
        df.to_parquet(tmp_path / f"{sym}.parquet")
    return str(tmp_path)


class TestFactorEvaluateTs:

    def test_basic_evaluation(self, data_dir: str) -> None:
        result = factor_evaluate_ts(
            indicator="SMA",
            params={"column": "close", "period": 10},
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            data_dir=data_dir,
            forward_periods=[1, 5],
            t1_offset=False,
        )
        assert "error" not in result
        assert "metrics" in result
        assert "hit_rate" in result["metrics"]
        assert "decay_curve" in result["metrics"]
        assert "profit_loss" in result["metrics"]
        assert "cash_period" in result["metrics"]

    def test_returns_charts(self, data_dir: str) -> None:
        result = factor_evaluate_ts(
            indicator="SMA",
            params={"column": "close", "period": 10},
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            data_dir=data_dir,
            forward_periods=[1, 5],
            t1_offset=False,
        )
        assert "charts" in result
        for path in result["charts"].values():
            assert os.path.exists(path)

    def test_multi_asset_includes_comparison(self, data_dir: str) -> None:
        result = factor_evaluate_ts(
            indicator="SMA",
            params={"column": "close", "period": 10},
            symbols=["AAPL", "GOOG"],
            start="2023-01-01",
            end="2023-12-31",
            data_dir=data_dir,
            forward_periods=[1, 5],
            t1_offset=False,
        )
        assert result["metrics"]["comparison"] is not None
        assert result["metrics"]["comparison"]["skipped"] is False

    def test_single_asset_skips_comparison(self, data_dir: str) -> None:
        result = factor_evaluate_ts(
            indicator="SMA",
            params={"column": "close", "period": 10},
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            data_dir=data_dir,
            forward_periods=[1, 5],
            t1_offset=False,
        )
        assert (
            result["metrics"]["comparison"] is None
            or result["metrics"]["comparison"]["skipped"] is True
        )

    def test_unknown_indicator_returns_error(self, data_dir: str) -> None:
        result = factor_evaluate_ts(
            indicator="NonExistent",
            params={},
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            data_dir=data_dir,
        )
        assert "error" in result

    def test_missing_data_returns_error(self, data_dir: str) -> None:
        result = factor_evaluate_ts(
            indicator="SMA",
            params={"column": "close", "period": 10},
            symbols=["NONEXIST"],
            start="2023-01-01",
            end="2023-12-31",
            data_dir=data_dir,
        )
        assert "error" in result

    def test_market_state_method(self, data_dir: str) -> None:
        result = factor_evaluate_ts(
            indicator="SMA",
            params={"column": "close", "period": 10},
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            data_dir=data_dir,
            forward_periods=[1, 5],
            t1_offset=False,
            market_state_method="sma",
        )
        assert result["metrics"]["conditional"] is not None
        assert result["metrics"]["conditional"]["skipped"] is False

    def test_config_in_result(self, data_dir: str) -> None:
        result = factor_evaluate_ts(
            indicator="SMA",
            params={"column": "close", "period": 10},
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            data_dir=data_dir,
            forward_periods=[1, 5],
            t1_offset=False,
            signal_threshold=0.5,
        )
        assert result["config"]["signal_threshold"] == 0.5
        assert result["config"]["t1_offset"] is False
