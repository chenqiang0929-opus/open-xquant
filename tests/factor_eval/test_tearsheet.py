"""Tests for tear sheet generation."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from oxq.factor_eval.bundle import create_bundle
from oxq.factor_eval.tearsheet import generate_tearsheet


class TestGenerateTearsheet:
    """Test tearsheet orchestration and output structure."""

    @pytest.fixture()
    def bundle(self):
        """A FactorBundle with 50 factor points and 120 price points."""
        np.random.seed(42)
        factor_dates = pd.date_range("2024-01-01", periods=50, freq="B")
        price_dates = pd.date_range("2024-01-01", periods=120, freq="B")

        factor_values = pd.Series(
            np.random.randn(50),
            index=pd.MultiIndex.from_arrays(
                [factor_dates, ["a"] * 50], names=["date", "asset"],
            ),
        )
        prices = pd.DataFrame(
            {"a": 100 * np.exp(np.cumsum(np.random.randn(120) * 0.01))},
            index=price_dates,
        )
        return create_bundle(factor_values, prices, forward_periods=[1, 5, 10, 20])

    def test_returns_summary_dict(self, bundle) -> None:
        result = generate_tearsheet(bundle, forward_periods=[1, 5])
        assert "summary" in result
        assert "alignment_report" in result["summary"]
        assert "lookahead_bias" in result["summary"]
        assert "hit_rate" in result["summary"]
        assert "decay_curve" in result["summary"]

    def test_returns_chart_paths(self, bundle) -> None:
        result = generate_tearsheet(bundle, forward_periods=[1, 5])
        assert "charts" in result
        assert "rolling_hit_rate" in result["charts"]
        assert "decay_curve" in result["charts"]

    def test_chart_files_exist(self, bundle, tmp_path) -> None:
        result = generate_tearsheet(
            bundle, forward_periods=[1, 5], output_dir=str(tmp_path),
        )
        assert os.path.exists(result["charts"]["rolling_hit_rate"])
        assert os.path.exists(result["charts"]["decay_curve"])

    def test_chart_files_are_png(self, bundle, tmp_path) -> None:
        result = generate_tearsheet(
            bundle, forward_periods=[1, 5], output_dir=str(tmp_path),
        )
        assert result["charts"]["rolling_hit_rate"].endswith(".png")
        assert result["charts"]["decay_curve"].endswith(".png")

    def test_summary_is_serializable(self, bundle) -> None:
        """All summary values should be JSON-serializable (no pd objects)."""
        import json

        result = generate_tearsheet(bundle, forward_periods=[1, 5])
        summary = result["summary"]
        # Remove rolling_hit_rate (Series) before serialization check
        hr = summary["hit_rate"].copy()
        hr.pop("rolling_hit_rate", None)
        json.dumps({"alignment": summary["alignment_report"],
                     "bias": summary["lookahead_bias"],
                     "hit_rate": hr,
                     "decay": summary["decay_curve"]})

    def test_respects_date_range(self, bundle) -> None:
        full = generate_tearsheet(bundle, forward_periods=[1])
        sliced = generate_tearsheet(
            bundle, forward_periods=[1],
            start_date="2024-02-01", end_date="2024-02-28",
        )
        assert sliced["summary"]["hit_rate"]["sample_count"] < full["summary"]["hit_rate"]["sample_count"]
