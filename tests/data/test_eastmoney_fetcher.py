"""Tests for EastMoneyFetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oxq.data.factors import EASTMONEY_FIELD_MAP, EastMoneyFetcher
from oxq.data.providers import FactorFetcher


def _make_abstract_df() -> pd.DataFrame:
    """Fake stock_financial_abstract response (pivot table: metrics × periods)."""
    return pd.DataFrame({
        "选项": [
            "常用指标", "常用指标", "常用指标", "常用指标",
            "常用指标", "常用指标",
        ],
        "指标": [
            "基本每股收益", "营业总收入", "净利润",
            "净资产收益率(ROE)", "每股净资产", "经营现金流量净额",
        ],
        "20240630": [29.42, 8.69e10, 4.16e10, 16.42, 179.83, 2.13e10],
        "20240331": [16.16, 4.59e10, 2.28e10, 8.87, 174.53, 7.45e9],
    })


def _make_analysis_df() -> pd.DataFrame:
    """Fake stock_financial_analysis_indicator response (flat table)."""
    return pd.DataFrame({
        "日期": [pd.Timestamp("2024-03-31").date(), pd.Timestamp("2024-06-30").date()],
        "总资产(元)": [2.45e11, 2.52e11],
        "每股净资产_调整前(元)": [174.53, 179.83],
        "股东权益比率(%)": [87.0, 86.5],
    })


class TestEastMoneyFetcher:
    def test_satisfies_protocol(self) -> None:
        fetcher = EastMoneyFetcher()
        assert isinstance(fetcher, FactorFetcher)

    def test_list_indicators(self) -> None:
        fetcher = EastMoneyFetcher()
        indicators = fetcher.list_indicators()
        assert len(indicators) == 8
        assert indicators == sorted(indicators)
        assert set(indicators) == set(EASTMONEY_FIELD_MAP)

    @patch("oxq.data.factors.akshare", create=True)
    def test_fetch_all_indicators(self, mock_ak: MagicMock) -> None:
        mock_ak.stock_financial_abstract.return_value = _make_abstract_df()
        mock_ak.stock_financial_analysis_indicator.return_value = _make_analysis_df()

        fetcher = EastMoneyFetcher()
        df = fetcher.fetch("600519", "2024-01-01", "2024-12-31")

        assert df.index.name == "report_date"
        assert len(df) == 2
        # Check key indicator columns
        for col in ["eps", "revenue", "net_income", "roe", "book_value_per_share",
                     "operating_cash_flow", "total_assets"]:
            assert col in df.columns, f"Missing column: {col}"
        assert "period" in df.columns

        # Spot-check values
        row = df.loc[pd.Timestamp("2024-06-30")]
        assert row["eps"] == pytest.approx(29.42)
        assert row["revenue"] == pytest.approx(8.69e10)
        assert row["total_assets"] == pytest.approx(2.52e11)
        assert row["operating_cash_flow"] == pytest.approx(2.13e10)

    @patch("oxq.data.factors.akshare", create=True)
    def test_fetch_filters_by_period_annual(self, mock_ak: MagicMock) -> None:
        abstract = _make_abstract_df()
        # Add an annual column (12-31)
        abstract["20241231"] = [58.84, 1.74e11, 8.32e10, 32.84, 185.0, 4.26e10]

        analysis = _make_analysis_df()
        annual_row = pd.DataFrame({
            "日期": [pd.Timestamp("2024-12-31").date()],
            "总资产(元)": [2.60e11],
            "每股净资产_调整前(元)": [185.0],
            "股东权益比率(%)": [85.0],
        })
        analysis = pd.concat([analysis, annual_row], ignore_index=True)

        mock_ak.stock_financial_abstract.return_value = abstract
        mock_ak.stock_financial_analysis_indicator.return_value = analysis

        fetcher = EastMoneyFetcher()
        df = fetcher.fetch("600519", "2024-01-01", "2025-12-31", period="annual")

        assert len(df) == 1
        assert all(df["period"] == "annual")

    @patch("oxq.data.factors.akshare", create=True)
    def test_fetch_specific_indicators_abstract_only(self, mock_ak: MagicMock) -> None:
        """Requesting only abstract-sourced indicators should not call analysis."""
        mock_ak.stock_financial_abstract.return_value = _make_abstract_df()

        fetcher = EastMoneyFetcher()
        df = fetcher.fetch(
            "600519", "2024-01-01", "2024-12-31", indicators=["eps", "revenue"]
        )

        assert "eps" in df.columns
        assert "revenue" in df.columns
        # analysis function should not be called
        mock_ak.stock_financial_analysis_indicator.assert_not_called()
