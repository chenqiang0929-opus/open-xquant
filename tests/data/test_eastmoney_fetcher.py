"""Tests for EastMoneyFetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oxq.data.factors import EastMoneyFetcher
from oxq.data.providers import FactorFetcher


def _make_yjbb_df():
    """Fake stock_yjbb_em."""
    return pd.DataFrame({
        "股票代码": ["600519", "600519"],
        "报告期": ["2024-06-30", "2024-03-31"],
        "公告日期": ["2024-08-24", "2024-04-26"],
        "基本每股收益": [29.42, 16.16],
        "营业收入": [8.69e10, 4.59e10],
        "净利润": [4.16e10, 2.28e10],
        "净资产收益率": [16.42, 8.87],
        "总股本": [1.256e9, 1.256e9],
    })


def _make_zcfz_df():
    """Fake stock_zcfz_em."""
    return pd.DataFrame({
        "股票代码": ["600519", "600519"],
        "报告期": ["2024-06-30", "2024-03-31"],
        "公告日期": ["2024-08-24", "2024-04-26"],
        "资产总计": [2.52e11, 2.45e11],
        "每股净资产": [179.83, 174.53],
    })


def _make_xjll_df():
    """Fake stock_xjll_em."""
    return pd.DataFrame({
        "股票代码": ["600519", "600519"],
        "报告期": ["2024-06-30", "2024-03-31"],
        "公告日期": ["2024-08-24", "2024-04-26"],
        "经营活动产生的现金流量净额": [2.13e10, 7.45e9],
    })


class TestEastMoneyFetcher:
    def test_satisfies_protocol(self):
        fetcher = EastMoneyFetcher()
        assert isinstance(fetcher, FactorFetcher)

    def test_list_indicators(self):
        fetcher = EastMoneyFetcher()
        indicators = fetcher.list_indicators()
        assert len(indicators) == 8
        assert indicators == sorted(indicators)
        expected = [
            "book_value_per_share",
            "eps",
            "net_income",
            "operating_cash_flow",
            "revenue",
            "roe",
            "total_assets",
            "total_shares",
        ]
        assert indicators == expected

    @patch("oxq.data.factors.akshare", create=True)
    def test_fetch_all_indicators(self, mock_ak):
        mock_ak.stock_yjbb_em = MagicMock(return_value=_make_yjbb_df())
        mock_ak.stock_zcfz_em = MagicMock(return_value=_make_zcfz_df())
        mock_ak.stock_xjll_em = MagicMock(return_value=_make_xjll_df())

        fetcher = EastMoneyFetcher()
        df = fetcher.fetch("600519", "2024-01-01", "2024-12-31")

        assert df.index.name == "report_date"
        assert len(df) == 2
        # All 8 indicator columns
        for col in [
            "eps", "revenue", "net_income", "roe", "total_shares",
            "total_assets", "book_value_per_share", "operating_cash_flow",
        ]:
            assert col in df.columns
        assert "publish_date" in df.columns
        assert "period" in df.columns
        # Spot-check values (first row by sorted date is 2024-03-31)
        row = df.loc["2024-06-30"]
        assert row["eps"] == pytest.approx(29.42)
        assert row["total_assets"] == pytest.approx(2.52e11)
        assert row["operating_cash_flow"] == pytest.approx(2.13e10)

    @patch("oxq.data.factors.akshare", create=True)
    def test_fetch_filters_by_period_annual(self, mock_ak):
        # Add an annual row (month=12)
        yjbb = _make_yjbb_df()
        annual_row = pd.DataFrame({
            "股票代码": ["600519"],
            "报告期": ["2024-12-31"],
            "公告日期": ["2025-03-28"],
            "基本每股收益": [58.84],
            "营业收入": [1.74e11],
            "净利润": [8.32e10],
            "净资产收益率": [32.84],
            "总股本": [1.256e9],
        })
        yjbb = pd.concat([yjbb, annual_row], ignore_index=True)

        zcfz = _make_zcfz_df()
        zcfz_annual = pd.DataFrame({
            "股票代码": ["600519"],
            "报告期": ["2024-12-31"],
            "公告日期": ["2025-03-28"],
            "资产总计": [2.60e11],
            "每股净资产": [185.0],
        })
        zcfz = pd.concat([zcfz, zcfz_annual], ignore_index=True)

        xjll = _make_xjll_df()
        xjll_annual = pd.DataFrame({
            "股票代码": ["600519"],
            "报告期": ["2024-12-31"],
            "公告日期": ["2025-03-28"],
            "经营活动产生的现金流量净额": [4.26e10],
        })
        xjll = pd.concat([xjll, xjll_annual], ignore_index=True)

        mock_ak.stock_yjbb_em = MagicMock(return_value=yjbb)
        mock_ak.stock_zcfz_em = MagicMock(return_value=zcfz)
        mock_ak.stock_xjll_em = MagicMock(return_value=xjll)

        fetcher = EastMoneyFetcher()
        df = fetcher.fetch("600519", "2024-01-01", "2025-12-31", period="annual")

        assert len(df) == 1
        assert all(df["period"] == "annual")

    @patch("oxq.data.factors.akshare", create=True)
    def test_fetch_specific_indicators(self, mock_ak):
        mock_ak.stock_yjbb_em = MagicMock(return_value=_make_yjbb_df())
        mock_ak.stock_zcfz_em = MagicMock(return_value=_make_zcfz_df())
        mock_ak.stock_xjll_em = MagicMock(return_value=_make_xjll_df())

        fetcher = EastMoneyFetcher()
        df = fetcher.fetch(
            "600519", "2024-01-01", "2024-12-31", indicators=["eps", "revenue"]
        )

        assert "eps" in df.columns
        assert "revenue" in df.columns
        # Only yjbb should be called
        mock_ak.stock_yjbb_em.assert_called_once()
        mock_ak.stock_zcfz_em.assert_not_called()
        mock_ak.stock_xjll_em.assert_not_called()
