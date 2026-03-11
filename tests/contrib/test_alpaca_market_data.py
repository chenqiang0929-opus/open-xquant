"""Tests for AlpacaMarketDataProvider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from oxq.contrib.alpaca.market_data import AlpacaMarketDataProvider

_BARS_RESPONSE = {
    "bars": {
        "AAPL": [
            {"t": "2024-01-02T05:00:00Z", "o": 150.0, "h": 155.0, "l": 149.0, "c": 153.0, "v": 1000},
            {"t": "2024-01-03T05:00:00Z", "o": 153.0, "h": 156.0, "l": 152.0, "c": 154.0, "v": 1200},
        ],
    },
    "next_page_token": None,
}

_LATEST_RESPONSE = {
    "bars": {
        "AAPL": {"t": "2024-01-04T05:00:00Z", "o": 154.0, "h": 157.0, "l": 153.0, "c": 155.0, "v": 800},
    },
}


class TestGetBars:
    def test_returns_dict_of_dataframes(self):
        provider = AlpacaMarketDataProvider(api_key="k", secret_key="s")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _BARS_RESPONSE
        with patch.object(provider._http, "get", return_value=mock_resp):
            result = provider.get_bars(["AAPL"], "2024-01-02", "2024-01-03")
        assert "AAPL" in result
        assert isinstance(result["AAPL"], pd.DataFrame)
        assert list(result["AAPL"].columns) == ["open", "high", "low", "close", "volume"]
        assert len(result["AAPL"]) == 2

    def test_datetime_index(self):
        provider = AlpacaMarketDataProvider(api_key="k", secret_key="s")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _BARS_RESPONSE
        with patch.object(provider._http, "get", return_value=mock_resp):
            result = provider.get_bars(["AAPL"], "2024-01-02", "2024-01-03")
        assert isinstance(result["AAPL"].index, pd.DatetimeIndex)


class TestGetLatest:
    def test_returns_single_row_dataframe(self):
        provider = AlpacaMarketDataProvider(api_key="k", secret_key="s")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _LATEST_RESPONSE
        with patch.object(provider._http, "get", return_value=mock_resp):
            result = provider.get_latest(["AAPL"])
        assert "AAPL" in result
        assert len(result["AAPL"]) == 1
        assert result["AAPL"].iloc[0]["close"] == 155.0


class TestInit:
    def test_base_url_is_data_api(self):
        provider = AlpacaMarketDataProvider(api_key="k", secret_key="s")
        assert "data.alpaca.markets" in str(provider._http.base_url)

    def test_env_vars_fallback(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "env_key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "env_secret")
        provider = AlpacaMarketDataProvider()
        assert provider._api_key == "env_key"
