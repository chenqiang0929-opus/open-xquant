from __future__ import annotations

import importlib
import os
from pathlib import Path

import pandas as pd

from oxq.core.errors import DownloadError
from oxq.data.providers import Downloader

__all__ = ["AkShareDownloader", "Downloader", "YFinanceDownloader", "resolve_data_dir"]


def resolve_data_dir(dest_dir: Path | None = None) -> Path:
    """Resolve data storage directory. Priority: parameter > OXQ_DATA_DIR > default."""
    if dest_dir is not None:
        return dest_dir
    env = os.environ.get("OXQ_DATA_DIR")
    if env:
        return Path(env) / "market"
    return Path.home() / ".oxq" / "data" / "market"


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw API DataFrame to standard schema."""
    df = df.rename(columns=str.lower)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df = df.tz_localize(None)  # type: ignore[arg-type]
    df = df.rename_axis("date")
    cols = ["open", "high", "low", "close", "volume"]
    df = df[cols]
    df["volume"] = df["volume"].astype("int64")
    return df


class YFinanceDownloader:
    """Download market data via yfinance. Covers US and global equities."""

    def __init__(self, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> Path:
        yfinance = globals().get("yfinance") or importlib.import_module("yfinance")

        data_dir = resolve_data_dir(dest_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        ticker = yfinance.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=self.auto_adjust)
        if df.empty:
            msg = f"No data returned for '{symbol}' ({start} to {end})."
            raise DownloadError(msg)

        df = _normalize_df(df)
        path = data_dir / f"{symbol}.parquet"
        df.to_parquet(path)
        return path

    def download_many(
        self,
        symbols: list[str],
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> dict[str, Path]:
        return {s: self.download(s, start, end, dest_dir) for s in symbols}


class AkShareDownloader:
    """Download A-share market data via akshare."""

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> Path:
        akshare = globals().get("akshare") or importlib.import_module("akshare")

        data_dir = resolve_data_dir(dest_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        df = akshare.stock_zh_a_hist(
            symbol=symbol,
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        if df.empty:
            msg = f"No data returned for '{symbol}' ({start} to {end})."
            raise DownloadError(msg)

        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df[["open", "high", "low", "close", "volume"]]
        df["volume"] = df["volume"].astype("int64")

        path = data_dir / f"{symbol}.parquet"
        df.to_parquet(path)
        return path

    def download_many(
        self,
        symbols: list[str],
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> dict[str, Path]:
        return {s: self.download(s, start, end, dest_dir) for s in symbols}
