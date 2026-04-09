from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from oxq.core.errors import SymbolNotFoundError
from oxq.data.loaders import resolve_data_dir

logger = logging.getLogger(__name__)


class LocalMarketDataProvider:
    """Read market data from local Parquet files. Implements MarketDataProvider Protocol."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = resolve_data_dir(data_dir)

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        path = self._data_dir / f"{symbol}.parquet"
        if not path.exists():
            msg = f"No data for '{symbol}'. Run downloader first."
            raise SymbolNotFoundError(msg)
        df = pd.read_parquet(path)
        if hasattr(df.index, "tz") and df.index.tz is None:
            logger.warning(
                "Parquet file for '%s' has no timezone on index. "
                "Assuming UTC. Re-download data to fix.",
                symbol,
            )
            df.index = df.index.tz_localize("UTC")
        return df.loc[start:end]  # type: ignore[misc]  # pandas string-based label slicing

    def get_latest(self, symbol: str) -> pd.Series:
        path = self._data_dir / f"{symbol}.parquet"
        if not path.exists():
            msg = f"No data for '{symbol}'. Run downloader first."
            raise SymbolNotFoundError(msg)
        df = pd.read_parquet(path)
        return df.iloc[-1]
