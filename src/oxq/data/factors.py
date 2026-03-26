"""Factor data: download from various sources and read locally."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from oxq.core.errors import DownloadError

# Human-readable name → World Bank indicator code
MACRO_INDICATOR_MAP: dict[str, str] = {
    "gdp": "NY.GDP.MKTP.CD",  # GDP (current USD)
    "gdp_per_capita": "NY.GDP.PCAP.CD",  # GDP per capita (current USD)
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",  # GDP growth (annual %)
    "cpi": "FP.CPI.TOTL.ZG",  # CPI inflation (annual %)
}

# Backward-compatible alias
INDICATOR_MAP = MACRO_INDICATOR_MAP

FINANCIAL_INDICATORS: list[str] = [
    "total_shares",
    "eps",
    "book_value_per_share",
    "net_income",
    "operating_cash_flow",
    "total_assets",
    "revenue",
    "roe",
]


def resolve_factor_dir(dest_dir: Path | None = None, sub: str | None = None) -> Path:
    """Resolve factor data directory.

    Priority: parameter > $OXQ_DATA_DIR/factor > ~/.oxq/data/factor.

    Parameters
    ----------
    dest_dir : Path | None
        Explicit base directory.
    sub : str | None
        Subdirectory name (e.g. "macro", "financial"). Appended when provided.
    """
    if dest_dir is not None:
        base = dest_dir
    else:
        env = os.environ.get("OXQ_DATA_DIR")
        if env:
            base = Path(env) / "factor"
        else:
            base = Path.home() / ".oxq" / "data" / "factor"

    if sub is not None:
        return base / sub
    return base


def _fetch_world_bank(
    indicator_code: str,
    countries: list[str],
    start_year: int,
    end_year: int,
    timeout: int = 60,
    retries: int = 3,
) -> list[dict[str, Any]]:
    """Fetch data from World Bank API v2. Returns raw JSON records."""
    import time

    country_str = ";".join(countries)
    url = (
        f"https://api.worldbank.org/v2/country/{country_str}"
        f"/indicator/{indicator_code}"
        f"?date={start_year}:{end_year}&format=json&per_page=10000"
    )
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
                body = json.loads(resp.read().decode())
            # World Bank returns [metadata, data] — data is the second element
            if not isinstance(body, list) or len(body) < 2 or body[1] is None:
                return []
            result: list[dict[str, Any]] = body[1]
            return result
        except (TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert World Bank JSON records to a DataFrame (index=year, cols=countries)."""
    rows: dict[int, dict[str, float | None]] = {}
    for rec in records:
        year = int(rec["date"])
        country = rec["countryiso3code"]
        value = rec["value"]
        if year not in rows:
            rows[year] = {}
        rows[year][country] = float(value) if value is not None else None

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "year"
    df = df.sort_index()
    # Reorder columns alphabetically for consistency
    df = df.reindex(sorted(df.columns), axis=1)
    return df


class WorldBankFetcher:
    """Fetch macro indicators from World Bank Open Data API.

    Implements the ``FactorFetcher`` protocol.
    """

    def fetch(
        self,
        target: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch indicator data from World Bank.

        Parameters
        ----------
        target : str
            Human-readable indicator name (e.g. "gdp", "cpi").
        start : str
            Start year (inclusive), e.g. "2020".
        end : str
            End year (inclusive), e.g. "2024".
        **kwargs
            countries : list[str] — ISO 3166-1 alpha-3 codes (required).

        Returns
        -------
        pd.DataFrame
            DataFrame with index=year (int), columns=country codes.
        """
        if target not in MACRO_INDICATOR_MAP:
            msg = (
                f"Unknown indicator '{target}'. "
                f"Available: {sorted(MACRO_INDICATOR_MAP)}"
            )
            raise ValueError(msg)

        countries: list[str] = kwargs.get("countries", [])
        indicator_code = MACRO_INDICATOR_MAP[target]

        try:
            records = _fetch_world_bank(
                indicator_code, countries, int(start), int(end)
            )
        except Exception as exc:
            msg = f"Failed to download '{target}' from World Bank: {exc}"
            raise DownloadError(msg) from exc

        if not records:
            msg = (
                f"No data returned for '{target}' "
                f"(countries={countries}, {start}-{end})."
            )
            raise DownloadError(msg)

        return _records_to_dataframe(records)

    def list_indicators(self) -> list[str]:
        """Return sorted list of available indicator names."""
        return sorted(MACRO_INDICATOR_MAP)


# Backward-compatible alias
WorldBankDownloader = WorldBankFetcher


class FactorDownloader:
    """Download factor data via a FactorFetcher and persist locally.

    Parameters
    ----------
    fetcher : FactorFetcher
        Data source to fetch from.
    sub : str
        Subdirectory name (e.g. "macro", "financial").
    """

    def __init__(self, fetcher: Any, sub: str) -> None:
        self.fetcher = fetcher
        self.sub = sub

    def download(
        self,
        target: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
        **kwargs: Any,
    ) -> Path:
        """Fetch data and save as parquet.

        If a file already exists, merges new data with existing
        (concat + drop_duplicates on index keeping last + sort_index).

        Returns
        -------
        Path
            Path to the saved parquet file.
        """
        df = self.fetcher.fetch(target, start, end, **kwargs)

        factor_dir = resolve_factor_dir(dest_dir, sub=self.sub)
        factor_dir.mkdir(parents=True, exist_ok=True)
        path = factor_dir / f"{target}.parquet"

        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

        df.to_parquet(path)
        return path

    def download_many(
        self,
        targets: list[str],
        start: str,
        end: str,
        dest_dir: Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Path]:
        """Download multiple targets. Returns {target: path}."""
        result: dict[str, Path] = {}
        for target in targets:
            result[target] = self.download(target, start, end, dest_dir=dest_dir, **kwargs)
        return result

    def list_available(self) -> list[str]:
        """Delegate to fetcher's list_indicators."""
        return self.fetcher.list_indicators()


def read_factor(
    target: str,
    countries: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    data_dir: Path | None = None,
    sub: str = "macro",
    indicators: list[str] | None = None,
    point_in_time: bool = False,
) -> pd.DataFrame:
    """Read local factor data.

    Parameters
    ----------
    target : str
        Factor name (e.g. "gdp").
    countries : list[str] | None
        Filter to these countries. None returns all available.
    start_year : int | None
        Filter start year (inclusive).
    end_year : int | None
        Filter end year (inclusive).
    data_dir : Path | None
        Override factor data directory.
    sub : str
        Subdirectory name (default "macro").
    indicators : list[str] | None
        Filter to these columns. Metadata columns (publish_date, period)
        are kept automatically when present.
    point_in_time : bool
        When True, filter by publish_date instead of report_date/index.

    Returns
    -------
    pd.DataFrame
        DataFrame with index=year (int), columns=country codes.
    """
    factor_dir = resolve_factor_dir(data_dir, sub=sub)
    path = factor_dir / f"{target}.parquet"
    if not path.exists():
        msg = f"Factor file not found: {path}"
        raise FileNotFoundError(msg)

    df = pd.read_parquet(path)

    if indicators is not None:
        metadata_cols = ["publish_date", "period"]
        keep = [c for c in indicators if c in df.columns]
        keep += [c for c in metadata_cols if c in df.columns and c not in keep]
        df = df[keep]

    if point_in_time and "publish_date" in df.columns:
        if start_year is not None:
            df = df[df["publish_date"].dt.year >= start_year]
        if end_year is not None:
            df = df[df["publish_date"].dt.year <= end_year]
    else:
        if start_year is not None:
            df = df[df.index >= start_year]
        if end_year is not None:
            df = df[df.index <= end_year]

    if countries is not None:
        available = [c for c in countries if c in df.columns]
        df = df[available]

    return df
