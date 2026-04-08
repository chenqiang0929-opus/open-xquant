"""Data tools — list, inspect, and load market/factor data."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oxq.data.factors import MACRO_INDICATOR_MAP as INDICATOR_MAP
from oxq.data.factors import resolve_factor_dir
from oxq.data.loaders import Downloader, resolve_data_dir
from oxq.tools.registry import registry


@registry.tool(
    name="data_list_symbols",
    description="List locally available market data symbols",
)
def list_symbols(data_dir: str | None = None) -> dict[str, Any]:
    """List locally available symbols."""
    path = resolve_data_dir(Path(data_dir) if data_dir else None)
    if not path.exists():
        return {"symbols": [], "count": 0, "data_dir": str(path)}
    symbols = sorted(p.stem for p in path.glob("*.parquet"))
    return {"symbols": symbols, "count": len(symbols), "data_dir": str(path)}


@registry.tool(
    name="data_inspect",
    description="Inspect data summary for a symbol (rows, date range, missing values)",
)
def inspect_symbol(symbol: str, data_dir: str | None = None) -> dict[str, Any]:
    """Inspect a symbol's local data."""
    path = resolve_data_dir(Path(data_dir) if data_dir else None)
    parquet_path = path / f"{symbol}.parquet"
    if not parquet_path.exists():
        return {"symbol": symbol, "error": f"No data for '{symbol}'. Run data_load_symbols first."}
    df = pd.read_parquet(parquet_path)
    return {
        "symbol": symbol,
        "rows": len(df),
        "columns": list(df.columns),
        "date_range": [str(df.index[0].date()), str(df.index[-1].date())],
        "missing_values": int(df.isna().sum().sum()),
        "sample_head": df.head(3).reset_index().astype(str).to_dict(orient="records"),
    }


@registry.tool(
    name="data_generate_mock",
    description=(
        "Generate synthetic OHLCV market data for the given symbols using a "
        "geometric Brownian motion model. Writes parquet files into the data "
        "dir, exactly where data_load_symbols would put real downloads. Use "
        "when real market data is unavailable (e.g. yfinance not installed, "
        "geo-blocked, or rate-limited) and the goal is to verify a strategy "
        "pipeline rather than its actual returns. Deterministic given the seed."
    ),
)
def data_generate_mock(
    symbols: list[str],
    start: str,
    end: str,
    seed: int = 42,
    annualized_drift: float = 0.05,
    annualized_vol: float = 0.20,
    initial_price: float = 100.0,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Generate synthetic OHLCV parquet files for *symbols* via GBM.

    Each symbol gets its own deterministic path derived from
    ``seed + (hash(symbol) & 0xffff)`` so the same call produces the same
    data on every machine, but different symbols have different paths.

    Output schema matches data_load_symbols / YFinanceDownloader exactly:
    DatetimeIndex named 'date', columns ['open', 'high', 'low', 'close',
    'volume'] with volume as int64. Daily, business-day frequency.
    """
    dest = Path(data_dir) if data_dir else None
    out_dir = resolve_data_dir(dest)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range(start=start, end=end, name="date")
    n = len(dates)
    if n == 0:
        return {
            "symbols": [],
            "rows": {},
            "data_dir": str(out_dir),
            "errors": {sym: "empty date range" for sym in symbols},
        }

    mu_daily = annualized_drift / 252.0
    sigma_daily = annualized_vol / math.sqrt(252.0)

    # Plan 036A: Generate per-symbol DataFrames in pass 1, compute the
    # synthetic market series (cross-sectional mean of close prices) in
    # the middle, then write out parquet files in pass 2 with the market
    # column attached. Cross-asset indicators (CAPM Beta, market-neutral
    # spreads, relative strength) can read mktdata['market'] directly
    # without needing a separate data source.
    per_symbol: dict[str, pd.DataFrame] = {}
    rows: dict[str, int] = {}
    errors: dict[str, str] = {}
    for sym in symbols:
        try:
            sym_seed = (seed + (hash(sym) & 0xFFFF)) & 0xFFFFFFFF
            rng = np.random.default_rng(sym_seed)

            # Daily log-returns ~ N(mu_daily - 0.5*sigma^2, sigma_daily)
            # (the -0.5*sigma^2 drift correction makes E[exp(r)] = exp(mu).)
            log_rets = rng.normal(
                loc=mu_daily - 0.5 * sigma_daily**2,
                scale=sigma_daily,
                size=n,
            )
            close = initial_price * np.exp(np.cumsum(log_rets))

            # Open: close[t-1] perturbed by a small overnight jitter, with
            # close[0]'s open seeded from initial_price.
            overnight = rng.normal(loc=0.0, scale=sigma_daily * 0.5, size=n)
            open_ = np.empty(n, dtype=float)
            open_[0] = initial_price * float(np.exp(overnight[0]))
            open_[1:] = close[:-1] * np.exp(overnight[1:])

            # Intraday range jitter: build high/low to envelope (open, close).
            range_jitter = np.abs(
                rng.normal(loc=0.0, scale=sigma_daily * 0.7, size=n)
            )
            body_max = np.maximum(open_, close)
            body_min = np.minimum(open_, close)
            high = body_max * np.exp(range_jitter)
            low = body_min * np.exp(-range_jitter)

            # Volume: lognormal around a fixed mean. int64 per oxq schema.
            vol_log = rng.normal(loc=14.0, scale=0.4, size=n)
            volume = np.exp(vol_log).astype("int64")

            per_symbol[sym] = pd.DataFrame(
                {
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                },
                index=dates,
            )
        except Exception as exc:  # noqa: BLE001
            errors[sym] = str(exc)

    # Plan 036A: synthetic market series — cross-sectional mean of all
    # successfully generated symbols' close prices. Single shared series,
    # same index as the per-symbol price columns. Identical across every
    # symbol so cross-asset indicators see one consistent benchmark.
    if per_symbol:
        market_series = (
            pd.concat(
                [df["close"].rename(sym) for sym, df in per_symbol.items()],
                axis=1,
            ).mean(axis=1)
        )
    else:
        market_series = pd.Series(dtype=float, index=dates)

    for sym, df in per_symbol.items():
        try:
            df = df.copy()
            df["market"] = market_series
            path = out_dir / f"{sym}.parquet"
            df.to_parquet(path)
            rows[sym] = len(df)
        except Exception as exc:  # noqa: BLE001
            errors[sym] = str(exc)

    result: dict[str, Any] = {
        "symbols": list(rows.keys()),
        "rows": rows,
        "data_dir": str(out_dir),
    }
    if errors:
        result["errors"] = errors
    return result


@registry.tool(
    name="data_load_symbols",
    description="Download market data for given symbols",
)
def load_symbols(
    symbols: list[str],
    start: str,
    end: str,
    source: str = "yfinance",
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Download symbols from an external source."""
    dest = Path(data_dir) if data_dir else None

    dl: Downloader
    if source == "yfinance":
        from oxq.data import YFinanceDownloader

        dl = YFinanceDownloader()
    elif source == "akshare":
        from oxq.data import AkShareDownloader

        dl = AkShareDownloader()
    else:
        return {"error": f"Unknown source '{source}'. Use 'yfinance' or 'akshare'."}

    rows: dict[str, int] = {}
    errors: dict[str, str] = {}
    for sym in symbols:
        try:
            dl.download(sym, start, end, dest_dir=dest)
            df = pd.read_parquet(resolve_data_dir(dest) / f"{sym}.parquet")
            rows[sym] = len(df)
        except Exception as e:
            errors[sym] = str(e)

    result: dict[str, Any] = {
        "symbols": list(rows.keys()),
        "rows": rows,
        "data_dir": str(resolve_data_dir(dest)),
    }
    if errors:
        result["errors"] = errors
    return result


# ---------------------------------------------------------------------------
# Factor tools
# ---------------------------------------------------------------------------


@registry.tool(
    name="factor_download",
    description="Download a macro indicator (gdp, gdp_per_capita, gdp_growth, cpi) from World Bank",
)
def factor_download(
    indicator: str,
    countries: list[str],
    start_year: int = 2000,
    end_year: int = 2024,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Download a macro indicator from World Bank and save locally."""
    from oxq.data.factors import FactorDownloader, WorldBankFetcher

    dest = Path(data_dir) if data_dir else None
    dl = FactorDownloader(WorldBankFetcher(), sub="macro")
    try:
        path = dl.download(
            indicator, str(start_year), str(end_year),
            dest_dir=dest, countries=countries,
        )
    except (ValueError, Exception) as exc:
        return {"error": str(exc)}

    df = pd.read_parquet(path)
    return {
        "indicator": indicator,
        "countries": list(df.columns),
        "year_range": [int(df.index.min()), int(df.index.max())],
        "rows": len(df),
        "path": str(path),
    }


@registry.tool(
    name="factor_list",
    description="List locally available factor files",
)
def factor_list(data_dir: str | None = None) -> dict[str, Any]:
    """List locally available factor data files."""
    dest = Path(data_dir) if data_dir else None
    path = resolve_factor_dir(dest, sub="macro")
    if not path.exists():
        return {"factors": [], "count": 0, "data_dir": str(path)}
    factors = sorted(p.stem for p in path.glob("*.parquet"))
    return {"factors": factors, "count": len(factors), "data_dir": str(path)}


@registry.tool(
    name="factor_inspect",
    description="Inspect a factor file (year range, countries, sample values)",
)
def factor_inspect(
    indicator: str,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Inspect a locally stored factor file."""
    dest = Path(data_dir) if data_dir else None
    path = resolve_factor_dir(dest, sub="macro")
    parquet_path = path / f"{indicator}.parquet"
    if not parquet_path.exists():
        return {
            "indicator": indicator,
            "error": f"No data for '{indicator}'. Run factor_download first.",
            "available_indicators": sorted(INDICATOR_MAP),
        }
    df = pd.read_parquet(parquet_path)
    return {
        "indicator": indicator,
        "countries": list(df.columns),
        "year_range": [int(df.index.min()), int(df.index.max())],
        "rows": len(df),
        "missing_values": int(df.isna().sum().sum()),
        "sample": df.tail(3).reset_index().astype(str).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Financial tools
# ---------------------------------------------------------------------------


@registry.tool(
    name="financial_download",
    description="Download financial statement data (eps, roe, revenue, etc.) for a stock",
)
def financial_download(
    symbol: str,
    start: str,
    end: str,
    source: str = "eastmoney",  # "eastmoney" | "yfinance"
    indicators: list[str] | None = None,
    period: str = "quarterly",
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Download financial data for a stock symbol."""
    from oxq.data.factors import (
        EastMoneyFetcher,
        FactorDownloader,
        YFinanceFinancialFetcher,
    )

    dest = Path(data_dir) if data_dir else None

    if source == "eastmoney":
        fetcher = EastMoneyFetcher()
    elif source == "yfinance":
        fetcher = YFinanceFinancialFetcher()
    else:
        return {"error": f"Unknown source '{source}'. Use 'eastmoney' or 'yfinance'."}

    dl = FactorDownloader(fetcher, sub="financial")
    try:
        path = dl.download(
            symbol, start, end, dest_dir=dest,
            indicators=indicators, period=period,
        )
    except Exception as exc:
        return {"error": str(exc)}

    df = pd.read_parquet(path)
    indicator_cols = [c for c in df.columns if c not in ("publish_date", "period")]
    return {
        "symbol": symbol,
        "indicators": indicator_cols,
        "rows": len(df),
        "date_range": [str(df.index.min().date()), str(df.index.max().date())] if len(df) > 0 else [],
        "path": str(path),
    }


@registry.tool(
    name="financial_list",
    description="List locally available financial data files",
)
def financial_list(data_dir: str | None = None) -> dict[str, Any]:
    """List locally available financial data files."""
    from oxq.data.factors import resolve_factor_dir

    dest = Path(data_dir) if data_dir else None
    path = resolve_factor_dir(dest, sub="financial")
    if not path.exists():
        return {"symbols": [], "count": 0, "data_dir": str(path)}
    symbols = sorted(p.stem for p in path.glob("*.parquet"))
    return {"symbols": symbols, "count": len(symbols), "data_dir": str(path)}


@registry.tool(
    name="financial_inspect",
    description="Inspect financial data for a symbol (date range, indicators, sample)",
)
def financial_inspect(
    symbol: str,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Inspect a symbol's financial data."""
    from oxq.data.factors import resolve_factor_dir

    dest = Path(data_dir) if data_dir else None
    path = resolve_factor_dir(dest, sub="financial")
    parquet_path = path / f"{symbol}.parquet"
    if not parquet_path.exists():
        return {
            "symbol": symbol,
            "error": f"No financial data for '{symbol}'. Run financial_download first.",
        }
    df = pd.read_parquet(parquet_path)
    indicator_cols = [c for c in df.columns if c not in ("publish_date", "period")]
    return {
        "symbol": symbol,
        "rows": len(df),
        "indicators": indicator_cols,
        "date_range": [str(df.index.min().date()), str(df.index.max().date())] if len(df) > 0 else [],
        "periods": sorted(df["period"].unique().tolist()) if "period" in df.columns else [],
        "missing_values": int(df[indicator_cols].isna().sum().sum()),
        "sample": df.tail(3).reset_index().astype(str).to_dict(orient="records"),
    }
