"""Factor evaluation demo: Momentum vs Risk-Adjusted Momentum.

Usage:
    uv run python examples/factor_eval_demo.py cn     # CSI 300 (沪深300)
    uv run python examples/factor_eval_demo.py us     # S&P 500 top ~130
    uv run python examples/factor_eval_demo.py both   # side-by-side comparison
"""

from __future__ import annotations

import sys
import warnings
from math import isnan
from pathlib import Path

import numpy as np
import pandas as pd

from oxq.core.registry import list_indicators
from oxq.data.loaders import resolve_data_dir
from oxq.factor_eval import (
    compute_decay,
    compute_ic,
    compute_icir,
    compute_rank_ic,
    compute_turnover,
)

# ── Config ──────────────────────────────────────────────────────────────────

START = "2022-01-01"
END = "2025-03-20"
FORWARD_DAYS = 5
DECAY_HORIZONS = [1, 5, 10, 20]
MOMENTUM_PERIOD = 20

# S&P 500 top ~130 stocks by market cap, all major sectors
SP500_TOP = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ADBE", "CRM",
    "ORCL", "AMD", "INTC", "CSCO", "QCOM", "TXN", "AMAT", "MU", "LRCX", "KLAC",
    "MRVL", "SNPS", "CDNS", "PANW", "NOW", "INTU", "PLTR", "CRWD", "ABNB",
    # Finance
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "C",
    "BLK", "SCHW", "CB", "MMC", "PGR", "CME", "ICE", "AON", "MCO", "SPGI",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN",
    "BMY", "GILD", "ISRG", "MDT", "SYK", "VRTX", "REGN", "ZTS", "BSX", "EW",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT", "LOW",
    "HD", "TJX", "ORLY", "ROST", "DG", "DLTR", "YUM", "CMG", "MNST", "CL",
    # Industrial / Energy / Materials
    "XOM", "CVX", "COP", "SLB", "EOG", "LIN", "APD", "SHW", "ECL", "FCX",
    "CAT", "DE", "HON", "UPS", "RTX", "LMT", "GE", "BA", "MMM", "EMR",
    # Utilities / Real Estate / Telecom
    "NEE", "DUK", "SO", "AEP", "D", "T", "VZ", "TMUS", "AMT", "PLD",
    "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "ARE", "AVB", "EQR",
]


# ── Data helpers ───────────────────────────────────────────────────────────

def get_csi300_tickers() -> list[str]:
    """Fetch CSI 300 constituent stock codes and convert to yfinance tickers."""
    import akshare as ak

    df = ak.index_stock_cons_csindex(symbol="000300")
    tickers = []
    for _, row in df.iterrows():
        code = row["成分券代码"]
        exchange = row["交易所"]
        suffix = ".SS" if "上海" in exchange else ".SZ"
        tickers.append(f"{code}{suffix}")
    return tickers


def ensure_data(tickers: list[str]) -> Path:
    """Download data via yfinance batch download."""
    import yfinance as yf

    data_dir = resolve_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    missing = [t for t in tickers if not (data_dir / f"{t}.parquet").exists()]
    if not missing:
        return data_dir

    print(f"  Downloading {len(missing)} / {len(tickers)} missing symbols ...")

    chunk_size = 50
    done = 0
    errors = 0
    for i in range(0, len(missing), chunk_size):
        chunk = missing[i : i + chunk_size]
        try:
            df = yf.download(
                chunk, start=START, end=END, auto_adjust=True,
                group_by="ticker", threads=True, progress=False,
            )
            for ticker in chunk:
                try:
                    if len(chunk) == 1:
                        ticker_df = df
                    else:
                        ticker_df = df[ticker]
                    ticker_df = ticker_df.dropna(how="all")
                    if len(ticker_df) < 50:
                        errors += 1
                        continue
                    ticker_df.columns = [c.lower() for c in ticker_df.columns]
                    cols = ["open", "high", "low", "close", "volume"]
                    ticker_df = ticker_df[cols]
                    ticker_df["volume"] = ticker_df["volume"].astype("int64")
                    ticker_df = ticker_df.rename_axis("date")
                    if hasattr(ticker_df.index, "tz") and ticker_df.index.tz is not None:
                        ticker_df = ticker_df.tz_localize(None)
                    ticker_df.to_parquet(data_dir / f"{ticker}.parquet")
                    done += 1
                except Exception:
                    errors += 1
        except Exception:
            errors += len(chunk)

        print(f"    progress: {i + len(chunk)}/{len(missing)} (ok: {done}, err: {errors})")

    print(f"    done: {done} downloaded, {errors} errors")
    return data_dir


# ── Factor computation ─────────────────────────────────────────────────────

def load_and_compute(
    tickers: list[str], data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data, compute Momentum and Risk-Adjusted Momentum."""
    indicators = list_indicators()
    momentum_ind = indicators["Momentum"]()
    vol_ind = indicators["RollingVolatility"]()

    mom_series: dict[str, pd.Series] = {}
    ram_series: dict[str, pd.Series] = {}
    price_series: dict[str, pd.Series] = {}

    for ticker in tickers:
        parquet = data_dir / f"{ticker}.parquet"
        if not parquet.exists():
            continue
        df = pd.read_parquet(parquet).loc[START:END]
        if len(df) < MOMENTUM_PERIOD * 2:
            continue

        mom = momentum_ind.compute(df, column="close", period=MOMENTUM_PERIOD)
        vol = vol_ind.compute(df, column="close", period=MOMENTUM_PERIOD)

        price_series[ticker] = df["close"]
        mom_series[ticker] = mom
        ram_series[ticker] = mom / vol.replace(0, np.nan)

    prices_df = pd.DataFrame(price_series)
    momentum_df = pd.DataFrame(mom_series)
    risk_adj_df = pd.DataFrame(ram_series)

    common_idx = (
        prices_df.index
        .intersection(momentum_df.index)
        .intersection(risk_adj_df.index)
    )
    return prices_df.loc[common_idx], momentum_df.loc[common_idx], risk_adj_df.loc[common_idx]


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_factor(
    name: str, factor_df: pd.DataFrame, prices_df: pd.DataFrame,
) -> dict:
    """Run full evaluation on a factor DataFrame."""
    fwd_returns = prices_df.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ic = compute_ic(factor_df, fwd_returns)
        rank_ic = compute_rank_ic(factor_df, fwd_returns)
        icir = compute_icir(float(ic["mean"]), float(ic["std"]))
        decay = compute_decay(factor_df, prices_df, DECAY_HORIZONS)
        turnover = compute_turnover(factor_df)

    return {
        "name": name,
        "ic_mean": float(ic["mean"]),
        "ic_std": float(ic["std"]),
        "icir": float(icir),
        "rank_ic_mean": float(rank_ic["mean"]),
        "rank_ic_std": float(rank_ic["std"]),
        "decay_horizons": decay["horizons"],
        "decay_ic": [float(v) for v in decay["ic_values"]],
        "turnover": float(turnover),
    }


# ── Report ─────────────────────────────────────────────────────────────────

def fmt(v: float, decimals: int = 4) -> str:
    if isnan(v):
        return "N/A"
    return f"{v:+.{decimals}f}" if v != 0 else f"{v:.{decimals}f}"


def run_market(market: str) -> dict:
    """Run evaluation for a single market, return summary dict."""
    if market == "cn":
        label = "沪深300 (A股)"
        print(f"\n{'─' * 72}")
        print(f"  Market: {label}")
        print(f"{'─' * 72}")
        print("  Fetching CSI 300 constituent list ...")
        tickers = get_csi300_tickers()
    else:
        label = "S&P 500 Top 130 (美股)"
        print(f"\n{'─' * 72}")
        print(f"  Market: {label}")
        print(f"{'─' * 72}")
        tickers = SP500_TOP[:]

    print(f"  Universe: {len(tickers)} tickers")
    data_dir = ensure_data(tickers)
    prices_df, momentum_df, risk_adj_df = load_and_compute(tickers, data_dir)

    n_symbols = prices_df.shape[1]
    n_days = prices_df.shape[0]
    print(f"  Loaded: {n_symbols} symbols x {n_days} days")

    mom_result = evaluate_factor("Momentum(20)", momentum_df, prices_df)
    ram_result = evaluate_factor("RiskAdj-Mom(20)", risk_adj_df, prices_df)

    return {
        "market": market,
        "label": label,
        "n_symbols": n_symbols,
        "n_days": n_days,
        "momentum": mom_result,
        "risk_adj": ram_result,
    }


def print_single_report(info: dict) -> None:
    """Print report for a single market."""
    print()
    print(f"  {'指标':<24} {'Momentum(20)':>16} {'RiskAdj-Mom(20)':>16}")
    print(f"  {'-' * 56}")
    for label, key in [
        ("IC (mean)", "ic_mean"), ("IC (std)", "ic_std"), ("ICIR", "icir"),
        ("RankIC (mean)", "rank_ic_mean"), ("Turnover", "turnover"),
    ]:
        m = info["momentum"]
        r = info["risk_adj"]
        print(f"  {label:<24} {fmt(m[key]):>16} {fmt(r[key]):>16}")

    print()
    print(f"  IC Decay:")
    print(f"  {'Horizon':<24} {'Momentum(20)':>16} {'RiskAdj-Mom(20)':>16}")
    print(f"  {'-' * 56}")
    for i, h in enumerate(info["momentum"]["decay_horizons"]):
        m_ic = info["momentum"]["decay_ic"][i]
        r_ic = info["risk_adj"]["decay_ic"][i]
        print(f"  {h:<24} {fmt(m_ic):>16} {fmt(r_ic):>16}")


def print_comparison(cn: dict, us: dict) -> None:
    """Print side-by-side comparison of two markets."""
    print()
    print("=" * 72)
    print("A股 vs 美股 动量因子对比 — Momentum Factor: CN vs US")
    print("=" * 72)
    print(f"区间: {START} ~ {END} | 前看: {FORWARD_DAYS}天 | 动量周期: {MOMENTUM_PERIOD}天")
    print()

    # Header
    print(f"  {'指标':<20} {'A股 Mom':>10} {'A股 RAM':>10} {'美股 Mom':>10} {'美股 RAM':>10}")
    print(f"  {'-' * 60}")

    for label, key in [
        ("IC (mean)", "ic_mean"),
        ("ICIR", "icir"),
        ("RankIC (mean)", "rank_ic_mean"),
        ("Turnover", "turnover"),
    ]:
        cn_m = fmt(cn["momentum"][key])
        cn_r = fmt(cn["risk_adj"][key])
        us_m = fmt(us["momentum"][key])
        us_r = fmt(us["risk_adj"][key])
        print(f"  {label:<20} {cn_m:>10} {cn_r:>10} {us_m:>10} {us_r:>10}")

    print()
    print(f"  IC Decay:")
    print(f"  {'Horizon':<20} {'A股 Mom':>10} {'A股 RAM':>10} {'美股 Mom':>10} {'美股 RAM':>10}")
    print(f"  {'-' * 60}")
    for i, h in enumerate(cn["momentum"]["decay_horizons"]):
        cn_m = fmt(cn["momentum"]["decay_ic"][i])
        cn_r = fmt(cn["risk_adj"]["decay_ic"][i])
        us_m = fmt(us["momentum"]["decay_ic"][i])
        us_r = fmt(us["risk_adj"]["decay_ic"][i])
        print(f"  {h:<20} {cn_m:>10} {cn_r:>10} {us_m:>10} {us_r:>10}")

    # Interpretation
    print()
    print("结论:")

    cn_ic = cn["momentum"]["ic_mean"]
    us_ic = us["momentum"]["ic_mean"]

    if not isnan(cn_ic) and not isnan(us_ic):
        if us_ic > cn_ic and us_ic > 0 and cn_ic < 0:
            print("  ✓ 动量因子在美股有正向预测力，在A股呈反转效应——你的猜想成立。")
        elif us_ic > cn_ic:
            print("  ✓ 动量因子在美股更有效——你的猜想成立。")
        elif abs(us_ic) < 0.03 and abs(cn_ic) < 0.03:
            print("  △ 两个市场的动量因子都较弱，差异不显著。")
        else:
            print("  ✗ 数据不支持这个猜想——A股动量反而更强。")

        print(f"    A股 Momentum IC = {fmt(cn_ic)}  ({'反转' if cn_ic < 0 else '动量'})")
        print(f"    美股 Momentum IC = {fmt(us_ic)}  ({'反转' if us_ic < 0 else '动量'})")

    cn_ram_ic = cn["risk_adj"]["ic_mean"]
    us_ram_ic = us["risk_adj"]["ic_mean"]
    if not isnan(cn_ram_ic) and not isnan(us_ram_ic):
        better_cn = "改善" if abs(cn_ram_ic) > abs(cn_ic) else "未改善"
        better_us = "改善" if abs(us_ram_ic) > abs(us_ic) else "未改善"
        print(f"    风险调整: A股{better_cn}, 美股{better_us}")

    print()
    print("=" * 72)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    market = sys.argv[1] if len(sys.argv) > 1 else "both"

    if market == "cn":
        info = run_market("cn")
        print_single_report(info)
    elif market == "us":
        info = run_market("us")
        print_single_report(info)
    elif market == "both":
        cn = run_market("cn")
        print_single_report(cn)
        us = run_market("us")
        print_single_report(us)
        print_comparison(cn, us)
    else:
        print(f"Unknown market: {market}. Use 'cn', 'us', or 'both'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
