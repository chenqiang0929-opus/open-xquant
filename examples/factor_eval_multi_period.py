"""Multi-period momentum factor evaluation: 10d / 30d / 60d on CN vs US.

Usage: uv run python examples/factor_eval_multi_period.py
"""

from __future__ import annotations

import warnings
from math import isnan
from pathlib import Path

import numpy as np
import pandas as pd

from oxq.core.registry import list_indicators
from oxq.data.loaders import resolve_data_dir
from oxq.factor_eval import compute_ic, compute_icir, compute_rank_ic

# ── Config ──────────────────────────────────────────────────────────────────

START = "2022-01-01"
END = "2025-03-20"
FORWARD_DAYS = 5
PERIODS = [10, 30, 60]

SP500_TOP = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ADBE", "CRM",
    "ORCL", "AMD", "INTC", "CSCO", "QCOM", "TXN", "AMAT", "MU", "LRCX", "KLAC",
    "MRVL", "SNPS", "CDNS", "PANW", "NOW", "INTU", "PLTR", "CRWD", "ABNB",
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "C",
    "BLK", "SCHW", "CB", "PGR", "CME", "ICE", "AON", "MCO", "SPGI",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN",
    "BMY", "GILD", "ISRG", "MDT", "SYK", "VRTX", "REGN", "ZTS", "BSX", "EW",
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT", "LOW",
    "HD", "TJX", "ORLY", "ROST", "DG", "DLTR", "YUM", "CMG", "MNST", "CL",
    "XOM", "CVX", "COP", "SLB", "EOG", "LIN", "APD", "SHW", "ECL", "FCX",
    "CAT", "DE", "HON", "UPS", "RTX", "LMT", "GE", "BA", "MMM", "EMR",
    "NEE", "DUK", "SO", "AEP", "D", "T", "VZ", "TMUS", "AMT", "PLD",
    "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "ARE", "AVB", "EQR",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def get_csi300_tickers() -> list[str]:
    import akshare as ak
    df = ak.index_stock_cons_csindex(symbol="000300")
    tickers = []
    for _, row in df.iterrows():
        code = row["成分券代码"]
        exchange = row["交易所"]
        suffix = ".SS" if "上海" in exchange else ".SZ"
        tickers.append(f"{code}{suffix}")
    return tickers


def load_prices(tickers: list[str]) -> pd.DataFrame:
    """Load close prices for all available tickers."""
    data_dir = resolve_data_dir()
    series: dict[str, pd.Series] = {}
    for ticker in tickers:
        parquet = data_dir / f"{ticker}.parquet"
        if not parquet.exists():
            continue
        df = pd.read_parquet(parquet).loc[START:END]
        if len(df) < 80:  # need enough data for 60-day momentum
            continue
        series[ticker] = df["close"]
    return pd.DataFrame(series)


def compute_momentum(prices_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Compute log momentum: (ln(P_t) - ln(P_{t-N})) / N."""
    log_prices = np.log(prices_df)
    return log_prices.diff(period) / period


def evaluate(factor_df: pd.DataFrame, prices_df: pd.DataFrame) -> dict:
    """Evaluate a factor, return IC/ICIR/RankIC."""
    fwd_returns = prices_df.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ic = compute_ic(factor_df, fwd_returns)
        rank_ic = compute_rank_ic(factor_df, fwd_returns)
        icir = compute_icir(float(ic["mean"]), float(ic["std"]))
    return {
        "ic": float(ic["mean"]),
        "icir": float(icir),
        "rank_ic": float(rank_ic["mean"]),
    }


def fmt(v: float) -> str:
    if isnan(v):
        return "    N/A"
    return f"{v:+.4f}"


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    # Load data
    print("Loading CSI 300 ...")
    cn_tickers = get_csi300_tickers()
    cn_prices = load_prices(cn_tickers)
    print(f"  {cn_prices.shape[1]} stocks x {cn_prices.shape[0]} days")

    print("Loading S&P 500 top ...")
    us_prices = load_prices(SP500_TOP)
    print(f"  {us_prices.shape[1]} stocks x {us_prices.shape[0]} days")

    # Evaluate each period
    results: list[dict] = []
    for period in PERIODS:
        print(f"\nEvaluating Momentum({period}) ...")
        cn_mom = compute_momentum(cn_prices, period)
        us_mom = compute_momentum(us_prices, period)

        cn_eval = evaluate(cn_mom, cn_prices)
        us_eval = evaluate(us_mom, us_prices)
        results.append({"period": period, "cn": cn_eval, "us": us_eval})

    # Print report
    print()
    print("=" * 72)
    print("多周期动量因子对比 — Multi-Period Momentum: A股 vs 美股")
    print("=" * 72)
    print(f"区间: {START} ~ {END}")
    print(f"前看收益: {FORWARD_DAYS} 天")
    print(f"A股: 沪深300 ({cn_prices.shape[1]} stocks) | 美股: S&P 500 Top ({us_prices.shape[1]} stocks)")
    print()

    # IC table
    print(f"  {'':>20}  {'──── A股 ────':>26}  {'──── 美股 ────':>26}")
    print(f"  {'Momentum Period':>20}  {'IC':>8} {'ICIR':>8} {'RankIC':>8}  {'IC':>8} {'ICIR':>8} {'RankIC':>8}")
    print(f"  {'-' * 70}")

    for r in results:
        p = r["period"]
        cn = r["cn"]
        us = r["us"]
        print(
            f"  {p:>17}天  "
            f"{fmt(cn['ic']):>8} {fmt(cn['icir']):>8} {fmt(cn['rank_ic']):>8}  "
            f"{fmt(us['ic']):>8} {fmt(us['icir']):>8} {fmt(us['rank_ic']):>8}"
        )

    # Interpretation
    print()
    print("解读:")

    for r in results:
        p = r["period"]
        cn_ic = r["cn"]["ic"]
        us_ic = r["us"]["ic"]

        cn_dir = "反转" if cn_ic < 0 else "动量"
        us_dir = "反转" if us_ic < 0 else "动量"
        cn_strength = "强" if abs(cn_ic) > 0.03 else "弱"
        us_strength = "强" if abs(us_ic) > 0.03 else "弱"

        print(f"  {p}天: A股={cn_dir}({cn_strength}, IC={fmt(cn_ic)}), 美股={us_dir}({us_strength}, IC={fmt(us_ic)})")

    # Overall conclusion
    print()
    cn_trend = [r["cn"]["ic"] for r in results]
    us_trend = [r["us"]["ic"] for r in results]

    cn_improving = cn_trend[-1] > cn_trend[0]
    us_improving = us_trend[-1] > us_trend[0]

    print("趋势:")
    if cn_improving:
        print(f"  A股: 随周期增长，动量效应增强 ({fmt(cn_trend[0])} → {fmt(cn_trend[-1])})")
    else:
        print(f"  A股: 随周期增长，反转效应增强 ({fmt(cn_trend[0])} → {fmt(cn_trend[-1])})")

    if us_improving:
        print(f"  美股: 随周期增长，动量效应增强 ({fmt(us_trend[0])} → {fmt(us_trend[-1])})")
    else:
        print(f"  美股: 随周期增长，反转效应增强 ({fmt(us_trend[0])} → {fmt(us_trend[-1])})")

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
