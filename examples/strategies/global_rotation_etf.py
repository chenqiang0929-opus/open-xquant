"""Global Rotation ETF Strategy — reproduced from xquant production.

Hypothesis:
    基于风险调整动量的全球资产配置策略，在纳指100ETF、沪深300ETF、
    黄金ETF三类资产之间进行轮动。使用归一化权重分配（按风险调整动量
    分数比例），每10个交易日调仓一次。

    Risk-adjusted momentum score = SimpleMomentum(20) / AnnualizedVolatility(20) ^ 0.5

Production baseline (2025-01-01 ~ 2026-03-18):
    Cumulative return: 68.12%
    Annualized return: 57.30%
    Max drawdown: -14.94%
    Sharpe ratio: 2.65
    Volatility: 20.84%
    Calmar ratio: 3.84

Usage:
    python examples/strategies/global_rotation_etf.py
"""

import sys

from oxq.core import Engine, Strategy
from oxq.data import LocalMarketDataProvider
from oxq.indicators import AnnualizedVolatility, PowerRatio, SimpleMomentum
from oxq.portfolio.optimizers import TopNRankingOptimizer
from oxq.rules.constraint import RebalanceFrequencyRule
from oxq.signals import Threshold
from oxq.trade import SimBroker
from oxq.trade.sim_broker import FillPriceMode
from oxq.universe import StaticUniverse

# ── 0. Constants ─────────────────────────────────────────────────────

SYMBOLS = ("513100.SS", "510300.SS", "518880.SS")
SYMBOL_NAMES = {
    "513100.SS": "纳指100ETF",
    "510300.SS": "沪深300ETF",
    "518880.SS": "黄金ETF",
}

DATA_START = "2024-12-01"   # warmup period for indicators
START = "2025-01-01"
END = "2026-03-17"
INITIAL_CASH = 1_000_000.0

# ── 1. Download data ─────────────────────────────────────────────────


def download_data() -> None:
    """Download market data: try YFinance first, fallback to AkShare."""
    provider = LocalMarketDataProvider()
    missing = []
    for symbol in SYMBOLS:
        try:
            bars = provider.get_bars(symbol, DATA_START, END)
            if len(bars) > 0:
                continue
        except Exception:
            pass
        missing.append(symbol)

    if not missing:
        return

    print(f"Downloading data for: {', '.join(missing)}")

    # Try YFinance first
    try:
        from oxq.data import YFinanceDownloader

        dl = YFinanceDownloader()
        for symbol in missing:
            try:
                dl.download(symbol, DATA_START, END)
                print(f"  [yfinance] {symbol} OK")
            except Exception as e:
                print(f"  [yfinance] {symbol} FAILED: {e}")
    except ImportError:
        print("  yfinance not available")

    # Fallback: try AkShare for any still missing
    still_missing = []
    for symbol in missing:
        try:
            bars = provider.get_bars(symbol, DATA_START, END)
            if len(bars) > 0:
                continue
        except Exception:
            pass
        still_missing.append(symbol)

    if still_missing:
        try:
            from oxq.data import AkShareDownloader

            dl = AkShareDownloader()
            for symbol in still_missing:
                try:
                    dl.download(symbol, DATA_START, END)
                    print(f"  [akshare] {symbol} OK")
                except Exception as e:
                    print(f"  [akshare] {symbol} FAILED: {e}")
                    print(f"  ERROR: Cannot download {symbol}. Exiting.")
                    sys.exit(1)
        except ImportError:
            print("  akshare not available. Cannot download data.")
            sys.exit(1)


# ── 2. Build signal with required_indicators ─────────────────────────

active_signal = Threshold()
active_signal.required_indicators = {
    "mom": (SimpleMomentum(), {"column": "close", "period": 20}),
    "vol": (AnnualizedVolatility(), {"column": "close", "period": 20}),
    "ram": (PowerRatio(), {"col_a": "mom", "col_b": "vol", "exponent": 0.5}),
}

# ── 3. Strategy definition ───────────────────────────────────────────

strategy = Strategy(
    name="global_rotation_etf",
    hypothesis=(
        "基于风险调整动量（SimpleMomentum(20) / AnnualizedVolatility(20)^0.5）"
        "的全球资产配置策略，在纳指、沪深300、黄金三类资产间轮动，"
        "归一化权重分配，单只上限90%，每10个交易日调仓"
    ),
    objectives={
        "total_return": {"min": 0.60},
        "annualized_return": {"min": 0.50},
        "sharpe_ratio": {"min": 2.0},
        "max_drawdown": {"min": -0.20},
    },
    benchmarks=list(SYMBOLS),
    universe=StaticUniverse(SYMBOLS),
    signals={
        "active": (active_signal, {"column": "ram", "threshold": 0, "relationship": "gt"}),
    },
    portfolio=TopNRankingOptimizer(score_col="ram", n=5, max_weight=0.9),
)

# ── 4. Run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_data()

    broker = SimBroker(fill_price_mode=FillPriceMode.MID)
    result = Engine().run(
        strategy,
        market=LocalMarketDataProvider(),
        broker=broker,
        start=START,
        end=END,
        initial_cash=INITIAL_CASH,
        lot_size=100,
        cash_annual_return=0.025,
        data_start=DATA_START,
        rules=[RebalanceFrequencyRule(interval_days=10)],
    )

    # ── 5. Results ───────────────────────────────────────────────────

    universe_str = ", ".join(f"{s}({SYMBOL_NAMES[s]})" for s in SYMBOLS)
    print("=" * 76)
    print("Global Rotation ETF Strategy")
    print(f"Universe: {universe_str}")
    print(f"Period: {START} ~ {END}  |  Init Cash: {INITIAL_CASH:,.0f}")
    print("=" * 76)

    rows = [
        ("Total Return", f"{result.total_return():.2%}"),
        ("Ann. Return", f"{result.annualized_return():.2%}"),
        ("Ann. Volatility", f"{result.annualized_volatility():.2%}"),
        ("Sharpe Ratio", f"{result.sharpe_ratio():.2f}"),
        ("Calmar Ratio", f"{result.calmar_ratio():.2f}"),
        ("Sortino Ratio", f"{result.sortino_ratio():.2f}"),
        ("Max Drawdown", f"{result.max_drawdown():.2%}"),
        ("Total Trades", f"{len(result.trades)}"),
        ("Final Cash", f"{result.portfolio.cash:,.2f}"),
        ("Total Value", f"{result.equity_curve[-1][1]:,.2f}"),
    ]

    for name, val in rows:
        print(f"  {name:>20}: {val}")