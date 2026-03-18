"""Momentum Rotation Strategy — risk-adjusted momentum with portfolio optimization.

Hypothesis:
    在纳指100ETF、沪深300ETF、黄金ETF三类资产中，按风险调整后的动量
    （momentum / volatility）排名，定期选择排名靠前的 2 只 ETF 并按
    比例分配权重，可以获得优于等权持有的收益。
    高动量低波动的品种更值得持有，低动量或高波动的品种应该减仓或清仓。

Objectives:
    - 年化收益率 ≥ 5%
    - 夏普比率 ≥ 0.5
    - 最大回撤 ≥ -20%

Pipeline:
    Indicator 层 — Momentum(20), RollingVolatility(20), Ratio(mom/vol)
                   (collected automatically from signal's required_indicators)
    Signal 层   — TopNRanking(score=ram, n=2, max_weight=0.6)
    Portfolio   — EqualWeightOptimizer (handles position sizing)

Architecture note:
    Strategy defines: name, universe, signals, portfolio (PortfolioOptimizer).
    Indicators are collected automatically from signals' required_indicators.
    No more RebalanceRule — portfolio optimization handles rebalancing.

    标的:
      - 513100.SS  纳指100ETF（美股科技）
      - 510300.SS  沪深300ETF（A股大盘）
      - 518880.SS  黄金ETF（贵金属）

Usage:
    # 1. Download data (akshare)
    python -c "
    from oxq.data import AkShareDownloader
    dl = AkShareDownloader()
    for s in ['513100.SS', '510300.SS', '518880.SS']:
        dl.download(s, '2024-11-15', '2026-02-28')
    "

    # 2. Run strategy
    python examples/strategies/momentum_rotation.py
"""

from oxq.core import Engine, Strategy
from oxq.data import LocalMarketDataProvider
from oxq.indicators import Momentum, Ratio, RollingVolatility
from oxq.portfolio.optimizers import EqualWeightOptimizer
from oxq.signals import TopNRanking
from oxq.trade import SimBroker
from oxq.universe import StaticUniverse

# ── 0. Constraints ──────────────────────────────────────────────────

SYMBOLS = ("513100.SS", "510300.SS", "518880.SS")
SYMBOL_NAMES = {
    "513100.SS": "纳指100ETF",
    "510300.SS": "沪深300ETF",
    "518880.SS": "黄金ETF",
}
INITIAL_CASH = 100_000.0
START = "2024-11-15"
END = "2026-02-28"

# ── 1. Build signal with required_indicators ─────────────────────────

ranking_signal = TopNRanking()
ranking_signal.required_indicators = {
    "mom": (Momentum(), {"column": "close", "period": 20}),
    "vol": (RollingVolatility(), {"column": "close", "period": 20}),
    "ram": (Ratio(), {"col_a": "mom", "col_b": "vol"}),
}

# ── 2. Strategy definition ───────────────────────────────────────────

strategy = Strategy(
    name="momentum_rotation",
    hypothesis=(
        "在纳指100ETF、沪深300ETF、黄金ETF中，"
        "按 Momentum(20)/Volatility(20) 风险调整动量排名，"
        "选 Top 2 归一化权重，单只上限 60%，"
        "定期调仓可获得正超额收益"
    ),
    objectives={
        "annualized_return": {"min": 0.05},
        "sharpe_ratio": {"min": 0.5},
        "max_drawdown": {"min": -0.20},
    },
    benchmarks=[],
    universe=StaticUniverse(SYMBOLS),
    signals={
        "tw": (ranking_signal, {"score": "ram", "n": 2, "max_weight": 0.6}),
    },
    portfolio=EqualWeightOptimizer(),
)

# ── 3. Run ────────────────────────────────────────────────────────────

broker = SimBroker()
result = Engine().run(
    strategy,
    market=LocalMarketDataProvider(),
    broker=broker,
    start=START,
    end=END,
    initial_cash=INITIAL_CASH,
)

# ── 4. Results ────────────────────────────────────────────────────────

universe_str = ", ".join(f"{s}({SYMBOL_NAMES[s]})" for s in SYMBOLS)
print("=" * 76)
print("Momentum Rotation Strategy")
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
    ("Final Cash", f"{result.portfolio.cash:,.0f}"),
    ("Total Value", f"{result.equity_curve[-1][1]:,.0f}"),
]

for name, val in rows:
    print(f"  {name:>20}: {val}")

# ── 5. Objectives check ──────────────────────────────────────────────

print()
objectives = strategy.objectives
metric_fns = {
    "annualized_return": lambda r: r.annualized_return(),
    "sharpe_ratio": lambda r: r.sharpe_ratio(),
    "max_drawdown": lambda r: r.max_drawdown(),
}

for metric_name, bounds in objectives.items():
    actual = metric_fns[metric_name](result)
    passed = True
    if "min" in bounds:
        passed = passed and actual >= bounds["min"]
    if "max" in bounds:
        passed = passed and actual <= bounds["max"]
    tag = "pass" if passed else "FAIL"
    print(f"  {metric_name:<20} {tag}  ({actual:.4f})")

# ── 6. Trade log (first 20 trades) ───────────────────────────────────

if result.trades:
    shown = result.trades[:20]
    more = len(result.trades) - len(shown)
    print(f"\nTrades ({len(result.trades)} total, showing first {len(shown)}):")
    print(f"  {'Date':<28} {'Side':>4}  {'Shares':>6} {'Symbol':<12} {'Price':>10}")
    print("  " + "-" * 64)
    for fill in shown:
        sym = fill.order.symbol
        name = SYMBOL_NAMES.get(sym, sym)
        print(
            f"  {fill.filled_at:<28} {fill.order.side:>4}  "
            f"{fill.order.shares:>6} {name:<12} "
            f"{fill.filled_price:>10.4f}"
        )
    if more > 0:
        print(f"  ... and {more} more trades")
