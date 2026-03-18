"""SMA Crossover Strategy — complete end-to-end example.

Hypothesis:
    当 SMA(10) 从下方穿越 SMA(50)（金叉）时，表明短期动量转强，
    买入后持有至 SMA(10) 回落到 SMA(50) 以下（死叉）时卖出，
    该策略在趋势行情中能获得正超额收益。

Objectives:
    - 总收益率 ≥ 5%
    - 夏普比率 ≥ 0.5
    - 最大回撤 ≥ -20%

Portfolio Optimization:
    Uses EqualWeightOptimizer — when a golden cross signal fires,
    the optimizer allocates equal weight to all signaled symbols.

Exit Rule:
    - SMA10 跌破 SMA50 时全仓卖出

Architecture note:
    Strategy defines: name, universe, signals, portfolio (PortfolioOptimizer).
    Indicators are collected automatically from signals' required_indicators.
    Rules (e.g. ExitRule) are passed separately to Engine.run(rules=[...]).

    "Backtest" = LocalMarketDataProvider + SimBroker
    "Paper"    = RealtimeDataProvider   + SimBroker     (future)
    "Live"     = RealtimeDataProvider   + BrokerAdapter  (future)

Usage:
    # 1. Download data
    python -c "from oxq.data import YFinanceDownloader; \
        YFinanceDownloader().download('AAPL', '2023-01-01', '2024-12-31')"

    # 2. Run strategy
    python examples/strategies/sma_crossover.py
"""

from oxq.core import Engine, Strategy
from oxq.data import LocalMarketDataProvider
from oxq.indicators import SMA
from oxq.portfolio.optimizers import EqualWeightOptimizer
from oxq.rules import ExitRule
from oxq.signals import Crossover
from oxq.trade import SimBroker
from oxq.universe import StaticUniverse

# ── 0. Constraints ──────────────────────────────────────────────────

SYMBOL = "AAPL"
INITIAL_CASH = 100_000.0
START = "2023-01-01"
END = "2024-12-31"

# ── 1. Build signal with required_indicators ─────────────────────────

crossover_signal = Crossover()
crossover_signal.required_indicators = {
    "sma_10": (SMA(), {"column": "close", "period": 10}),
    "sma_50": (SMA(), {"column": "close", "period": 50}),
}

# ── 2. Strategy definition ───────────────────────────────────────────

strategy = Strategy(
    name="sma_crossover",
    hypothesis=(
        "当 SMA(10) 从下方穿越 SMA(50) 时买入，"
        "SMA(10) 回落到 SMA(50) 以下时卖出，"
        "该策略在趋势行情中能获得正超额收益"
    ),
    objectives={
        "total_return": {"min": 0.05},
        "sharpe_ratio": {"min": 0.5},
        "max_drawdown": {"max": -0.20},
    },
    benchmarks=["SPY"],
    universe=StaticUniverse((SYMBOL,)),
    signals={
        "golden_cross": (crossover_signal, {"fast": "sma_10", "slow": "sma_50"}),
    },
    portfolio=EqualWeightOptimizer(),
)

# ── 3. Rules (passed to Engine, not Strategy) ────────────────────────

rules = [ExitRule(fast="sma_10", slow="sma_50")]

# ── 4. Run ────────────────────────────────────────────────────────────

broker = SimBroker()
result = Engine().run(
    strategy,
    market=LocalMarketDataProvider(),
    broker=broker,
    start=START,
    end=END,
    initial_cash=INITIAL_CASH,
    rules=rules,
)

# ── 5. Results ────────────────────────────────────────────────────────

print("=" * 72)
print("SMA Crossover Strategy")
print(f"Symbol: {SYMBOL}  |  Period: {START} ~ {END}  |  Init Cash: {INITIAL_CASH:,.0f}")
print("=" * 72)

rows = [
    ("Total Return", f"{result.total_return():.2%}"),
    ("Sharpe Ratio", f"{result.sharpe_ratio():.2f}"),
    ("Max Drawdown", f"{result.max_drawdown():.2%}"),
    ("Total Trades", f"{len(result.trades)}"),
    ("Final Cash", f"{result.portfolio.cash:,.0f}"),
    ("Total Value", f"{result.equity_curve[-1][1]:,.0f}"),
]

for name, val in rows:
    print(f"  {name:>16}: {val}")

# ── 6. Objectives check ──────────────────────────────────────────────

print()
objectives = strategy.objectives
metrics = {
    "total_return": result.total_return(),
    "sharpe_ratio": result.sharpe_ratio(),
    "max_drawdown": result.max_drawdown(),
}
for metric_name, bounds in objectives.items():
    actual = metrics[metric_name]
    passed = True
    if "min" in bounds:
        passed = passed and actual >= bounds["min"]
    if "max" in bounds:
        passed = passed and actual <= bounds["max"]
    tag = "pass" if passed else "FAIL"
    print(f"  {metric_name:<16} {tag}  ({actual:.4f})")

# ── 7. Trade log ─────────────────────────────────────────────────────

if result.trades:
    print(f"\nTrades ({len(result.trades)}):")
    print(f"  {'Date':<28} {'Side':>4}  {'Shares':>6} {'Symbol':<6} {'Price':>10}")
    print("  " + "-" * 58)
    for fill in result.trades:
        print(
            f"  {fill.filled_at:<28} {fill.order.side:>4}  "
            f"{fill.order.shares:>6} {fill.order.symbol:<6} "
            f"{fill.filled_price:>10.2f}"
        )
