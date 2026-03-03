"""SMA Crossover Strategy — complete end-to-end example.

Hypothesis:
    当 SMA(10) 从下方穿越 SMA(50)（金叉）时，表明短期动量转强，
    买入后持有至 SMA(10) 回落到 SMA(50) 以下（死叉）时卖出，
    该策略在趋势行情中能获得正超额收益。

Objectives:
    - 总收益率 ≥ 5%
    - 夏普比率 ≥ 0.5
    - 最大回撤 ≥ -20%

Entry Rules (三种买入规则对比):
    - EntryRule:              固定买入 100 股
    - TargetValueEntryRule:   按目标市值 80,000 买入
    - FullPositionEntryRule:  全仓买入（用全部可用现金）

Exit Rule:
    - SMA10 跌破 SMA50 时全仓卖出

Architecture note:
    Strategy 定义与执行环境解耦。Engine 只依赖三个 Protocol 接口：
    MarketDataProvider, OrderRouter, FillReceiver。

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
from oxq.rules import EntryRule, ExitRule, FullPositionEntryRule, TargetValueEntryRule
from oxq.signals import Crossover
from oxq.trade import SimBroker
from oxq.universe import StaticUniverse

# ── 0. Constraints ──────────────────────────────────────────────────

SYMBOL = "AAPL"
INITIAL_CASH = 100_000.0
START = "2023-01-01"
END = "2024-12-31"

# ── 1. Common pipeline components ───────────────────────────────────

COMMON = dict(
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
    indicators={
        "sma_10": (SMA(), {"column": "close", "period": 10}),
        "sma_50": (SMA(), {"column": "close", "period": 50}),
    },
    signals={
        "golden_cross": (Crossover(), {"fast": "sma_10", "slow": "sma_50"}),
    },
    exit_rules=[ExitRule(fast="sma_10", slow="sma_50")],
)

# ── 2. Three entry rule variants ────────────────────────────────────

STRATEGIES = {
    "固定100股": Strategy(
        name="fixed_shares",
        entry_rules=[EntryRule(signal="golden_cross", shares=100)],
        **COMMON,
    ),
    "目标市值8万": Strategy(
        name="target_value",
        entry_rules=[TargetValueEntryRule(signal="golden_cross", target_value=80_000)],
        **COMMON,
    ),
    "全仓买入": Strategy(
        name="full_position",
        entry_rules=[FullPositionEntryRule(signal="golden_cross")],
        **COMMON,
    ),
}

# ── 3. Run all variants ─────────────────────────────────────────────

results = {}
for label, strategy in STRATEGIES.items():
    broker = SimBroker()
    result = Engine().run(
        strategy,
        market=LocalMarketDataProvider(),
        router=broker,
        receiver=broker,
        start=START,
        end=END,
        initial_cash=INITIAL_CASH,
    )
    results[label] = result

# ── 4. Comparison table ─────────────────────────────────────────────

print("=" * 72)
print("SMA Crossover Strategy — Entry Rule Comparison")
print(f"Symbol: {SYMBOL}  |  Period: {START} ~ {END}  |  Init Cash: {INITIAL_CASH:,.0f}")
print("=" * 72)

header = f"{'':>16}" + "".join(f"{label:>18}" for label in results)
print(header)
print("-" * len(header))

rows = [
    ("Total Return", lambda r: f"{r.total_return():.2%}"),
    ("Sharpe Ratio", lambda r: f"{r.sharpe_ratio():.2f}"),
    ("Max Drawdown", lambda r: f"{r.max_drawdown():.2%}"),
    ("Total Trades", lambda r: f"{len(r.trades)}"),
    ("Final Cash", lambda r: f"{r.portfolio.cash:,.0f}"),
    ("Total Value", lambda r: f"{r.equity_curve[-1][1]:,.0f}"),
]

for name, fn in rows:
    vals = "".join(f"{fn(r):>18}" for r in results.values())
    print(f"{name:>16}{vals}")

# ── 5. Objectives check per variant ─────────────────────────────────

print()
objectives = COMMON["objectives"]
for label, result in results.items():
    metrics = {
        "total_return": result.total_return(),
        "sharpe_ratio": result.sharpe_ratio(),
        "max_drawdown": result.max_drawdown(),
    }
    checks = []
    for metric_name, bounds in objectives.items():
        actual = metrics[metric_name]
        passed = True
        if "min" in bounds:
            passed = passed and actual >= bounds["min"]
        if "max" in bounds:
            passed = passed and actual <= bounds["max"]
        checks.append("✅" if passed else "❌")
    print(f"  {label:<12} return {checks[0]}  sharpe {checks[1]}  drawdown {checks[2]}")

# ── 6. Trade log for each variant ───────────────────────────────────

for label, result in results.items():
    if not result.trades:
        continue
    print(f"\n📋 {label} — Trades ({len(result.trades)}):")
    print(f"  {'Date':<28} {'Side':>4}  {'Shares':>6} {'Symbol':<6} {'Price':>10}")
    print("  " + "-" * 58)
    for fill in result.trades:
        print(
            f"  {fill.filled_at:<28} {fill.order.side:>4}  "
            f"{fill.order.shares:>6} {fill.order.symbol:<6} "
            f"{fill.filled_price:>10.2f}"
        )
