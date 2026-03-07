"""Signal Comparison Strategy — comparing weight allocation methods.

Hypothesis:
    在纳指100ETF、沪深300ETF、黄金ETF三类资产中，使用相同的动量指标体系
    （Momentum/Volatility 风险调整动量），但采用不同的信号层权重分配方法
    （等权、归一化排名、风险平价），在固定 10 天调仓频率下，
    不同信号方法会产生显著不同的收益与风险特征。

Objectives:
    - 年化收益率 ≥ 5%
    - 夏普比率 ≥ 0.5
    - 最大回撤 ≥ -20%

Pipeline:
    Indicator 层 — Momentum(20), RollingVolatility(20), Ratio(mom/vol)
    Signal 层   — EqualWeight / TopNRanking / RiskParity
    Rule 层     — RebalanceRule(weight_col=tw, frequency=10)

Variants (三种信号方法对比):
    - 等权 (EqualWeight)   — 所有资产等权分配
    - 动量排名 (TopNRanking) — 按动量排名，Top 3 归一化权重，过滤负动量
    - 风险平价 (RiskParity) — 按波动率倒数分配权重，低波动资产获得更高权重

Architecture note:
    Strategy 定义与执行环境解耦。Engine 只依赖三个 Protocol 接口：
    MarketDataProvider, OrderRouter, FillReceiver。

    本例使用 3 只跨资产类别 ETF（股票+商品），展示不同信号层权重分配方法
    在相同指标和调仓频率下的表现差异。

    标的:
      - 513100.SS  纳指100ETF（美股科技）
      - 510300.SS  沪深300ETF（A股大盘）
      - 518880.SS  黄金ETF（贵金属）

Usage:
    # Data is downloaded automatically via YFinanceDownloader
    python examples/strategies/signal_comparison.py
"""

from oxq.core import Engine, Strategy
from oxq.data import LocalMarketDataProvider, YFinanceDownloader
from oxq.indicators import Momentum, Ratio, RollingVolatility
from oxq.rules import RebalanceRule
from oxq.signals import EqualWeight, RiskParity, TopNRanking
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
START = "2021-01-01"
END = "2026-03-05"

# ── 0.5. Download data ─────────────────────────────────────────────

downloader = YFinanceDownloader()
for sym in SYMBOLS:
    downloader.download(sym, start=START, end=END)

# ── 1. Common pipeline components (indicators only) ────────────────

COMMON = dict(
    hypothesis=(
        "在纳指100ETF、沪深300ETF、黄金ETF中，"
        "使用 Momentum(20)/Volatility(20) 风险调整动量指标，"
        "对比等权、归一化排名、风险平价三种信号方法，"
        "固定 10 天调仓频率，观察收益与风险差异"
    ),
    objectives={
        "annualized_return": {"min": 0.05},
        "sharpe_ratio": {"min": 0.5},
        "max_drawdown": {"min": -0.20},
    },
    benchmarks=[],
    universe=StaticUniverse(SYMBOLS),
    indicators={
        "mom": (Momentum(), {"column": "close", "period": 20}),
        "vol": (RollingVolatility(), {"column": "close", "period": 20}),
        "ram": (Ratio(), {"col_a": "mom", "col_b": "vol"}),
    },
    entry_rules=[],
    exit_rules=[],
    rebalance_rules=[RebalanceRule(weight_col="tw", frequency=10)],
)

# ── 2. Three signal method variants ────────────────────────────────

STRATEGIES = {
    "等权": Strategy(
        name="equal_weight",
        signals={
            "tw": (EqualWeight(), {}),
        },
        **COMMON,
    ),
    "动量排名": Strategy(
        name="momentum_ranking",
        signals={
            "tw": (TopNRanking(), {"score": "ram", "n": 3, "filter_negative": True}),
        },
        **COMMON,
    ),
    "风险平价": Strategy(
        name="risk_parity",
        signals={
            "tw": (RiskParity(), {"vol": "vol"}),
        },
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

universe_str = ", ".join(f"{s}({SYMBOL_NAMES[s]})" for s in SYMBOLS)
print("=" * 76)
print("Signal Comparison Strategy — Weight Allocation Method Comparison")
print(f"Universe: {universe_str}")
print(f"Period: {START} ~ {END}  |  Init Cash: {INITIAL_CASH:,.0f}")
print("=" * 76)

header = f"{'':>20}" + "".join(f"{label:>18}" for label in results)
print(header)
print("-" * len(header))

rows = [
    ("Total Return", lambda r: f"{r.total_return():.2%}"),
    ("Ann. Return", lambda r: f"{r.annualized_return():.2%}"),
    ("Ann. Volatility", lambda r: f"{r.annualized_volatility():.2%}"),
    ("Sharpe Ratio", lambda r: f"{r.sharpe_ratio():.2f}"),
    ("Calmar Ratio", lambda r: f"{r.calmar_ratio():.2f}"),
    ("Sortino Ratio", lambda r: f"{r.sortino_ratio():.2f}"),
    ("Max Drawdown", lambda r: f"{r.max_drawdown():.2%}"),
    ("Total Trades", lambda r: f"{len(r.trades)}"),
    ("Final Cash", lambda r: f"{r.portfolio.cash:,.0f}"),
    ("Total Value", lambda r: f"{r.equity_curve[-1][1]:,.0f}"),
]

for name, fn in rows:
    vals = "".join(f"{fn(r):>18}" for r in results.values())
    print(f"{name:>20}{vals}")

# ── 5. Objectives check per variant ─────────────────────────────────

print()
objectives = COMMON["objectives"]
metric_fns = {
    "annualized_return": lambda r: r.annualized_return(),
    "sharpe_ratio": lambda r: r.sharpe_ratio(),
    "max_drawdown": lambda r: r.max_drawdown(),
}

for label, result in results.items():
    checks = []
    for metric_name, bounds in objectives.items():
        actual = metric_fns[metric_name](result)
        passed = True
        if "min" in bounds:
            passed = passed and actual >= bounds["min"]
        if "max" in bounds:
            passed = passed and actual <= bounds["max"]
        checks.append(("pass" if passed else "FAIL", metric_name))
    status = "  ".join(f"{name} {tag}" for tag, name in checks)
    print(f"  {label:<12} {status}")

# ── 6. Trade log (first 20 trades per variant) ──────────────────────

for label, result in results.items():
    if not result.trades:
        continue
    shown = result.trades[:20]
    more = len(result.trades) - len(shown)
    print(f"\n  {label} — Trades ({len(result.trades)} total, showing first {len(shown)}):")
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
