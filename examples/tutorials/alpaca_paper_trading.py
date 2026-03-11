"""Alpaca 模拟交易 Tutorial — 从回测到实盘的完整闭环。

本教程演示 open-xquant 从回测到模拟交易的完整工作流：

  1. 回测策略，使用不同成交价模式压力测试 (FillPriceMode)
  2. 收盘后生成次日交易计划 (OrderGenerator)
  3. 使用 Engine.setup() + step() 逐 bar 执行（实盘模式）
  4. 对比回测与实盘成交偏差 (ExecutionReport)
  5. 连接 Alpaca 获取行情 + 下单（需要 API Key）

第 1-4 节不需要 Alpaca API Key，使用本地数据即可运行。
第 5 节需要设置环境变量：
    export ALPACA_API_KEY="your-paper-api-key"
    export ALPACA_SECRET_KEY="your-paper-secret-key"

Usage:
    python examples/tutorials/alpaca_paper_trading.py
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from oxq.core import Engine, Strategy
from oxq.core.types import Fill, Order, Position
from oxq.indicators import SMA
from oxq.portfolio import ExecutionReport
from oxq.rules import EntryRule, ExitRule
from oxq.signals import Crossover
from oxq.trade import FillPriceMode, PlannedOrder, SimBroker, generate_orders
from oxq.universe import StaticUniverse

# ══════════════════════════════════════════════════════════════════════
# Helper: 构造本地行情数据（避免依赖外部数据源）
# ══════════════════════════════════════════════════════════════════════


def make_sample_data() -> dict[str, pd.DataFrame]:
    """构造模拟行情：下跌 → 上涨 → 下跌，触发 SMA 金叉/死叉。

    120 bars:
    - Bars 0-49:  下跌 200 → 102  (SMA10 < SMA50)
    - Bars 50-89: 上涨 102 → 182  (SMA10 上穿 SMA50 → 金叉买入)
    - Bars 90-119: 下跌 182 → 122 (SMA10 下穿 SMA50 → 死叉卖出)
    """
    n = 120
    dates = pd.bdate_range("2024-01-01", periods=n)
    closes: list[float] = []
    for i in range(50):
        closes.append(200 - i * 2)       # 200 → 102
    for i in range(40):
        closes.append(102 + i * 2)       # 102 → 180
    for i in range(30):
        closes.append(180 - i * 2)       # 180 → 122

    df = pd.DataFrame(
        {
            "open": [c - 1 for c in closes],
            "high": [c + 2 for c in closes],
            "low": [c - 2 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n,
        },
        index=dates,
    )
    return {"AAPL": df}


class FakeMarketDataProvider:
    """本地行情 Provider，用于教程演示（无需外部数据源）。"""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        df = self._data[symbol]
        return df[(df.index >= start) & (df.index <= end)]

    def get_latest(self, symbol: str) -> pd.Series:
        return self._data[symbol].iloc[-1]


def make_strategy() -> Strategy:
    """SMA(10)/SMA(50) 金叉买入、死叉卖出策略。"""
    return Strategy(
        name="sma_crossover_tutorial",
        hypothesis="SMA10 上穿 SMA50 时买入，下穿时卖出",
        universe=StaticUniverse(("AAPL",)),
        indicators={
            "sma_10": (SMA(), {"period": 10}),
            "sma_50": (SMA(), {"period": 50}),
        },
        signals={
            "golden_cross": (Crossover(), {"fast": "sma_10", "slow": "sma_50"}),
        },
        entry_rules=[EntryRule(signal="golden_cross", shares=100)],
        exit_rules=[ExitRule(fast="sma_10", slow="sma_50")],
    )


# ══════════════════════════════════════════════════════════════════════
# Section 1: FillPriceMode — 不同成交价模式压力测试
# ══════════════════════════════════════════════════════════════════════


def section_1_fill_price_modes() -> dict[str, list[Fill]]:
    """用 4 种成交价模式回测同一策略，对比收益差异。

    - CLOSE:     当前 bar 收盘价成交（默认）
    - NEXT_OPEN: 次日开盘价成交（最常用的实盘模拟）
    - NEXT_HIGH: 次日最高价成交（买入最差情况）
    - NEXT_LOW:  次日最低价成交（卖出最差情况）

    Returns
    -------
    dict[str, list[Fill]]
        每种模式的成交记录，供后续 ExecutionReport 使用。
    """
    print("=" * 72)
    print("Section 1: FillPriceMode — 成交价模式压力测试")
    print("=" * 72)

    data = make_sample_data()
    market = FakeMarketDataProvider(data)
    strategy = make_strategy()
    fills_by_mode: dict[str, list[Fill]] = {}

    modes = [
        ("CLOSE (默认)", FillPriceMode.CLOSE),
        ("NEXT_OPEN", FillPriceMode.NEXT_OPEN),
        ("NEXT_HIGH (买入最差)", FillPriceMode.NEXT_HIGH),
        ("NEXT_LOW (卖出最差)", FillPriceMode.NEXT_LOW),
    ]

    print(f"\n{'Mode':<24} {'Return':>10} {'Sharpe':>10} {'MDD':>10} {'Trades':>8}")
    print("-" * 64)

    for label, mode in modes:
        broker = SimBroker(fill_price_mode=mode)
        result = Engine().run(
            strategy,
            market=market,
            broker=broker,
            start="2024-01-01",
            end="2024-12-31",
        )
        fills_by_mode[label] = result.trades

        print(
            f"{label:<24} "
            f"{result.total_return():>9.2%} "
            f"{result.sharpe_ratio():>10.2f} "
            f"{result.max_drawdown():>9.2%} "
            f"{len(result.trades):>8}"
        )

    print("\n  CLOSE 是默认模式。NEXT_OPEN 最接近实际执行场景。")
    print("  NEXT_HIGH/LOW 用于压力测试极端情况。\n")

    return fills_by_mode


# ══════════════════════════════════════════════════════════════════════
# Section 2: OrderGenerator — 生成次日交易计划
# ══════════════════════════════════════════════════════════════════════


def section_2_order_generator() -> list[PlannedOrder]:
    """模拟「收盘后跑策略，生成次日交易计划」的场景。

    场景：当前持有 AAPL 30 股，策略建议调整为：
      - AAPL: 40% 权重
      - GOOG: 30% 权重
      - MSFT: 30% 权重

    Returns
    -------
    list[PlannedOrder]
        带上下文的计划订单列表。
    """
    print("=" * 72)
    print("Section 2: OrderGenerator — 生成次日交易计划")
    print("=" * 72)

    # 当前持仓状态
    positions = {
        "AAPL": Position(symbol="AAPL", shares=30, avg_cost=Decimal("150")),
    }

    # 策略输出的目标权重
    target_weights = {
        "AAPL": Decimal("0.4"),
        "GOOG": Decimal("0.3"),
        "MSFT": Decimal("0.3"),
    }

    # 当前市场价格
    prices = {
        "AAPL": Decimal("180"),
        "GOOG": Decimal("140"),
        "MSFT": Decimal("420"),
    }

    total_capital = Decimal("100000")

    # 生成交易计划
    planned = generate_orders(
        target_weights=target_weights,
        positions=positions,
        prices=prices,
        total_capital=total_capital,
    )

    # 展示计划（供交易员审核）
    print(f"\n  总资金: ${total_capital:,.0f}")
    print(f"  当前持仓: AAPL x {positions['AAPL'].shares}")
    print(f"\n  {'Symbol':<8} {'Side':>5} {'Shares':>8} {'Current':>10} {'Target':>10} {'Amount':>12}")
    print("  " + "-" * 55)
    for p in planned:
        print(
            f"  {p.order.symbol:<8} "
            f"{p.order.side:>5} "
            f"{p.order.shares:>8} "
            f"{p.current_shares:>10} "
            f"{p.target_shares:>10} "
            f"${p.estimated_amount:>10,.0f}"
        )
    print(f"\n  共 {len(planned)} 笔订单待审核。")
    print("  交易员确认后，可通过 LiveBroker 逐笔提交。\n")

    return planned


# ══════════════════════════════════════════════════════════════════════
# Section 3: Engine.setup() + step() — 逐 bar 执行（实盘模式）
# ══════════════════════════════════════════════════════════════════════


def section_3_engine_step() -> None:
    """演示 Engine.setup() + step() 逐 bar 执行模式。

    实盘场景中：
    - setup() 在启动时调用一次（加载历史数据，计算指标/信号）
    - step(date) 由外部调度器（cron/APScheduler）每日调用
    - 引擎不含定时器，完全由外部控制节奏
    """
    print("=" * 72)
    print("Section 3: Engine.setup() + step() — 逐 bar 执行")
    print("=" * 72)

    data = make_sample_data()
    market = FakeMarketDataProvider(data)
    strategy = make_strategy()

    # ── 方式 A: 传统 run() 一次性执行 ──
    engine_a = Engine()
    result_a = engine_a.run(
        strategy, market=market, broker=SimBroker(),
        start="2024-01-01", end="2024-12-31",
    )

    # ── 方式 B: setup() + step() 逐 bar 执行 ──
    engine_b = Engine()
    engine_b.setup(
        strategy=strategy,
        market=market,
        broker=SimBroker(),
        start="2024-01-01",
        end="2024-12-31",
    )

    print(f"\n  共 {len(engine_b.dates)} 根 bar 待处理")
    print("  逐 bar 执行中...")

    for i, date in enumerate(engine_b.dates):
        engine_b.step(date)
        # 实盘中这里可以加入日志、通知、风控检查等
        if i % 30 == 0:
            equity = engine_b.result.equity_curve[-1][1] if engine_b.result.equity_curve else 0
            print(f"    [{date.date()}] bar {i+1}/{len(engine_b.dates)}, equity={equity:,.0f}")

    result_b = engine_b.result

    # 验证两种方式结果一致
    eq_a = [v for _, v in result_a.equity_curve]
    eq_b = [v for _, v in result_b.equity_curve]
    match = all(abs(a - b) < 0.01 for a, b in zip(eq_a, eq_b))

    print(f"\n  run() 总收益:  {result_a.total_return():.2%}")
    print(f"  step() 总收益: {result_b.total_return():.2%}")
    print(f"  结果一致: {'YES' if match else 'NO'}")
    print()

    assert match, "run() 和 setup()+step() 结果不一致！"


# ══════════════════════════════════════════════════════════════════════
# Section 4: ExecutionReport — 回测 vs 实盘偏差分析
# ══════════════════════════════════════════════════════════════════════


def section_4_execution_report() -> None:
    """对比回测成交与「模拟实盘」成交的偏差。

    场景：
    - 回测使用 CLOSE 成交价（理想化）
    - 「实盘」使用 NEXT_OPEN 成交价（更接近真实执行）
    - ExecutionReport 按 (symbol, date, side) 聚合，计算滑点
    """
    print("=" * 72)
    print("Section 4: ExecutionReport — 回测 vs 实盘偏差分析")
    print("=" * 72)

    data = make_sample_data()
    market = FakeMarketDataProvider(data)
    strategy = make_strategy()

    # 回测: CLOSE 模式
    sim_result = Engine().run(
        strategy, market=market, broker=SimBroker(fill_price_mode=FillPriceMode.CLOSE),
        start="2024-01-01", end="2024-12-31",
    )

    # 「实盘」: NEXT_OPEN 模式（模拟次日开盘执行）
    live_result = Engine().run(
        strategy, market=market, broker=SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN),
        start="2024-01-01", end="2024-12-31",
    )

    # 生成偏差报告
    report = ExecutionReport(
        sim_fills=sim_result.trades,
        live_fills=live_result.trades,
    )

    # 逐笔对比
    print(f"\n  {'Symbol':<8} {'Date':<12} {'Side':>5} {'Sim Price':>11} {'Live Price':>11} {'Slippage':>10}")
    print("  " + "-" * 60)
    for c in report.comparisons:
        print(
            f"  {c.symbol:<8} {c.date:<12} {c.side:>5} "
            f"${c.sim_avg_price:>9.2f} "
            f"${c.live_avg_price:>9.2f} "
            f"{c.price_slippage:>9.2%}"
        )

    # 汇总统计
    s = report.summary()
    print("\n  汇总:")
    print(f"    总交易笔数:   {s['total_trades']}")
    print(f"    匹配交易:     {s['matched_trades']}")
    print(f"    仅回测有:     {s['sim_only_trades']}")
    print(f"    仅实盘有:     {s['live_only_trades']}")
    print(f"    平均滑点:     {s['avg_price_slippage']:.4%}")
    print()


# ══════════════════════════════════════════════════════════════════════
# Section 5: Alpaca 集成（需要 API Key）
# ══════════════════════════════════════════════════════════════════════


def section_5_alpaca_integration() -> None:
    """连接 Alpaca Paper Trading，获取行情 + 下单。

    需要环境变量:
        ALPACA_API_KEY, ALPACA_SECRET_KEY

    跳过条件: 未设置环境变量时自动跳过。
    """
    import os

    print("=" * 72)
    print("Section 5: Alpaca 集成 — 行情获取 + 下单")
    print("=" * 72)

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("\n  [SKIP] 未设置 ALPACA_API_KEY / ALPACA_SECRET_KEY，跳过本节。")
        print("  设置方法:")
        print("    export ALPACA_API_KEY='your-paper-api-key'")
        print("    export ALPACA_SECRET_KEY='your-paper-secret-key'")
        print()
        _show_alpaca_code_example()
        return

    # ── 5a: AlpacaMarketDataProvider ──
    from oxq.contrib.alpaca import AlpacaMarketDataProvider

    print("\n  5a. 从 Alpaca 获取行情数据...")
    provider = AlpacaMarketDataProvider(api_key=api_key, secret_key=secret_key)
    try:
        bars = provider.get_bars(["AAPL"], "2024-12-01", "2024-12-31")
        if "AAPL" in bars:
            df = bars["AAPL"]
            print(f"    获取到 {len(df)} 根日线 bar")
            print(f"    最新收盘价: ${df['close'].iloc[-1]:.2f}")
            print(f"    列: {list(df.columns)}")

        latest = provider.get_latest(["AAPL"])
        if "AAPL" in latest:
            print(f"    最新 bar 收盘价: ${latest['AAPL']['close'].iloc[0]:.2f}")
    finally:
        provider.close()

    # ── 5b: LiveBroker ──
    from oxq.trade import LiveBroker

    print("\n  5b. 通过 LiveBroker 下单...")
    broker = LiveBroker(paper=True)
    try:
        order = Order(symbol="AAPL", side="BUY", shares=1)
        order_id = broker.submit_order(order)
        print(f"    提交订单: {order_id}")

        import time
        for _ in range(10):
            fills = broker.get_fills()
            if fills:
                for fill in fills:
                    print(
                        f"    成交: {fill.order.symbol} "
                        f"{fill.order.side} {fill.order.shares} 股 "
                        f"@ ${fill.filled_price}"
                    )
                break
            time.sleep(1)
        else:
            print("    等待超时，未收到成交回报")
    finally:
        broker.close()

    print()


def _show_alpaca_code_example() -> None:
    """展示 Alpaca 集成的代码示例（不实际执行）。"""
    print("\n  以下是 Alpaca 集成的代码示例:\n")
    print("  # ── 获取行情 ──")
    print("  from oxq.contrib.alpaca import AlpacaMarketDataProvider")
    print()
    print("  provider = AlpacaMarketDataProvider()")
    print("  bars = provider.get_bars(['AAPL', 'GOOG'], '2024-01-01', '2024-12-31')")
    print("  latest = provider.get_latest(['AAPL'])")
    print()
    print("  # ── 下单 ──")
    print("  from oxq.trade import LiveBroker")
    print("  from oxq.core.types import Order")
    print()
    print("  broker = LiveBroker(paper=True)")
    print("  order = Order(symbol='AAPL', side='BUY', shares=1)")
    print("  broker.submit_order(order)")
    print("  fills = broker.get_fills()")
    print()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    """运行完整教程。"""
    print()
    print("  Alpaca Paper Trading Tutorial")
    print("  从回测到模拟交易的完整工作流")
    print()

    section_1_fill_price_modes()
    section_2_order_generator()
    section_3_engine_step()
    section_4_execution_report()
    section_5_alpaca_integration()

    print("=" * 72)
    print("Tutorial 完成！")
    print("=" * 72)
    print()
    print("  本教程演示了从回测到模拟交易的完整闭环:")
    print("    1. FillPriceMode   — 不同成交价假设的压力测试")
    print("    2. OrderGenerator  — 收盘后生成次日交易计划")
    print("    3. Engine.step()   — 逐 bar 执行（实盘模式基础）")
    print("    4. ExecutionReport — 回测 vs 实盘偏差分析")
    print("    5. Alpaca 集成     — 真实行情获取 + 下单")
    print()


if __name__ == "__main__":
    main()
