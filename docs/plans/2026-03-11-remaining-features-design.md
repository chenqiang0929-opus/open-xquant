# 剩余功能设计 — OrderGenerator / SimBroker FillPriceMode / AlpacaMarketDataProvider / Engine.step() / ExecutionReport

## 目标

实现从回测到实盘的完整闭环：OrderGenerator 生成交易计划 → SimBroker 多模式回测 → Alpaca 数据源 + Engine 实盘模式 → ExecutionReport 回测 vs 实盘偏差分析。

## 整体架构与依赖关系

5 个功能分 4 层，有明确的依赖链：

```
Layer 1 (基础工具):
  OrderGenerator          — 纯函数，无外部依赖
  SimBroker FillPriceMode — 扩展现有 SimBroker

Layer 2 (数据源):
  AlpacaMarketDataProvider — 实现 MarketDataProvider Protocol

Layer 3 (引擎):
  Engine.step()           — 依赖 MarketDataProvider + Broker

Layer 4 (分析):
  ExecutionReport         — 依赖 Fill 数据（来自 SimBroker 和 LiveBroker）
```

实现顺序遵循依赖关系：Step 1 → Step 2 → Step 3 → Step 4。

### 文件布局

```
src/oxq/
├── contrib/
│   └── alpaca/
│       ├── __init__.py
│       ├── client.py           # AlpacaClient（从 trade/ 迁入）
│       └── market_data.py      # AlpacaMarketDataProvider
├── trade/
│   ├── order_generator.py      # 新增：OrderGenerator 纯函数
│   ├── sim_broker.py           # 修改：加 FillPriceMode
│   ├── live_broker.py          # 修改：import 路径改为 contrib.alpaca.client
│   └── ...
├── core/
│   └── engine.py               # 修改：新增 setup() + step()
└── portfolio/
    └── execution_report.py     # 新增：ExecutionReport
```

## 1. OrderGenerator

纯函数，用于"收盘后跑策略，生成次日交易计划"的场景。交易员（或自动化系统）审核后执行。

```python
# src/oxq/trade/order_generator.py

@dataclass(frozen=True)
class PlannedOrder:
    """一笔带上下文的计划订单，供交易员审核。"""
    order: Order                    # 实际的 oxq Order 对象
    current_shares: int             # 当前持仓
    target_shares: int              # 目标持仓
    current_weight: Decimal         # 当前权重
    target_weight: Decimal          # 目标权重
    estimated_amount: Decimal       # 预估交易金额 (shares * price)

def generate_orders(
    target_weights: dict[str, Decimal],   # {"AAPL": Decimal("0.3"), "GOOG": Decimal("0.2")}
    positions: dict[str, Position],       # 当前持仓
    prices: dict[str, Decimal],           # 当前价格
    total_capital: Decimal,               # 总资金（cash + 持仓市值）
    lot_size: int = 1,                    # 整手约束，A 股传 100
) -> list[PlannedOrder]:
    ...
```

### 核心逻辑

1. 对每个 symbol 计算 `target_shares = floor(total_capital * target_weight / price / lot_size) * lot_size`
2. `delta = target_shares - current_shares`
3. `delta > 0` → BUY，`delta < 0` → SELL，`delta == 0` → 跳过
4. 不在 `target_weights` 中但在 `positions` 中的 symbol → 全部 SELL（清仓）
5. 所有订单默认 `order_type="market"`（次日开盘执行）

### 与 RebalanceRule 的关系

RebalanceRule 保持不变（引擎内部逐 bar 使用），OrderGenerator 是独立的外部工具。两者逻辑相似但职责不同——RebalanceRule 在引擎循环中运行，OrderGenerator 在引擎外部供人工审核流程使用。

## 2. SimBroker FillPriceMode

扩展 SimBroker，支持 4 种成交价模式。

```python
# src/oxq/trade/sim_broker.py

class FillPriceMode(Enum):
    CLOSE = "close"               # 当前 bar 收盘价（默认，现有行为）
    NEXT_OPEN = "next_open"       # 次日开盘价
    NEXT_HIGH = "next_high"       # 次日最高价（压力测试：买在最高）
    NEXT_LOW = "next_low"         # 次日最低价（压力测试：卖在最低）
```

### SimBroker 改动

1. `__init__` 增加 `fill_price_mode: FillPriceMode = FillPriceMode.CLOSE` 参数
2. 提取内部方法 `_get_fill_price(order, mktdata, date) -> Decimal | None`：
   - `CLOSE` 模式：取 `mktdata[symbol].loc[date, "close"]`（现有逻辑）
   - `NEXT_*` 模式：取 `mktdata[symbol]` 中 `date` 的下一行，读取对应列。如果没有下一行（最后一根 bar），返回 None 表示跳过不成交
3. `fill_market_orders()` 和 `process_pending_orders()` 调用 `_get_fill_price()` 替代硬编码的 close 价格

### 成交价含义

- BUY + NEXT_HIGH → 买在次日最高价（最差买入，压力测试）
- SELL + NEXT_LOW → 卖在次日最低价（最差卖出，压力测试）
- 不做自动翻转，用户显式选择模式，自己理解含义

### 对 Engine 的影响

无。Engine 只调用 `broker.on_bar_open()` / `on_bar_close()`，fill price 选择完全在 SimBroker 内部。

## 3. AlpacaClient 迁移 + AlpacaMarketDataProvider

### AlpacaClient 迁移

将 `src/oxq/trade/alpaca_client.py` 迁移到 `src/oxq/contrib/alpaca/client.py`。`contrib/alpaca/` 只放和 Alpaca API 直接交互的代码。`trade/live_broker.py` 改为 `from oxq.contrib.alpaca.client import AlpacaClient`。

`trade/__init__.py` 保持 re-export LiveBroker（向后兼容）。

### AlpacaMarketDataProvider

```python
# src/oxq/contrib/alpaca/market_data.py

class AlpacaMarketDataProvider:
    """从 Alpaca Market Data API 获取行情数据。"""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        # 独立的 httpx.Client，指向 data API
        # Market Data API base URL 固定为 https://data.alpaca.markets
        ...

    def get_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
        timeframe: str = "1Day",
    ) -> dict[str, pd.DataFrame]:
        """获取历史 OHLCV bars。"""
        # GET /v2/stocks/bars (批量，支持分页)
        ...

    def get_latest(
        self, symbols: list[str],
    ) -> dict[str, pd.DataFrame]:
        """获取最新一根 bar（实盘 step() 用）。"""
        # GET /v2/stocks/bars/latest (批量最新 bar)
        ...
```

### 为什么独立 httpx.Client

Market Data API (`https://data.alpaca.markets`) 和 Trading API (`https://paper-api.alpaca.markets`) 是不同的服务，base URL 不同。AlpacaMarketDataProvider 自己创建独立的 httpx.Client，不修改 AlpacaClient。认证方式相同（APCA-API-KEY-ID + APCA-API-SECRET-KEY）。

### 返回格式

统一为 `dict[str, pd.DataFrame]`，DataFrame 列为 `open, high, low, close, volume`，index 为 `pd.DatetimeIndex`。与 LocalMarketDataProvider 保持一致。

## 4. Engine.step() 实盘模式

在 Engine 上新增 `setup()` + `step()` 方法，复用现有的 bar 处理逻辑。

```python
# src/oxq/core/engine.py

class Engine:
    # 现有 run() 保持不变，内部重构为 setup() + 循环 step()

    def setup(
        self,
        mktdata_provider: MarketDataProvider,
        broker: Broker,
        portfolio: Portfolio,
        rules: list[Rule],
        signals: list[Signal] | None = None,
        indicators: list[Indicator] | None = None,
    ) -> None:
        """初始化 Engine 状态，供 step() 使用。"""
        ...

    def step(self, date: pd.Timestamp) -> None:
        """处理单根 bar：获取最新数据 → 计算指标/信号 → 评估规则 → 执行订单。"""
        # 1. 从 provider 获取最新 bar，追加到 _mktdata
        # 2. 复用现有 Phase 1-3 逻辑（compute_indicators → compute_signals → evaluate_rules）
        ...
```

### 关键设计决策

1. **`setup()` + `step()` 分离**：`setup()` 初始化一次，`step()` 可被反复调用
2. **`run()` 重构**：`run()` 内部变成 `setup() + for date in dates: step(date)`，保持向后兼容
3. **外部调度**：Engine 不含定时器。用户用 cron、APScheduler 或手动脚本调用 `step()`
4. **数据累积**：每次 `step()` 将新 bar 追加到 `_mktdata`，指标计算（如 SMA_20）能拿到完整窗口。首次 `setup()` 时可预加载历史数据

## 5. ExecutionReport

按 symbol + date 聚合对比回测 vs 实盘成交。

```python
# src/oxq/portfolio/execution_report.py

@dataclass(frozen=True)
class SymbolDateFill:
    """某 symbol 某天的聚合成交。"""
    symbol: str
    date: str
    side: str                    # BUY / SELL
    total_shares: int            # 总成交量
    avg_price: Decimal           # 加权平均成交价
    total_fee: Decimal           # 总手续费

@dataclass(frozen=True)
class FillComparison:
    """一笔聚合成交的回测 vs 实盘对比。"""
    symbol: str
    date: str
    side: str
    # 回测
    sim_shares: int
    sim_avg_price: Decimal
    sim_fee: Decimal
    # 实盘
    live_shares: int
    live_avg_price: Decimal
    live_fee: Decimal
    # 偏差
    shares_diff: int             # live - sim
    price_slippage: Decimal      # (live - sim) / sim，百分比
    fee_diff: Decimal

class ExecutionReport:
    def __init__(
        self,
        sim_fills: list[Fill],
        live_fills: list[Fill],
    ) -> None:
        self._comparisons = _match_and_compare(sim_fills, live_fills)

    @property
    def comparisons(self) -> list[FillComparison]:
        """逐笔对比明细。"""
        return self._comparisons

    def summary(self) -> dict[str, Decimal]:
        """汇总统计。"""
        return {
            "total_trades": ...,
            "matched_trades": ...,       # 回测和实盘都有的
            "sim_only_trades": ...,      # 回测有、实盘没有
            "live_only_trades": ...,     # 实盘有、回测没有
            "avg_price_slippage": ...,   # 平均价格滑点
            "total_fee_diff": ...,       # 总手续费差异
        }
```

### 匹配逻辑（`_match_and_compare`）

1. 将 sim_fills 和 live_fills 分别按 `(symbol, date, side)` 聚合为 `SymbolDateFill`
2. 以 `(symbol, date, side)` 为 key 做 full outer join
3. 两边都有 → 生成 FillComparison 带偏差计算
4. 只有一边有 → `sim_only` 或 `live_only`（对方 shares/price 填 0）

### 使用方式

```python
report = ExecutionReport(sim_fills=backtest_result.trades, live_fills=live_fills)
for c in report.comparisons:
    print(f"{c.symbol} {c.date}: slippage={c.price_slippage:.2%}")
print(report.summary())
```

## 测试策略

| 功能 | 单元测试 | 说明 |
|------|----------|------|
| OrderGenerator | `tests/trade/test_order_generator.py` | 多 symbol 权重分配、lot_size 约束、清仓逻辑 |
| FillPriceMode | `tests/trade/test_sim_broker.py` (扩展) | 4 种模式各一个 case、最后一根 bar 跳过 |
| AlpacaMarketDataProvider | `tests/contrib/test_alpaca_market_data.py` | Mock HTTP，验证 DataFrame 格式 |
| Engine.step() | `tests/core/test_engine.py` (扩展) | step() 逐步调用 vs run() 结果一致 |
| ExecutionReport | `tests/portfolio/test_execution_report.py` | 完全匹配、sim_only、live_only、partial fill 聚合 |

集成测试标记 `@pytest.mark.integration`，连接 Alpaca Paper Trading。

## 架构文档更新

需同步更新 `docs/architecture.md`：
- 第 2 节项目结构：新增 `contrib/` 子包说明
- 第 4.6 节 Broker Protocol：更新运行模式表，将"未来"改为已实现
- 第 5.12 节交易执行：新增 OrderGenerator、FillPriceMode 说明
- 第 9 节实现路线：更新 Phase 3 已完成项
