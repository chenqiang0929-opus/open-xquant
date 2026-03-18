# open-xquant 架构文档

## 1. 设计哲学

open-xquant 是一个 **Agent-First** 的开源量化交易框架。引擎（SDK + Tool 定义）是地基，Skill 才是 Agent-First 的交付面——用户感知到的 Agent 体验，由 Skill 层交付。

底层是严谨的量化金融引擎，经 Indicator → Universe → Signal → Portfolio → Rule → Broker 管道生成交易决策；核心资产是 **Python SDK + 协议无关的 Tool 定义**（名称、参数、语义），每个工作流编写 skill.md，指导 Agent 如何组合 tools 完成复杂任务。

**三种使用角色与入口**：

- **Coding Agent / 开发者** → `import oxq`（主要方式），直接调用 SDK 和 Tool 函数
- **非 Coding AI 客户端**（Claude Desktop、Windsurf 等）→ 通过 MCP Server 调用（可选分发层）
- **平台方** → 基于 SDK + Tool 定义自建接口（REST API、gRPC 等）

**四大设计原则**：

- **声明式**：策略定义与执行分离。策略是"做什么"的声明，引擎负责"怎么做"
- **确定性**：相同输入必须产生相同输出。不可变数据类型 + 纯函数计算 + 审计追踪
- **约束即自由**：统一的 Protocol 接口收窄 AI 的选择空间到只有正确的做法，消除幻觉温床
- **全流程**：从策略构建、回测、参数优化、统计检验到交易执行，端到端覆盖

---

## 2. 项目结构

```
open-xquant/
├── src/oxq/                        # 主 Python 包（pip install open-xquant）
│   ├── core/                       # 核心引擎（类型、策略定义、执行引擎、注册中心、异常）
│   ├── indicators/                 # 技术指标库（SMA, EMA, RSI, MACD, BBands...）
│   ├── signals/                    # 信号生成器（交叉、阈值、比较、公式、组合）
│   ├── rules/                      # 交易规则（入场、出场、仓位管理、风控、再平衡）
│   ├── portfolio/                  # 组合管理（持仓、订单簿、记账、绩效分析）
│   ├── optimize/                   # 参数优化（网格/随机/贝叶斯搜索、滚动前推、统计检验）
│   ├── trade/                      # 交易执行（SimBroker、LiveBroker、费率、滑点、OrderGenerator）
│   ├── contrib/                    # 第三方券商/数据源集成（按券商组织）
│   │   └── alpaca/                # Alpaca 集成（AlpacaClient、AlpacaMarketDataProvider）
│   ├── universe/                   # Universe 构建（静态池、指数成分、条件过滤）
│   ├── data/                       # 数据层（Provider 协议、行情/因子数据、数据加载）
│   ├── observe/                    # 可观测性（追踪、日志、事件总线、审计）
│   └── tools/                      # 协议无关的 Tool 定义（核心资产）
│
├── mcp_server/                     # MCP 协议适配层（可选分发渠道）
├── skills/                         # Agent Skill 定义（markdown）
├── examples/                       # 示例策略、demo 应用、教程
├── tests/                          # 测试（镜像 src/oxq/ 结构）
├── docs/                           # 文档
├── pyproject.toml
├── LICENSE                         # MIT
└── README.md
```

---

## 3. 分层架构

```
┌──────────────────────────────────────────────────────┐
│              Skill Layer (skill.md)                   │  ← Agent 工作流指导
│  strategy-builder / engine-runner / tuner ...          │
├──────────────────────────────────────────────────────┤
│              SDK + Tool Layer                         │  ← 核心资产
│  oxq.universe / oxq.core / oxq.trade / ...           │  Python SDK
│  oxq.tools (协议无关的 Tool 定义)                     │  Tool 定义
│          ┆                                            │
│     ┌────┴──────────────────────┐                    │
│     │ MCP Server (可选分发层)    │  ← 非 Coding AI    │
│     │ mcp_server/               │    客户端适配       │
│     └───────────────────────────┘                    │
├──────────────────────────────────────────────────────┤
│              Engine Layer                             │  ← 纯计算，无 I/O
│  Indicator → Universe → Signal → Portfolio → Rule → Broker │
├──────────────────────────────────────────────────────┤
│              Provider Layer                           │  ← 数据注入（Protocol）
│  MarketData / Factor / Portfolio                      │
└──────────────────────────────────────────────────────┘
```

---

## 4. 核心引擎设计

### 4.1 Strategy = Universe + Signal + Portfolio

策略始于**假设和目标**，而非代码——明确的假设（hypothesis）定义了策略试图捕捉的市场现象，目标（objectives）量化了成功标准，基准（benchmarks）提供了比较的参照系。Strategy 由三个核心组件构成：Universe（标的池）、Signal（信号生成）、Portfolio（组合构建）。Strategy 是纯声明式容器，直接传给 Engine 执行。

Engine 驱动完整管道：**Indicators → Universe → Signal → Portfolio → Pre-trade Rule → Trading Algorithm → Broker → Post-trade Rule**。Indicators 由 Strategy 声明但由 Engine 计算；Rules 不属于 Strategy，而是传给 `Engine.run(rules=[...])`。

**核心组件的精确语义**：

- **Universe**：选择参与计算的标的（Filter），可以是假设的一部分（"该现象存在于大盘股中"），也可以是业务约束（"只交易沪深300成分股"）
- **Indicator**：从行情数据衍生的量化值。**路径无关**——不知道当前持仓和交易历史，可向量化计算
- **Signal**：对某个时间点的**方向性预测**。描述"交易的欲望"而非行动本身——信号可能因规则约束而不触发交易
- **Portfolio**：组合构建层，通过 PortfolioOptimizer Protocol 将信号转化为目标权重/持仓
- **Rule**：返回 RuleResult（weights/constraints/target_positions/hold），而非 Order。Rules 传给 Engine.run()，不属于 Strategy

**SDK 方式定义策略**：

Strategy 是纯声明式 dataclass，通过构造器传入所有组件。Indicator/Signal 存储为 `(instance, params)` 元组，引擎执行时按名称追加为宽表列。信号和规则通过字符串列名引用指标输出（如 `"sma_fast"`），无需额外绑定语法。

```python
from oxq.core import Engine, Strategy
from oxq.indicators import SMA
from oxq.signals import Crossover
from oxq.universe import StaticUniverse

strategy = Strategy(
    name="sma_crossover",
    hypothesis="短期均线上穿长期均线的标的在后续持有期内有正超额收益",
    objectives={
        "total_return": {"min": 0.05},
        "sharpe_ratio": {"min": 0.5, "target": 1.5},
        "max_drawdown": {"max": -0.25, "target": -0.15},
    },
    benchmarks=["SPY"],
    universe=StaticUniverse(("AAPL",)),
    indicators={
        "sma_fast": (SMA(), {"column": "close", "period": 10}),
        "sma_slow": (SMA(), {"column": "close", "period": 50}),
    },
    signals={
        "golden_cross": (Crossover(), {"fast": "sma_fast", "slow": "sma_slow"}),
    },
)

# 运行（Provider 决定模式：回测 / 模拟 / 实盘）
# Rules 传给 Engine.run()，不属于 Strategy
engine = Engine()
result = engine.run(strategy,
    market=LocalMarketDataProvider(),
    broker=sim_broker,
    rules=[StopLossRule(threshold=0.05)],
    start="2023-01-01", end="2024-12-31")
```

**等价的 Tool 调用（AI Agent 方式）**：

Tool 定义在 `oxq.tools` 中，协议无关——Coding Agent 直接 `import` 调用，MCP 客户端通过 MCP 协议调用，平台方也可通过 REST/gRPC 等任意方式触发。

```
→ strategy_create(name="sma_crossover",
    hypothesis="短期均线上穿长期均线的标的在后续持有期内有正超额收益",
    objectives={"total_return": {"min": 0.05}, "sharpe_ratio": {"min": 0.5}},
    benchmarks=["SPY"])
→ strategy_add_indicator(strategy="sma_crossover", name="sma_fast", type="SMA",
    params={"column": "close", "period": 10})
→ strategy_add_indicator(strategy="sma_crossover", name="sma_slow", type="SMA",
    params={"column": "close", "period": 50})
→ strategy_add_signal(strategy="sma_crossover", name="golden_cross", type="Crossover",
    inputs={"fast": "sma_fast", "slow": "sma_slow"})
→ engine_run(strategy="sma_crossover", symbols=["AAPL"],
    start="2023-01-01", end="2024-12-31")
→ engine_results(run_id="...")
→ engine_trade_list(run_id="...")
```

### 4.2 宽表数据模型

`mktdata` 是按 symbol 索引的宽表集合（`Dict[str, pd.DataFrame]`），每个 symbol 对应一张独立的 DataFrame。Indicator、Signal 阶段通过**追加列**逐步加宽各 symbol 的宽表，而非各自维护独立的输出。

**Per-symbol 宽表**（以单个 symbol 为例）：

```
原始行情             Indicator 后              Signal 后               Portfolio+Rule 阶段
+-----------+       +------------------+      +---------------------+
| Open      |       | Open             |      | Open                |
| High      |       | High             |      | High                |
| Low       | ───►  | Low              | ───► | Low                 | ───► 只读,
| Close     |       | Close            |      | Close               |      生成 RuleResult
| Volume    |       | Volume           |      | Volume              |
|           |       | ma50    (新增)   |      | ma50                |
|           |       | ma200   (新增)   |      | ma200               |
|           |       |                  |      | ma50_gt_ma200 (新增)|
+-----------+       +------------------+      +---------------------+
```

**多 symbol 全景**：

```
mktdata（per symbol 的宽表集合）

symbol_A:                         symbol_B:
+-------------------------+      +-------------------------+
| Open | Close | ...      |      | Open | Close | ...      |
| ma50          (Ind层加) |      | ma50          (Ind层加) |
| golden_cross  (Sig层加) |      | golden_cross  (Sig层加) |
+-------------------------+      +-------------------------+
          \                             /
           \                           /
      Signal 层：截面操作（跨 symbol 比较、排名、归一化）
                       ↓
      Portfolio 层：PortfolioOptimizer 优化目标权重
                       ↓
      Rule 层：逐 bar 读取信号/权重 + 持仓状态 → RuleResult
```

各层对 `mktdata` 的操作方式：

| 层 | 数据视角 | 引擎行为 | 模式 |
|---|---|---|---|
| Indicator | per symbol — 每次看到一个 symbol 的 DataFrame | `compute` 返回 Series，**引擎**将其追加为新列 | 纯函数计算 + 引擎回写 |
| Signal | cross-sectional — 读取全 universe 的指标列 | 结果写回各 symbol 的宽表，或生成独立的权重向量（`SparseVector`） | 截面读取 + 引擎回写 |
| Portfolio | cross-sectional — 将信号权重转化为目标持仓 | PortfolioOptimizer 优化权重 | 截面优化 |
| Rule | per bar × 全 universe — 读取当前 bar 的信号/权重 + 持仓状态 | 返回 RuleResult（weights/constraints/hold），不修改 mktdata | **只读** |

> **compute 纯函数 vs 引擎回写**：Indicator/Signal 的 `compute` 方法是纯函数，不修改 mktdata。引擎负责将 `compute` 的返回值追加为 mktdata 的新列。这保证了组件的可测试性和确定性——相同输入必然产生相同输出。

**设计动机**：宽表避免了层间数据传递的复杂性。Signal 无需知道 Indicator 的输出格式，只需按列名引用；Rule 同理。这也天然适合 AI Agent——所有中间结果在同一张表上可见可查。

> **Universe 与宽表的关系**：mktdata 中包含哪些 symbol 的宽表，由 Universe 在每个时间截面上动态决定。Universe 变化时（如指数成分股调整），mktdata 随之增减 symbol 键，确保后续各阶段计算不会引入 survivorship bias。

### 4.3 执行模型

#### 逐层推进，全 symbol 执行

采用逐层推进模式，每个阶段接收**全 universe 的数据**，天然支持截面操作（如跨 symbol 的动量归一化、行业轮动排序）：

```
┌─────────────────────────────────────────────────────┐
│  Phase 1: Indicator                                  │
│  对全 universe 计算指标/因子                          │
│  输入: mktdata (per symbol)                          │
│  输出: mktdata + 指标列 (per symbol)                 │
│  示例: SMA, RSI, 动量因子, 波动率因子                 │
├─────────────────────────────────────────────────────┤
│  Phase 2: Universe Resolution                        │
│  确定当前时间截面参与计算的 symbol 集合                │
│  输入: UniverseProvider + as_of_date                 │
│  输出: list[str] (symbols)                           │
│  示例: 沪深300成分股(PIT), 流动性过滤, 静态列表        │
├─────────────────────────────────────────────────────┤
│  Phase 3: Signal                                     │
│  基于全 universe 的指标结果生成信号                    │
│  输入: 全部 symbol 的指标结果                         │
│  输出: 信号值 / 目标权重 (cross-sectional)            │
│  示例: 截面排序, top_n 筛选, 归一化, 风控调整          │
├─────────────────────────────────────────────────────┤
│  Phase 4: Portfolio                                  │
│  通过 PortfolioOptimizer 将信号转化为目标持仓          │
│  输入: 信号权重 + 约束条件                            │
│  输出: 目标权重 / 目标持仓                            │
│  示例: 等权分配, 风险平价, Kelly 公式                  │
├─────────────────────────────────────────────────────┤
│  Phase 5: Pre-trade Rule                             │
│  交易前风控检查，返回 RuleResult                      │
│  输入: 目标持仓 + 当前 portfolio 状态                  │
│  输出: RuleResult (weights/constraints/hold)          │
│  示例: 回撤熔断, 日内亏损限制                         │
├─────────────────────────────────────────────────────┤
│  Phase 6: Trading Algorithm → Broker                 │
│  生成订单并提交给 Broker 执行                         │
│  输入: 目标持仓 + RuleResult 约束                     │
│  输出: 订单提交 + 成交回报                            │
├─────────────────────────────────────────────────────┤
│  Phase 7: Post-trade Rule                            │
│  交易后处理，返回 RuleResult                          │
│  输入: 成交结果 + portfolio 状态                      │
│  输出: RuleResult (constraints/adjustments)           │
│  示例: 止损止盈挂单, 追踪止损更新                      │
└─────────────────────────────────────────────────────┘
```

外层循环按 phase 推进（`for phase in [indicator, universe, signal, portfolio, pre-trade rule, trading algorithm, broker, post-trade rule]`）。Indicator 最先计算，然后 Universe 确定当前时间截面的 symbol 列表，后续各层输入该 universe 的数据。回测中 universe 可能随再平衡日变化（如指数成分股季度调整），引擎在每个再平衡日重新执行 Universe Resolution。当 universe 只有一个 symbol 时，执行流程退化为逐 symbol 模式，完全兼容单标的策略。Rules 通过 `Engine.run(rules=[...])` 传入，不属于 Strategy 定义。

**分阶段执行（Partial Execution）**：引擎支持在任意阶段终止执行，用于逐组件独立评估：

```python
# 只执行到指标阶段（用于 Indicator 独立评估）
result = engine.run(strategy, providers, run_through="indicator")

# 只执行到信号阶段（用于 Signal 预测力评估）
result = engine.run(strategy, providers, run_through="signal")
```

分阶段执行是逐组件评估的架构基础。先独立验证每个组件的 "goodness of fit"，再组装完整策略，可以更早淘汰无效假设，避免在有缺陷的组件上浪费优化时间。

#### 向量化与逐 bar 状态机

各阶段采用不同的计算模式：

| 阶段 | 计算模式 | 路径依赖 | 调用次数 |
|---|---|---|---|
| Indicator | **向量化** — 对全量时间序列一次调用 | 否 | 每个 symbol 1 次 |
| Signal | **向量化** — 对全量时间序列一次调用 | 否 | 1 次（跨 symbol） |
| Portfolio | **截面** — 对当前 bar 的权重进行优化 | 否 | 每个再平衡点 1 次 |
| Rule | **逐 bar 循环** — 状态机模式，返回 RuleResult | 是 | N 次（N = 触发点数） |

**向量化阶段**（Indicator / Signal）：

```python
# Indicator：per symbol，一次调用处理全部历史（在 Universe Resolution 之前）
for symbol in universe:
    result = indicator.compute(mktdata[symbol], **params)
    mktdata[symbol]["sma_fast"] = result

# Universe：从 UniverseProvider 获取当前 universe
universe = universe_provider.get_universe(as_of_date=context.as_of_date)
# universe = ["600519.SS", "000858.SZ", ...]  随时间变化

# Signal：cross-sectional，一次处理全 universe
scores = {}
for symbol in universe:
    scores[symbol] = mktdata[symbol]["momentum"]
# 截面归一化
weights = normalize(top_n(scores, n=5))
```

**状态机阶段**（Rule）：

```python
# 逐 bar 推进，依赖当前持仓、挂单等状态
while cur_index <= total_bars:
    # Pre-trade Rule → Portfolio Optimizer → Trading Algorithm → Broker → Post-trade Rule
    # Rule 返回 RuleResult（weights/constraints/target_positions/hold）
    # 读取当前 bar 的信号/权重
    # 生成目标持仓，提交订单，更新持仓状态
    cur_index = next_index(cur_index)
```

**dindex 优化**：Rule 循环不需要遍历每一个 bar。利用 Signal 阶段已生成的布尔列，可以预先建立"信号触发索引"（dindex），仅在信号实际触发的时间点执行 Rule，大幅减少迭代次数。

**设计动机**：Indicator/Signal 的值仅取决于截至 T 时刻的行情数据，与历史交易无关，故可向量化一次算完。Rule 的决策取决于当前持仓、挂单、资金等状态，必须逐步推进。这种分离既保证了前两层的计算效率，又保证了 Rule 层的正确性。

### 4.4 核心数据类型

```python
SparseVector = Dict[str, float]           # 权重/分数向量

@dataclass(frozen=True)
class Order:                              # 不可变订单请求
    symbol: str
    side: Literal["BUY", "SELL"]
    shares: int
    order_type: Literal["market", "limit", "stop", "stop_limit", "trailing_stop"] = "market"
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    trail_pct: float | None = None

@dataclass(frozen=True)
class Fill:                               # 成交回报
    order: Order
    filled_price: Decimal                 # 实际成交价（含滑点）
    filled_at: str                        # ISO 日期
    fee: Decimal = Decimal("0")

class ManagedOrder:                       # 订单生命周期状态（OrderBook 管理）
    order: Order                          # 不可变订单请求
    id: str                               # 唯一标识，由 OrderBook 分配
    status: Literal["open", "filled", "partial", "canceled", "expired"]
    created_at: str
    filled_at: str | None
    filled_price: Decimal | None
    filled_shares: int | None
    trail_high_water: Decimal | None      # 追踪止损高水位

@dataclass(frozen=True)
class Position:                           # 单个持仓
    symbol: str
    shares: int
    avg_cost: Decimal                     # 加权平均成本

@dataclass
class Portfolio:                          # 组合状态（可变，Rule 阶段逐 bar 更新）
    cash: Decimal
    positions: dict[str, Position]
    bar_prices: dict[str, Decimal]
    # total_value(prices) 方法

@dataclass(frozen=True)
class UniverseSnapshot:                   # 某时间截面的标的池快照
    as_of_date: str                       # 截面日期
    symbols: tuple[str, ...]              # 当前成分（不可变）
    source: str                           # 来源标识（"static", "000300.SS", "filter:liquidity"）
    metadata: Dict[str, Any]              # 附加信息（权重、行业等）

@dataclass(frozen=True)
class ParamDistribution:                  # 参数分布（用于参数优化）
    component: str                        # "sma_fast" (indicator name)
    param: str                            # "period"
    values: list                          # [5, 10, 15, 20, 25, 30]
    distribution: str | None              # "uniform", "log_uniform"
    low: float | None
    high: float | None

@dataclass(frozen=True)
class ParamConstraint:                    # 参数约束
    expr: str                             # "sma_fast.period < sma_slow.period"
```

> **Order vs ManagedOrder 分离**：Order 是冻结的值对象，Rule 只负责生成意图；
> ManagedOrder 由 OrderBook 持有，跟踪 open → filled/canceled/expired 生命周期。
> 这一分离遵循不可变性原则——策略代码永远无法篡改已提交的订单字段。

> **未实现类型**：`RunContext`（执行上下文，含 as_of_date / trace_id 等）计划在 Phase 3+ 实现；
> `TradeMethod` / `RoundTripTrade`（完整交易统计）计划在 Phase 2+ 实现。

### 4.5 列名引用

组件间通过**宽表列名**直接引用，无需额外绑定语法。Indicator 输出追加为 `mktdata[symbol][name]` 列，Signal 和 Rule 按字符串列名引用：

```python
# Indicator "sma_fast" 的输出追加为列 mktdata["AAPL"]["sma_fast"]
# Signal 通过 inputs={"fast": "sma_fast"} 引用该列
# Rule 通过 signal="golden_cross" 引用 Signal 输出列
```

这种设计保证了引用在运行时总是可验证的——如果列名不存在，会立即抛出 KeyError，而非静默失败。

### 4.6 Broker Protocol 架构：策略与执行分离

策略层（Universe + Signal + Portfolio）只负责"做什么"的声明。策略自身包含 Universe，通过两个可替换 Protocol 与执行环境解耦：

```
策略定义层（Universe + Signal + Portfolio）
         │
         │  不关心下面是回测还是实盘
         │
    ┌────▼──────────────────────────────────────────┐
    │           两个可替换 Protocol                    │
    │                                                │
    │  1. MarketDataProvider ── 数据从哪来？         │
    │  2. Broker             ── 订单/成交/生命周期    │
    └────────────────────────────────────────────────┘
```

`Broker` Protocol 统一了订单提交、成交回报和 bar 生命周期管理，继承自 `OrderRouter` + `FillReceiver` 并添加生命周期钩子：

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MarketDataProvider(Protocol):
    """行情接口：策略获取行情数据的唯一入口"""
    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame: ...
    def get_latest(self, symbol: str) -> pd.Series: ...

@runtime_checkable
class OrderRouter(Protocol):
    """订单接口：策略提交订单的唯一出口"""
    def submit_order(self, order: Order) -> str: ...

@runtime_checkable
class FillReceiver(Protocol):
    """成交接口：成交回报回填到 portfolio 的唯一入口"""
    def get_fills(self) -> list[Fill]: ...

@runtime_checkable
class Broker(OrderRouter, FillReceiver, Protocol):
    """统一 Broker 接口：Engine 的唯一执行依赖"""
    def on_bar_open(self, mktdata, date) -> None: ...   # SimBroker: process pending orders
    def on_bar_close(self, mktdata, date) -> None: ...  # SimBroker: fill market orders
    def get_open_orders(self, symbol=None) -> list[ManagedOrder]: ...
    def cancel_orders(self, symbol, side=None) -> list[ManagedOrder]: ...
    def cap_pending_sells(self, symbol, max_shares) -> None: ...
```

> **设计动机**：Engine 之前通过 `hasattr` duck-typing 调用 SimBroker 特有的方法（如 `process_pending_orders`、`fill_market_orders`），这在引入 `LiveBroker` 时会导致问题。`Broker` Protocol 将这些方法抽象为 `on_bar_open` / `on_bar_close` 生命周期钩子，`SimBroker` 在钩子内委托给具体实现，`LiveBroker` 可以实现为轮询成交或 no-op。

三种运行模式通过注入不同实现切换，策略代码零修改（Universe 在策略内部，不随运行模式变化）：

| 模式 | MarketDataProvider | Broker |
|---|---|---|
| **回测** | `LocalMarketDataProvider` — 加载本地 Parquet | `SimBroker` — 模拟撮合 |
| **Paper Trade** | `AlpacaMarketDataProvider` — Alpaca 行情 | `SimBroker` |
| **实盘** | `AlpacaMarketDataProvider` — Alpaca 行情 | `LiveBroker` — Alpaca API |

Strategy 的 `universe` 字段在构造时设定，`engine.run()` 通过关键字参数注入 Provider：

```python
# 回测模式
engine.run(strategy,
    market=LocalMarketDataProvider(),
    broker=sim_broker,
    start="2023-01-01", end="2024-12-31",
    initial_cash=100_000.0)

# Paper trade（未来）：仅替换行情源
engine.run(strategy,
    market=RealtimeDataProvider(source="websocket"),
    broker=sim_broker, ...)

# 实盘（未来）：两个接口全部替换
engine.run(strategy,
    market=RealtimeDataProvider(source="websocket"),
    broker=live_broker, ...)
```

> **当前阶段**：已实现 `StaticUniverse` + `FilterUniverse`（策略组件）、`LocalMarketDataProvider`（历史数据）、`SimBroker`（模拟撮合，实现 `Broker` Protocol）。`IndexUniverse`（Point-in-Time）留给 Phase 2。Paper trade 和实盘的实现留给后续 Phase，但 `Broker` Protocol 从第一天就定义好，确保架构不需要重构。

---

## 5. 功能模块

### 5.1 数据层 (oxq.data)

**统一因子模型**：框架将所有外部数据统一视为 Factor（因子），无论其原始来源是行情、财务报表、宏观经济指标还是新闻舆情。从策略视角，PE ratio、GDP 增速、舆情分数与 RSI 本质相同——都是"某个时间点上的一个数值"。所有 factor 最终以列的形式汇入宽表 `mktdata`，参与 Indicator → Signal → Rule 管道。

**核心原则**：

1. **一切皆 Factor**：财务数据（PE、ROE、营收）、宏观数据（GDP、CPI、利率）、另类数据（新闻舆情、社交媒体情绪）在进入宽表前均完成预处理，以数值型 factor 的形式存在
2. **Point-in-Time 对齐**：factor 必须携带时间属性，按数据的实际可用日（announce_date）而非报告期（period_date）对齐进宽表，防止前视偏差
3. **频率打平**：低频 factor（季度财报、月度宏观）通过 forward-fill 对齐到日频宽表；事件型数据（新闻）需先聚合到日频再注入
4. **全局数据广播**：无 symbol 维度的数据（宏观指标）广播到全 universe 的每个 symbol，在宽表中作为同名列存在，Signal 层可自然地将其用作全局条件

### 5.2 Universe 构建 (oxq.universe)

Universe 决定"每个时间截面上，哪些 symbol 参与计算"。缺少显式 Universe 管理会导致 survivorship bias——用当前成分股回测历史数据，高估策略收益。

**UniverseProvider Protocol**：

```python
class UniverseProvider(Protocol):
    """标的池接口：确定参与计算的 symbol 集合"""
    def get_universe(self, as_of_date: str) -> UniverseSnapshot: ...
    def get_history(self, start: str, end: str) -> list[UniverseSnapshot]: ...
```

**三种内置实现**：

| 实现 | 说明 | 适用场景 |
|------|------|----------|
| `StaticUniverse` | 固定 symbol 列表，不随时间变化 | 单标的策略、手动指定标的池 |
| `IndexUniverse` | 指数成分股，支持 Point-in-Time 历史成分查询 | 沪深300轮动、行业指数策略 |
| `FilterUniverse` | 基于因子条件动态过滤（流动性、市值、ST 排除等） | 全市场因子策略 |

```python
# 静态列表
universe = StaticUniverse(["600519.SS", "000858.SZ", "000333.SZ"])

# 指数成分股（Point-in-Time，回测安全）
universe = IndexUniverse("000300.SS", point_in_time=True)

# 条件过滤：排除 ST、日均成交额 > 1000 万
universe = FilterUniverse(
    base=IndexUniverse("000985.SS"),  # 中证全指作为基础池
    filters=[
        ExcludeST(),
        MinTurnover(threshold=10_000_000),
    ]
)
```

**Tool 调用**：
```
→ universe_set(strategy="momentum", type="index", code="000300.SS")
→ universe_inspect(strategy="momentum", as_of_date="2023-06-30")  # 查看某日成分
→ universe_history(strategy="momentum", start="2020-01-01", end="2024-12-31")  # 成分变动历史
```

> **Point-in-Time 的重要性**：回测中使用当天实际生效的成分股（而非最新成分股）是消除 survivorship bias 的关键。`IndexUniverse` 的 `point_in_time=True` 确保在 2020 年的回测日使用 2020 年的成分股数据，而非 2024 年的。

### 5.3 指标库 (oxq.indicators)

**契约优先，而非约定优先**：

open-xquant 采用契约优先的设计——统一的 Protocol 接口取代松散的函数约定。原因是 Agent First 设计哲学——约定意味着"文档里写了但代码不强制"，正好是 AI 幻觉的温床。统一的 Protocol 让 AI 的选择空间收窄到只有正确的做法。

```python
from typing import Protocol

class Indicator(Protocol):
    """所有指标必须实现的契约"""
    name: str

    def compute(self, mktdata: pd.DataFrame, **params) -> pd.Series:
        """
        输入：mktdata 宽表（OHLCV + 已有指标列）
        输出：与 mktdata 等长的 Series，将追加为新列
        约束：纯函数，不得依赖外部状态，不得修改 mktdata
        """
        ...

# 内置指标实现示例
class SMA:
    name = "SMA"

    def compute(self, mktdata: pd.DataFrame, column: str = "close", period: int = 20) -> pd.Series:
        return mktdata[column].rolling(period).mean()

# 接入第三方库（如 ta-lib）需要写 adapter
class TalibSMA:
    """ta-lib SMA 的适配器"""
    name = "SMA_talib"

    def compute(self, mktdata: pd.DataFrame, column: str = "close", timeperiod: int = 20) -> pd.Series:
        import talib
        return pd.Series(talib.SMA(mktdata[column].values, timeperiod=timeperiod), index=mktdata.index)
```

> **设计取舍**：接入第三方库需要写 adapter 是有意为之的代价。adapter 本身是一次性工作，且内置指标库会覆盖最常用的指标。对 AI Agent 而言，统一接口带来的确定性远比"拿来即用"更重要。

**指标评估**：Indicator 是对市场某个侧面的度量（趋势、波动率、估值等），应在纳入策略前独立评估其度量质量。评估方法包括：构造 symmetric filter（基于完整数据的"完美后见"滤波器）并度量 indicator 与它的距离；对不同参数化做稳健性检验（小的参数变化应导致小的输出变化）；检验 indicator 在不同 bar 生成起点下的稳定性。

如果 indicator 无法通过独立评估，应在此阶段淘汰，而非等到完整回测后才发现问题。引擎的分阶段执行能力（`run_through="indicator"`）为此提供了架构支持。

**内置指标**：

| 类别 | 指标 | 说明 |
|------|------|------|
| 趋势 | SMA, EMA, WMA, DEMA, TEMA | 移动平均系列 |
| 动量 | RSI, ROC, PPO, Momentum, NdayReturn, LogReturn | 动量与收益指标 |
| MACD | MACDLine, MACDSignal, MACDHistogram | MACD 拆分为三个独立指标（各返回一列） |
| 波动 | BollingerUpper, BollingerLower, ATR, RollingVolatility, RollingMDD | 波动率指标 |
| 成交量 | OBV, VWAP, MFI | 量价指标 |
| 方向 | ADX, AROON, StochK, CCI | 趋势强度与方向指标 |
| 因子 | Ratio | 比率因子 |

> **指标拆分原则**：Indicator Protocol 约束 `compute()` 返回单个 `pd.Series`（一列）。多输出指标必须拆分为独立类：BBands 拆为 `BollingerUpper` + `BollingerLower`，MACD 拆为 `MACDLine` + `MACDSignal` + `MACDHistogram`。这不是妥协，而是 Protocol 约束的必然结果——每个指标对应宽表中的一列。

**Agent 体验**：Agent 不需要知道具体实现，只需说"给这个策略加一个 20 日 RSI 指标"，Tool 处理其余一切。

**Indicator 返回值的两条路径**：

| 路径 | 条件 | 去向 |
|---|---|---|
| A（主路径） | 返回与 mktdata 等长的时间序列 | 追加为 mktdata 的新列 |
| B（例外） | 返回不等长对象或非时间序列 | 存入独立的附加存储（`extras`） |

路径 A 是绝大多数场景。路径 B 极少使用（例如某些统计检验返回标量或不等长序列），此时 Signal/Rule 需从 `extras` 中显式读取，而非从 mktdata 列引用。框架优先引导用户走路径 A，路径 B 作为逃生舱口保留。

**预计算指标（因子）的接入**：

在实际量化研究中，指标/因子往往是预先计算好的（如离线因子库）。框架支持两种接入方式：

**方案一：直接注入宽表，不注册为 Indicator**

Signal/Rule 仅按列名引用数据，不关心列的来源。只要不注册同名 Indicator，预计算列不会被覆盖。

```python
# 1. 预计算结果直接附加到行情数据
mktdata["alpha_momentum"] = load_factor("momentum", symbols, dates)

# 2. 策略定义跳过 add_indicator，直接从 Signal 开始
strategy.add_signal("momentum_high", Threshold,
    inputs={"value": "alpha_momentum"},  # 直接引用列名
    params={"threshold": 0.8, "relationship": "gt"})
```

优点：最简单，零重算。缺点：策略定义不完整，无法对该指标参数做参数优化。

**方案二：注册为缓存读取 Indicator**

注册一个符合 Indicator Protocol 的类，其 `compute` 方法仅从缓存/因子库读取预计算结果：

```python
class CachedFactor:
    """从因子库加载预计算因子，符合 Indicator Protocol"""
    name = "CachedFactor"

    def compute(self, mktdata: pd.DataFrame, factor_name: str = "", **kwargs) -> pd.Series:
        return load_factor(factor_name, mktdata.index)

strategy.add_indicator("alpha_momentum", CachedFactor,
    params={"factor_name": "momentum"})
```

优点：策略定义完整，兼容参数优化（可优化 factor_name 的选择或后处理参数）。缺点：需自行管理缓存一致性。

> **推荐**：研究阶段用方案一快速迭代；生产阶段用方案二保证策略定义的完整性和可审计性。

### 5.4 信号生成器 (oxq.signals)

**信号的本质**：Signal 是对某个时间点的**方向性预测**（directional prediction），而非交易指令。信号描述"交易的欲望"——策略可能因仓位限制、风控规则或再平衡周期等原因选择不执行信号。将信号与行动分离，使得信号的预测力可以独立于执行假设进行评估。

提供 7 种内置信号类型（均为 per-symbol 信号，生成布尔或数值列）：

| 信号类型 | 说明 | 状态 |
|----------|------|------|
| `Crossover` | 两条线交叉（上穿检测） | ✅ 已实现 |
| `Threshold` | 超过/低于阈值 | ✅ 已实现 |
| `Comparison` | 两个值比较 | ✅ 已实现 |
| `Formula` | 自定义布尔公式 | ✅ 已实现 |
| `Peak` | 峰值/谷值检测 | ✅ 已实现 |
| `Timestamp` | 时间条件触发 | ✅ 已实现 |
| `Composite` | 多信号 AND/OR 组合 | ✅ 已实现 |

所有信号均为 per-symbol 操作，生成布尔或数值列。原来的截面权重信号（`EqualWeight`、`TopNRanking`、`RiskParity`）已迁移到 PortfolioOptimizer 层（`EqualWeightOptimizer`、`TopNRankingOptimizer`、`RiskParityOptimizer`），负责截面权重分配。

**信号评估**：信号触发后的前瞻收益分布（forward return distribution）是评估信号质量的核心工具。以信号触发时刻 t₀ 为锚点，统计 t₁...tₙ 期间的收益分布——无需任何执行假设即可判断信号是否具有预测力。引擎的分阶段执行能力（`run_through="signal"`）为此提供了架构支持。稳健的信号应在相邻参数组合中呈现"稳定区域"（stable region）：相似的参数产生相似的正向或负向预期。如果正向预期在参数空间中随机散布，说明假设本身可能有问题。

### 5.5 交易规则 (oxq.rules)

Rule 返回 **RuleResult**，而非 Order。RuleResult 可携带 weights（目标权重调整）、constraints（约束条件）、target_positions（目标持仓）或 hold（冻结后续规则）。Rules 不属于 Strategy，而是传给 `Engine.run(rules=[...])`。

Engine 执行管道中，Rule 分为两个阶段：

```
执行顺序（每个时间步）：
Pre-trade Rule  → 交易前风控检查（如回撤熔断、日内亏损限制）
                  返回 RuleResult，可冻结后续交易
Portfolio       → PortfolioOptimizer 将信号转化为目标持仓
Trading Algo    → 生成订单
Broker          → 提交并执行订单
Post-trade Rule → 交易后处理（如止损止盈挂单、追踪止损更新）
                  返回 RuleResult
```

**RuleResult**：

```python
@dataclass(frozen=True)
class RuleResult:
    """Rule 的统一返回类型"""
    weights: SparseVector | None = None          # 目标权重调整
    constraints: dict[str, Any] | None = None    # 约束条件
    target_positions: SparseVector | None = None  # 目标持仓
    hold: bool = False                            # True 时冻结后续所有规则
```

**已实现的风控规则**（Pre-trade Rule）：

| 规则 | 参数 | 逻辑 |
|------|------|------|
| `MaxDrawdownRisk` | max_drawdown | 组合回撤超阈值时返回 hold=True |
| `DailyLossLimitRisk` | max_daily_loss | 日内亏损超阈值时返回 hold=True |

**已实现的订单规则**（Post-trade Rule）：

| 规则 | 参数 | 逻辑 |
|------|------|------|
| `StopLossRule` | threshold | 亏损超阈值时止损卖出 |
| `TakeProfitRule` | threshold | 盈利超阈值时止盈卖出 |
| `TrailingStopRule` | trail_pct | 追踪止损 |

**已实现的出场规则**：

| 规则 | 参数 | 卖出逻辑 |
|------|------|----------|
| `ExitRule` | fast, slow | 快线跌破慢线时全仓卖出 |

**已实现的再平衡规则**：

| 规则 | 参数 | 逻辑 |
|------|------|------|
| `RebalanceRule` | weight_col, frequency | 按权重信号列定期再平衡 |

### 5.5.1 PortfolioOptimizer Protocol

PortfolioOptimizer 取代了旧的 sizing 函数（如 clip_to_max_position、clip_to_pct_equity、os_equal_weight 等），提供统一的组合优化接口：

```python
@runtime_checkable
class PortfolioOptimizer(Protocol):
    """组合优化器：将信号权重转化为目标持仓"""
    def optimize(self, weights: SparseVector, portfolio: Portfolio,
                 constraints: dict[str, Any] | None = None) -> SparseVector:
        """返回优化后的目标权重"""
        ...
```

PortfolioOptimizer 属于 Portfolio 阶段，在 Signal 之后、Rule 之前执行。

**规则评估**：Risk rule 的止损阈值应基于 MAE 分布的统计分析，而非主观设定。Exit rule 分为信号驱动（如反向交叉）和经验驱动（如基于 MAE/MFE 设置的止损/止盈），后者需要从交易统计中实证推导。

**Rule Burden**：规则数量是过拟合的重要信号。每添加一条规则都会增加策略的自由度。特别危险的模式是：在参数优化后因结果不满意而添加新规则，这本质上是 data snooping。`strategy_validate` 会报告策略的规则数量和总自由度，作为过拟合风险的参考指标。

**仓位管理的两阶段原则**：研究阶段使用最小化仓位（如固定 1 手），目标是验证信号和规则的有效性，而非优化执行。生产阶段的仓位管理应基于微观结构分析和实盘交易统计来校准。

### 5.6 组合管理 (oxq.portfolio)

- **订单请求与生命周期分离**：`Order`（frozen 值对象）→ 提交给 Broker → `ManagedOrder`（OrderBook 管理生命周期：open → filled/partial/canceled/expired）
- **订单簿（OrderBook）**：管理 `ManagedOrder` 集合，支持 market/limit/stop/stop_limit/trailing_stop 订单类型，提供 `add()`、`fill()`、`get_open_orders()`、`cancel_orders()`、`cap_pending_sells()` 方法
- **Portfolio**：管理持仓状态（`positions: dict[str, Position]`）、现金余额（`cash`）、当前价格快照（`bar_prices`）、组合总值（`total_value(prices)`）
- **成交记账**：`Fill`（frozen）记录每笔成交的价格和手续费，引擎通过 `_apply_fill()` 更新 Portfolio 状态
- **执行报告（ExecutionReport）**：对比模拟成交与实盘成交（`FillComparison`），计算价格滑点和费用差异

**Round-trip Trade 定义**：交易统计（胜率、盈亏比、MAE/MFE 等）的计算结果取决于如何定义一笔"交易"的起止。框架默认使用 **flat-to-reduced** 方法——从首次增仓标记交易开始，任何减仓标记交易结束。该方法与券商对账单一致，便于回测与实盘结果对照。支持的方法见 `TradeMethod` 枚举。

### 5.7 执行引擎 (oxq.core.engine)

Engine 是通用策略执行引擎，驱动 Indicators → Universe → Signal → Portfolio → Pre-trade Rule → Trading Algorithm → Broker → Post-trade Rule 管道。它是 **provider-agnostic** 的——不知道自己在运行回测、模拟盘还是实盘，区别仅在于接入的 Protocol 实现：

| 模式 | MarketDataProvider | Broker |
|------|-------------------|--------|
| 回测 | `LocalMarketDataProvider` | `SimBroker` |
| 模拟盘 | `AlpacaMarketDataProvider` | `SimBroker` |
| 实盘 | `AlpacaMarketDataProvider` | `LiveBroker` |

**`Engine.run()` 签名**：

```python
def run(self, strategy, market, broker,
        start, end, initial_cash=100_000.0,
        rules=None, run_through=None, tracer=None) -> RunResult
```

- `run_through="indicator"` / `"signal"` 支持分阶段运行，用于逐组件评估
- `tracer: DefaultTracer | None` — 可选追踪器，传入后自动记录每个阶段的 TraceSpan
- `SimBroker`（`oxq.trade`）实现 `Broker` Protocol，在每个 bar 的 close 价格模拟撮合
- `RunResult`（`oxq.portfolio.analytics`）包含 portfolio、trades（`list[Fill]`）、equity_curve、mktdata、benchmark_prices，提供 `total_return()`、`sharpe_ratio()`、`max_drawdown()`、`annualized_return()`、`annualized_volatility()`、`calmar_ratio()`、`sortino_ratio()`、`daily_returns()`、`monthly_returns()`、`drawdown_series()` 方法
- 策略评估达标检查：通过 `engine_results` tool 自动与 `strategy.objectives` 对比，输出各指标的达标状态（pass/fail）

### 5.8 参数优化 (oxq.optimize)

```python
# SDK 用法
paramset = ParamSet("sma_tuning")
paramset.add_distribution("sma_fast", "period", values=range(5, 30, 5))
paramset.add_distribution("sma_slow", "period", values=range(20, 100, 10))
paramset.add_constraint("sma_fast.period < sma_slow.period")

# 网格搜索
results = grid_search(strategy, paramset, data, metric="sharpe_ratio")

# 滚动前推分析
wfa_results = walk_forward(
    strategy, paramset, data,
    train_period="2Y",          # 训练窗口
    test_period="6M",           # 测试窗口
    step="3M",                  # 滚动步长
    optimize_metric="sharpe_ratio",
    anchored=False              # 滚动窗口 vs 扩展窗口
)

# 统计检验（防过拟合）
deflated_sr = deflated_sharpe(results, num_trials=len(paramset))
profit_hurdle = profit_hurdle_test(results, num_trials=len(paramset))
haircut_sr = haircut_sharpe(results, method="holm")
```

**Tool 调用**：
```
→ optimize_define_paramset(strategy="momentum", distributions=[...], constraints=[...])
→ optimize_run_walk_forward(strategy="momentum", paramset="sma_tuning", train="2Y", test="6M")
→ analysis_deflated_sharpe(results_id="wfa_001")
→ analysis_profit_hurdle(results_id="wfa_001")
```

### 5.9 统计检验 (oxq.optimize.validation)

| 方法 | 用途 |
|------|------|
| `deflated_sharpe()` | 校正多重比较后的 Sharpe Ratio |
| `haircut_sharpe()` | 对 Sharpe Ratio 施加 haircut |
| `profit_hurdle()` | 最低利润门槛检验 |
| `degrees_of_freedom()` | 自由度估计 |
| `white_reality_check()` | Bootstrap 检验策略收益是否显著优于基准 |
| `k_fold_cv()` | k 折交叉验证（时间序列适配版） |
| `cscv()` | 组合对称交叉验证（Combinatorially Symmetric Cross Validation） |
| `oos_deterioration()` | 度量样本外退化（IS 与 OOS 的绩效差异） |

### 5.10 多策略编排 (oxq.orchestrator)

```python
orchestrator = StrategyOrchestrator()
orchestrator.add("momentum", strategy_a, weight=0.4)
orchestrator.add("mean_reversion", strategy_b, weight=0.3)
orchestrator.add("trend_following", strategy_c, weight=0.3)

# 策略间约束
orchestrator.set_total_exposure(max=1.5, min=0.3)   # 总敞口限制
orchestrator.set_correlation_limit(max=0.7)          # 策略相关性限制
orchestrator.set_capital_allocation("risk_parity")   # 资金分配方式

result = orchestrator.run(context, providers)
```

### 5.11 可观测性 (oxq.observe)

```python
# 执行追踪：每步自动记录输入/输出/耗时
@dataclass(frozen=True)
class TraceSpan:
    trace_id: str
    span_id: str
    parent_id: str | None
    component: str          # "indicator:sma_fast", "signal:golden_cross", "rule:enter"
    inputs: dict[str, Any]  # 输入快照
    output_summary: dict[str, Any]  # 输出摘要（便于序列化和审计）
    started_at: str
    ended_at: str
    duration_ms: float      # 耗时（毫秒）
    status: str             # "ok" | "error" | "skipped"
    error: str | None       # 错误信息（status="error" 时）

# 审计日志：保证可复现
@dataclass(frozen=True)
class AuditRecord:
    run_id: str
    strategy_name: str
    strategy_config: dict[str, Any]   # 完整策略配置快照
    start_date: str
    end_date: str
    initial_cash: float
    trace_spans: tuple[TraceSpan, ...]  # 完整执行追踪
    mktdata_hash: str                   # 行情数据哈希
    trades_hash: str                    # 交易记录哈希
    equity_hash: str                    # 权益曲线哈希
    result_hash: str                    # 综合结果哈希（验证确定性）
    created_at: str
```

`DefaultTracer` 提供生命周期钩子：`on_run_start()`、`on_indicator()`、`on_signal()`、`on_rule()`、`on_run_end()`。传入 `Engine.run(tracer=tracer)` 后，引擎在执行各阶段时自动调用对应钩子，生成 TraceSpan。`AuditRecord.build()` 从 tracer 和 RunResult 自动构建审计记录，包含四个维度的哈希用于确定性验证。

**策略监控与分析**：

- **StrategyMonitor**：监控策略运行状态，检测绩效偏离和异常期（`BadPeriod`）
- **MarketStateDetector**：基于波动率检测市场状态（高波/低波/正常），分析策略在不同市场状态下的表现差异
- **ExperimentLog / Experiment**：结构化实验日志，记录策略研究过程中的假设（hypothesis）、观察（observation）、结论（conclusion），支持从策略运行结果自动生成实验记录

**Tool 调用**：
```
→ observe_trace(audit_id="...")                           # 查看执行追踪
→ observe_audit_log(run_id="...", strategy="...")          # 查看/创建审计日志
→ observe_audit_compare(audit_id_a="...", audit_id_b="...") # 对比两次审计
→ observe_monitor_create(run_id="...", benchmark="SPY")   # 创建策略监控
→ observe_monitor_summary(monitor_id="...")               # 查看监控摘要
→ observe_detect_market_state(run_id="...", symbols=["AAPL"])  # 检测市场状态
→ observe_experiment_create(name="research_v1")           # 创建实验日志
→ observe_experiment_add(log_id="...", ...)               # 添加实验记录
```

### 5.12 交易执行 (oxq.trade)

- 完整订单簿：支持 limit/stop/trailing_stop，追踪订单生命周期
- 执行反馈闭环：实际成交 → 更新 portfolio → 记录 slippage
- 多交易所支持：SSE, SZSE, NYSE, NASDAQ, HKEX
- 券商 API 抽象：定义 Executor Protocol，社区可实现各券商适配器

策略定义与执行环境的分离详见 [4.6 三接口架构](#46-三接口架构策略与执行分离)。

---

## 6. Tool 定义与分发

### 6.1 Tool 定义（oxq.tools）

Tool 定义是框架的核心资产之一，与传输协议无关。每个 Tool 包含：名称、参数 schema、语义描述、调用逻辑。Tool 定义在 `src/oxq/tools/` 中实现，可通过多种方式触发：Coding Agent 直接 `import` 调用、MCP 客户端通过 MCP 协议调用、平台方通过 REST/gRPC 等自建接口调用。

**命名规范**：Tool 名称统一使用 snake_case（如 `strategy_create`），不使用点号分隔。这确保与 OpenAI function calling 等 LLM API 的命名约束（`^[a-zA-Z0-9_-]+$`）兼容。

**工具清单**：

| 工具组 | 工具名 | 说明 |
|--------|--------|------|
| **strategy** | `strategy_create` | 创建策略 |
| | `strategy_add_indicator` | 添加指标 |
| | `strategy_add_signal` | 添加信号 |
| | `strategy_add_rule` | 添加规则 |
| | `strategy_inspect` | 查看策略详情 |
| | `indicator_list` | 列出所有可用指标类型 |
| | `indicator_describe` | 查看某指标类型的参数说明 |
| **data** | `data_load_symbols` | 加载标的行情数据 |
| | `data_list_symbols` | 列出已有标的 |
| | `data_inspect` | 查看数据摘要（时间范围、缺失值等） |
| | `factor_download` | 下载宏观因子数据（World Bank） |
| | `factor_list` | 列出已有因子 |
| | `factor_inspect` | 查看因子数据摘要 |
| **universe** | `universe_set` | 设置 Universe（静态列表/指数/过滤条件） |
| | `universe_list_indexes` | 列出可用的指数代码 |
| | `universe_inspect` | 查看 Universe 成分快照 |
| | `universe_history` | 查看 Universe 成分变动历史 |
| **engine** | `engine_run` | 运行策略（回测） |
| | `engine_results` | 获取运行结果和绩效指标 |
| | `engine_trade_list` | 查看交易记录 |
| **optimize** | `paramset_create` | 定义参数搜索空间 |
| | `paramset_inspect` | 查看参数空间定义 |
| | `grid_search` | 网格搜索 |
| | `walk_forward` | 前推分析 |
| | `cross_validate` | 时间序列交叉验证 |
| | `overfit_analysis` | 过拟合分析（Deflated Sharpe 等） |
| **observe** | `observe_trace` | 查看执行追踪（TraceSpan） |
| | `observe_audit_log` | 查看/创建审计日志 |
| | `observe_audit_compare` | 对比两次审计记录 |
| | `observe_monitor_create` | 创建策略监控 |
| | `observe_monitor_summary` | 查看监控摘要 |
| | `observe_detect_market_state` | 检测市场状态 |
| | `observe_performance_by_state` | 按市场状态分析绩效 |
| | `observe_experiment_create` | 创建实验日志 |
| | `observe_experiment_add` | 添加实验记录 |
| | `observe_experiment_add_from_strategy` | 从策略运行自动添加实验记录 |
| | `observe_experiment_list` | 列出实验记录 |
| **live** | `live_connect` | 连接券商（Alpaca） |
| | `live_account` | 查看账户信息 |
| | `live_positions` | 查看实盘持仓 |
| | `live_bars` | 获取实时行情 |
| | `live_generate_orders` | 按目标权重生成订单计划 |
| | `live_submit_order` | 提交订单 |
| | `live_order_status` | 查询订单状态 |
| | `live_open_orders` | 查看未成交订单 |
| | `live_cancel_order` | 撤单 |

**未实现的 Tool**（待后续 Phase）：

- `strategy_list` — 列出所有策略
- `strategy_validate` — 验证策略配置（含 rule burden 检查）
- `strategy_export` / `strategy_import` — 导出/导入策略
- `data_query` — 查询特定数据
- `engine_compare` — 对比多次运行
- `observe_replay` — 重放历史执行
- `orchestrator_*` — 多策略编排（Phase 4）

**设计原则**：

1. **原子性**：每个 tool 做一件事，返回结构化 JSON
2. **幂等性**：相同输入产生相同输出（除 execute 外）
3. **可组合**：tools 之间通过 ID 引用关联（strategy_id, run_id, paramset_id）
4. **错误友好**：返回清晰的错误信息 + 建议的修复动作
5. **渐进披露**：简单场景用少量参数，复杂场景可展开全部参数
6. **Thin Wrapper**：Tool 是 SDK 的薄封装，不得包含业务逻辑。每个 tool 函数体只做三件事：参数解析 → 调用 `oxq` SDK → 格式化返回。所有计算、状态管理、规则执行等逻辑必须实现在 `src/oxq/` 中，Tool 层只负责接口适配

### 6.2 MCP Server（可选分发层）

MCP Server 是 `oxq.tools` 的 MCP 协议适配，用于支持不能执行代码的 AI 客户端（Claude Desktop、Windsurf 等）。MCP Server 本身不包含业务逻辑，只做两件事：MCP 协议适配、从 `oxq.tools.registry` 自动注册所有 Tool。

会话状态管理由 `oxq.tools.session` 模块负责（文件持久化，跨 MCP 进程重启保持状态）。MCP Server 目录下只有 `server.py` 一个文件，无子目录。

> **注意**：Coding Agent（如 Claude Code、Cursor）直接 `import oxq` 即可，不需要 MCP Server。MCP 的价值是**分发渠道**——让本地 SDK 能被非 Coding AI 客户端开箱即用。

```python
# mcp_server/server.py — 薄适配层
from mcp.server.fastmcp import FastMCP
from oxq.tools import registry

mcp = FastMCP("open-xquant")

# 自动从 oxq.tools 注册所有 tool 到 MCP server
for tool_def in registry.all_tools():
    mcp.tool(name=tool_def.name, description=tool_def.description)(tool_def.fn)
```

---

## 7. Agent Skills

每个 skill.md 描述一个完整的 Agent 工作流，指导 AI Agent 如何组合 tools 完成任务。

> **Tool 引用是协议无关的**：skill 中引用的 tool 名称（如 `universe_*`、`strategy_*`）是 `oxq.tools` 中定义的协议无关 Tool。Coding Agent 通过 `import oxq.tools` 调用等价函数，MCP 客户端通过 MCP 协议调用——Tool 名称和语义完全一致，仅传输方式不同。

### 7.1 strategy-builder.md

策略构建的完整工作流，覆盖从约束确认到回测评估的 6 个阶段（Phase 0–5）：

```markdown
---
name: strategy-builder
description: 指导 Agent 构建量化交易策略并进行回测评估
tools_required: [strategy_*, engine_*, data_*, universe_*]
---

Phase 0: 业务约束 — 明确资金、品种、频率
Phase 1: 基准与目标 — 设定可量化的 total_return / sharpe_ratio / max_drawdown 目标
Phase 2: 假设 — 5 要素可测试假设（信号、品种、方向、依据、退出）
Phase 3: 数据准备 — data_load_symbols → data_inspect → universe_set
Phase 4: 逐层构建 — strategy_create → add_indicator → add_signal → add_rule → inspect
Phase 5: 回测与达标检查 — engine_run → engine_results → engine_trade_list（三个工具必须按顺序全部调用）

## 红线规则
- 不替用户编造假设、约束、目标
- 不在回测后添加新规则（Rule Burden）
- 规格变更必须记录
```

### 7.2 parameter-tuner.md

```markdown
---
name: parameter-tuner
description: 指导 Agent 进行参数优化并验证结果的统计显著性
tools_required: [optimize.*, analysis.*]
---

## 工作流

1. 定义参数空间：optimize_define_paramset
2. 运行前推分析：optimize_run_walk_forward（优先于 grid_search）
3. 统计检验：
   a. analysis_deflated_sharpe → SR 是否统计显著？
   b. analysis_profit_hurdle → 收益是否超过随机？
4. 如果不显著 → 建议简化策略（减少参数）或增加数据
5. 如果显著 → 输出最优参数 + 样本外表现

## 红线规则
- 永远不要只看 in-sample 结果
- 参数组合 > 100 时必须做 Deflated Sharpe 校正
- Walk-forward 窗口至少覆盖 2 个完整市场周期
- 不要在参数优化后添加新规则——这是 Rule Burden，本质上是 data snooping
```

### 7.3 其他 Skills

| Skill | 状态 | 核心工作流 |
|-------|------|-----------|
| `data-explorer.md` | 已实现 | 检查本地数据 → 下载行情/宏观因子 → 质量检查 |
| `backtest-runner.md` | 已重定向 | → strategy-builder（回测集成在策略构建流程中） |
| `universe-builder.md` | 模板 | Universe 构建专用（Phase 2+） |
| `parameter-tuner.md` | 模板 | 参数优化 + 统计检验（Phase 2+） |
| `performance-reviewer.md` | 模板 | 多维绩效分析 + 归因 + 基准对比 |
| `risk-analyzer.md` | 模板 | 回撤分析 + 压力测试 + 尾部风险 |
| `trade-executor.md` | 模板 | 生成订单 → 确认执行 → 监控成交（Phase 3+） |
| `strategy-monitor.md` | 模板 | 实盘监控 + 偏离检测 + 风控预警（Phase 3+） |

---

## 8. 技术选型

| 决策 | 选择 | 理由 |
|------|------|------|
| 语言 | Python 3.12+ | AI 生态最丰富 |
| 类型系统 | dataclass(frozen=True) + Protocol | 不可变 + 鸭子类型 |
| 金融精度 | Decimal | 避免浮点误差 |
| 时间序列 | pandas DataFrame/Series | Indicator Protocol 标准输入输出类型 |
| 核心依赖 | pandas, numpy | 向量化计算基础设施 |
| 可选依赖 | scipy (optimize), ta-lib (指标加速) | 仅在特定模块引入 |
| 可选依赖 | mcp (Python) | MCP Server 分发时需要，官方 Python SDK |
| 配置格式 | YAML + JSON Schema | Agent 友好（结构化），人也可读 |
| 构建工具 | hatch / uv | 现代 Python 项目管理 |
| 测试 | pytest | 标准选择 |
| 文档 | mkdocs-material | 适合开源项目 |

---

## 9. 实现路线

### Phase 1: 核心引擎 + SDK (MVP) ✅ 已完成
- `oxq.core`: Strategy (Universe + Signal + Portfolio), Engine (Indicator → Universe → Signal → Portfolio → Rule → Broker pipeline), types (Order, Fill, Portfolio, Position, RuleResult, Protocol 定义)
- `oxq.universe`: UniverseProvider Protocol, StaticUniverse, FilterUniverse
- `oxq.indicators`: SMA（其余指标文件已创建，待实现）
- `oxq.signals`: Crossover（其余信号文件已创建，待实现）
- `oxq.rules`: ExitRule, RuleResult
- `oxq.portfolio`: Portfolio, Position, RunResult + analytics (total_return, sharpe_ratio, max_drawdown)
- `oxq.core.engine`: 通用执行引擎（provider-agnostic），支持 `run_through` 分阶段执行
- `oxq.trade`: SimBroker（模拟撮合，同时实现 OrderRouter + FillReceiver）
- `oxq.data`: LocalMarketDataProvider, YFinanceDownloader, AkShareDownloader, WorldBank 因子
- `oxq.tools`: data_* (6) + universe_* (4) + strategy_* (5) + engine_* (3) = 18 个 tool
- `oxq.tools.session`: 文件持久化的会话状态（跨 MCP 进程重启保持策略和运行结果）
- `mcp_server`: FastMCP 适配层，自动注册所有 tool
- `skills/`: strategy-builder.md（已实现 6 阶段流程）, data-explorer.md（已实现）
- **目标**: ✅ Coding Agent / 开发者可以通过 SDK 构建 SMA 策略并回测

### Phase 2: 参数优化 + 统计检验 + 指标扩展 ✅ 大部分已完成
- `oxq.indicators`: 扩展至 27 个内置指标（SMA, EMA, WMA, DEMA, TEMA, RSI, MACDLine/Signal/Histogram, ROC, PPO, CCI, BollingerUpper/Lower, ATR, OBV, VWAP, MFI, ADX, AROON, StochK, Momentum, NdayReturn, LogReturn, RollingVolatility, RollingMDD, Ratio）
- `oxq.signals`: 新增 Comparison、Composite、Formula、Peak、Threshold、Timestamp per-symbol 信号（原 EqualWeight/TopNRanking/RiskParity 已迁移至 PortfolioOptimizer 层）
- `oxq.rules`: 新增 StopLossRule、TakeProfitRule、TrailingStopRule、RebalanceRule、MaxDrawdownRisk、DailyLossLimitRisk、RuleResult、PortfolioOptimizer Protocol
- `oxq.optimize`: ParameterSet、GridSearch、WalkForward、TimeSeriesCV、过拟合分析
- `oxq.tools`: paramset_create、grid_search、walk_forward、cross_validate、overfit_analysis、paramset_inspect
- **未完成**: `IndexUniverse`（Point-in-Time）
- **目标**: ✅ Agent 可以优化参数并验证统计显著性

### Phase 3: 交易执行 + 可观测性 🔄 部分已完成
- `oxq.observe`: DefaultTracer + TraceSpan（执行追踪）、AuditRecord（审计日志）、StrategyMonitor（策略监控）、MarketStateDetector（市场状态检测）、ExperimentLog（实验日志）
- `oxq.trade`: LiveBroker（Alpaca 实盘）、FeeModel、SlippageModel、OrderGenerator、FillPriceMode
- `oxq.contrib.alpaca`: AlpacaClient、AlpacaMarketDataProvider
- `oxq.tools`: observe_* (11) + live_* (9) = 20 个工具
- `oxq.portfolio`: ExecutionReport（模拟 vs 实盘对比）、ManagedOrder + OrderBook（订单生命周期管理）
- **未完成**: observe_replay（执行重放）、EventBus
- **目标**: 🔄 端到端交易链路可用，可观测性基础完成

### Phase 4: 多策略 + 高级特性
- `oxq.orchestrator`: 多策略编排 + 资金分配
- 高级规则: 追踪止损、风险平价、因子风控
- 更多指标/信号
- `oxq.tools`: orchestrator_* tool 定义
- `skills/`: risk-analyzer.md
- **目标**: 机构级多策略管理能力

---

## 10. 验证方案

### 冒烟测试（SDK 主路径）
```bash
# 安装
pip install -e ".[dev]"

# 单元测试
pytest tests/ -v

# SDK 集成测试：直接 import oxq 构建策略 → 回测 → 查看结果
python examples/ma_crossover.py
```

### MCP Server 补充测试（可选分发路径）
```bash
# 需要安装 MCP 依赖
pip install -e ".[mcp]"

# MCP server 启动
python -m mcp_server.server

# 验证 MCP 协议适配正确
```

### Agent 端到端测试
```
主路径（Coding Agent）：
1. Coding Agent 直接 import oxq
2. 用户: "帮我构建一个 A 股 ETF 动量轮动策略，回测 2020-2024 年"
3. Agent 调用 strategy-builder skill，通过 SDK 依次调用 oxq.tools
4. 验证：策略创建成功 → 回测完成 → 绩效报告生成

补充路径（MCP 客户端）：
1. 启动 MCP server
2. Claude Desktop 等非 Coding 客户端连接 MCP server
3. 同样的用户请求，通过 MCP 协议调用相同的 Tool
4. 验证：结果与 SDK 直接调用一致
```

### 可复现性测试（依赖 observe_replay，待实现）
```
1. 运行策略，记录 trace_id + result_hash
2. 使用 observe_replay(trace_id) 重放
3. 验证 result_hash 完全一致
```

> **当前可用的确定性验证**：通过 `observe_audit_log` 创建审计记录，`observe_audit_compare` 对比两次运行的 `result_hash` 是否一致。完整的执行重放（`observe_replay`）待后续实现。

---

## 参考

- **quantstrat (R)**: indicator → signal → rule 分层模型、paramset 参数优化、walk-forward analysis、Deflated Sharpe Ratio 统计检验、order book 管理
- **xquant.shop**: agent pipeline 架构、immutable specs、provider injection、SparseVector 权重表示
- **Peterson, Brian G. (2017)**: *"Developing & Backtesting Systematic Trading Strategies"* — 假设驱动开发、逐组件评估、信号预测力评估、MAE/MFE 分析、Rule Burden、Walk Forward Analysis、统计检验方法论
