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
│   ├── signals/                    # 信号生成器（交叉、阈值、比较、公式、组合、峰值、时间条件）
│   ├── rules/                      # 交易规则（止损、止盈、追踪止损、回撤熔断、条件退出）
│   ├── portfolio/                  # 组合管理（组合优化器、持仓、订单簿、记账、绩效分析）
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

### 4.1 一切皆组合

本框架的核心建模假设：**量化策略输出的一切皆组合**。即使策略只交易一个标的物，它产出的也是该标的物与现金（CASH）的组合——全仓买入是 `{AAPL: 1.0, CASH: 0.0}`，空仓是 `{CASH: 1.0}`，半仓是 `{AAPL: 0.5, CASH: 0.5}`。

这意味着策略管道的终点始终是一组目标权重，而非单个买卖指令。交易算法负责将当前持仓调整到目标组合，Broker 负责执行。这种统一模型天然覆盖了单标的策略、多标的轮动、行业配置等所有场景，不需要为不同策略类型设计不同的执行路径。

### 4.2 Strategy = Universe + Signal + Portfolio

Strategy 由三个核心组件构成：

- **Universe** — 确定标的池。可以是固定列表（StaticUniverse）、指数成分（IndexUniverse）或基于条件的动态过滤（FilterUniverse）
- **Signal** — 逐 symbol 产出交易意图。输出布尔或分类标签（buy/hold/sell），描述"交易的欲望"而非交易指令
- **Portfolio** — 跨 symbol 组合优化。接收 Signal 输出，通过 PortfolioOptimizer 计算目标权重

Strategy 是纯声明式容器——直接传给 Engine 执行。它始于假设和目标：hypothesis 定义了策略试图捕捉的市场现象，objectives 量化了成功标准，benchmarks 提供了比较的参照系。

**Indicator** 服务于上述三个组件以及 Rule，各模块通过 `required_indicators` 属性声明自己依赖的指标，Engine 负责统一收集和计算。

**Rule** 不属于 Strategy。Rule 的职责是对持仓组合的准入约束和持仓监控，通过 `Engine.run(rules=[...])` 传入。这一分离使得同一个 Strategy 可以在不同的风控规则下执行。

Engine 驱动完整管道：**Indicator → Universe → Signal → Portfolio → Pre-trade Rule → Trading Algorithm → Broker → Post-trade Rule**。

```python
from oxq.core import Engine, Strategy
from oxq.indicators import SMA
from oxq.signals import Crossover
from oxq.portfolio.optimizers import EqualWeightOptimizer
from oxq.rules import ExitRule, StopLossRule
from oxq.universe import StaticUniverse

# 创建信号并声明依赖的指标
crossover = Crossover()
crossover.required_indicators = {
    "sma_fast": (SMA(), {"column": "close", "period": 10}),
    "sma_slow": (SMA(), {"column": "close", "period": 50}),
}

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
    signals={
        "golden_cross": (crossover, {"fast": "sma_fast", "slow": "sma_slow"}),
    },
    portfolio=EqualWeightOptimizer(),
)

# 运行（Provider 决定模式：回测 / 模拟 / 实盘）
# Rules 传给 Engine.run()，不属于 Strategy
engine = Engine()
result = engine.run(strategy,
    market=LocalMarketDataProvider(),
    broker=sim_broker,
    rules=[ExitRule(fast="sma_fast", slow="sma_slow"),
           StopLossRule(threshold=0.05)],
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

### 4.3 组件 Protocol

框架的核心组件均通过 Protocol 定义契约。Indicator 和 Signal 签名相同——都是逐 symbol 的纯函数，输入单个 DataFrame，输出单个 Series。区别在于语义：Indicator 输出连续数值（描述市场状态），Signal 输出离散标签（判断交易意图）。

```python
@runtime_checkable
class Indicator(Protocol):
    """逐 symbol 向量化计算，输出数值列。"""
    name: str
    def compute(self, mktdata: pd.DataFrame, **params) -> pd.Series: ...

@runtime_checkable
class Signal(Protocol):
    """逐 symbol 向量化计算，输出布尔/分类标签。"""
    name: str
    def compute(self, mktdata: pd.DataFrame, **params) -> pd.Series: ...

@runtime_checkable
class PortfolioOptimizer(Protocol):
    """跨 symbol 组合优化，输出目标权重。"""
    name: str
    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]: ...

@runtime_checkable
class Rule(Protocol):
    """逐 bar 有状态评估，输出 RuleResult。"""
    name: str
    def evaluate(
        self, symbol: str, row: pd.Series, portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult: ...
```

PortfolioOptimizer 的 `optimize()` 返回目标权重 `dict[str, float]`，所有权重之和为 1.0（含 CASH）。Rule 的 `evaluate()` 返回 RuleResult（权重覆盖、约束条件、目标仓位、交易冻结），而非 Order。

### 4.4 宽表数据模型

`mktdata` 是按 symbol 索引的宽表集合（`dict[str, pd.DataFrame]`），每个 symbol 对应一张独立的 DataFrame。Indicator、Signal 阶段通过追加列逐步加宽各 symbol 的宽表。每个 DataFrame 携带 `attrs["timezone"]` 和 `attrs["currency"]` 元数据，Engine 统一换算到基准时区（Asia/Shanghai）和基准币种（CNY）。

```
原始行情             Indicator 后              Signal 后
+-----------+       +------------------+      +---------------------+
| open      |       | open             |      | open                |
| high      |       | high             |      | high                |
| low       | ───►  | low              | ───► | low                 |
| close     |       | close            |      | close               |
| volume    |       | volume           |      | volume              |
|           |       | sma_fast  (新增) |      | sma_fast            |
|           |       | sma_slow  (新增) |      | sma_slow            |
|           |       |                  |      | golden_cross (新增) |
+-----------+       +------------------+      +---------------------+
```

各层对 `mktdata` 的操作方式：

| 层 | 数据视角 | 引擎行为 | 模式 |
|---|---|---|---|
| Indicator | per symbol | `compute(df)` 返回 Series，引擎追加为新列 | 纯函数 + 引擎回写 |
| Signal | per symbol | `compute(df)` 返回 Series，引擎追加为新列 | 纯函数 + 引擎回写 |
| Portfolio | cross-sectional | `optimize(signals, indicators)` 返回目标权重 | 截面优化 |
| Rule | per bar × per symbol | `evaluate(symbol, row, portfolio)` 返回 RuleResult | 只读，有状态 |

Indicator/Signal 的 `compute` 方法是纯函数，不修改 mktdata。引擎负责将返回值追加为新列。这保证了组件的可测试性和确定性。

宽表避免了层间数据传递的复杂性。Signal 无需知道 Indicator 的输出格式，只需按列名引用；Rule 同理。所有中间结果在同一张表上可见可查。

### 4.5 执行模型

#### 管道流程

Engine 按阶段逐层推进，每一步接收前一步的输出。回测时 Engine 逐 bar 驱动管道，每个 bar 都产出一个目标组合：

```
Engine.setup() — 向量化阶段:
  Phase 1: Indicator   → 从各组件的 required_indicators 收集，逐 symbol 计算，追加为宽表列
  Phase 2: Signal      → 逐 symbol 计算信号，追加为宽表列

Engine.step(date) — 逐 bar 阶段:
  Phase 3: Portfolio    → PortfolioOptimizer 产出目标权重
  Phase 4: Pre-trade Rule  → 检查约束（回撤熔断、调仓频率等），调整权重或冻结交易
  Phase 5: Trading Algorithm → 目标权重 + 当前持仓 → 生成订单
  Phase 6: Broker       → 提交订单、撮合成交、更新持仓
  Phase 7: Post-trade Rule → 监控持仓（止损、止盈、追踪止损等），触发减仓
  Phase 8: Broker       → 执行减仓订单
```

#### 向量化与逐 bar 状态机

| 阶段 | 计算模式 | 路径依赖 | 调用次数 |
|---|---|---|---|
| Indicator | 向量化 — 对全量时间序列一次计算 | 否 | 每个 symbol 1 次 |
| Signal | 向量化 — 对全量时间序列一次计算 | 否 | 每个 symbol 1 次 |
| Portfolio | 截面 — 对当前 bar 的全 universe 优化 | 否 | 每个 bar 1 次 |
| Rule | 逐 bar 循环 — 状态机模式 | 是 | 每个 bar × 每个 symbol |

Indicator/Signal 的值仅取决于截至 T 时刻的行情数据，与历史交易无关，可向量化一次算完。Rule 的决策取决于当前持仓、挂单、资金等状态，必须逐步推进。Portfolio 每个 bar 产出一组新的目标权重——调仓频率等约束由 Rule 层处理。

**分阶段执行**：引擎支持 `run_through` 参数在任意阶段终止，用于逐组件独立评估：

```python
# 只执行到指标阶段
result = engine.run(strategy, market, broker, run_through="indicator")

# 只执行到信号阶段
result = engine.run(strategy, market, broker, run_through="signal")
```

### 4.6 核心数据类型

```python
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

@dataclass(frozen=True)
class Position:                           # 单个持仓
    symbol: str
    shares: int
    avg_cost: Decimal                     # 加权平均成本

@dataclass
class Portfolio:                          # 组合状态（可变，逐 bar 更新）
    cash: Decimal
    positions: dict[str, Position]
    bar_prices: dict[str, Decimal]
    # total_value(prices) → Decimal

@dataclass
class Constraint:                         # 单标的交易约束
    max_shares: int | None = None
    max_value: float | None = None

@dataclass
class RuleResult:                         # Rule 的统一返回类型
    weights: dict[str, float] | None = None
    constraints: dict[str, Constraint] | None = None
    target_positions: dict[str, float] | None = None
    hold: bool = False                    # True 时冻结后续交易
    reason: str = ""                      # 审计用

@dataclass(frozen=True)
class UniverseSnapshot:                   # 某时间截面的标的池快照
    as_of_date: str
    symbols: tuple[str, ...]
    source: str
    metadata: dict[str, Any]
```

Order 是冻结的值对象，Rule 只负责生成意图（RuleResult），不直接产生 Order。OrderBook 中的 ManagedOrder 跟踪订单的生命周期（open → filled/canceled/expired），与 Order 本身分离。

### 4.7 Broker Protocol：策略与执行分离

策略层（Universe + Signal + Portfolio）只负责"做什么"的声明，通过两个可替换 Protocol 与执行环境解耦：

```python
@runtime_checkable
class MarketDataProvider(Protocol):
    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame: ...
    def get_latest(self, symbol: str) -> pd.Series: ...

@runtime_checkable
class Broker(OrderRouter, FillReceiver, Protocol):
    def submit_order(self, order: Order) -> str: ...
    def get_fills(self) -> list[Fill]: ...
    def on_bar_open(self, mktdata, date) -> None: ...
    def on_bar_close(self, mktdata, date) -> None: ...
    def get_open_orders(self, symbol=None) -> list[ManagedOrder]: ...
    def cancel_orders(self, symbol, side=None) -> list[ManagedOrder]: ...
    def cap_pending_sells(self, symbol, max_shares) -> None: ...
```

三种运行模式通过注入不同实现切换，策略代码零修改：

| 模式 | MarketDataProvider | Broker |
|---|---|---|
| 回测 | `LocalMarketDataProvider` | `SimBroker` |
| Paper Trade | `AlpacaMarketDataProvider` | `SimBroker` |
| 实盘 | `AlpacaMarketDataProvider` | `LiveBroker` |

```python
# 回测
engine.run(strategy, market=LocalMarketDataProvider(),
           broker=SimBroker(), rules=[...],
           start="2023-01-01", end="2024-12-31")

# 实盘（未来）：仅替换 Provider
engine.run(strategy, market=RealtimeDataProvider(),
           broker=LiveBroker(), rules=[...])
```

---

## 5. 功能模块

### 5.1 数据层 (oxq.data)

框架将所有外部数据统一视为 indicator——无论其原始来源是行情、财务报表、宏观经济指标还是新闻舆情。从策略视角，PE ratio、GDP 增速、舆情分数与 RSI 本质相同——都是"某个时间点上的一个数值"。所有数据最终以列的形式汇入宽表 `mktdata`，参与 Indicator → Signal → Portfolio 管道。

核心原则：

1. **一切皆 indicator**：不区分 indicator 和 factor，统一称为 indicator。它可能来自数学公式对量价数据的计算，可能由现有 indicator 衍生，也可能由外部系统（如机器学习模型）计算后注入
2. **Point-in-Time 对齐**：数据按实际可用日（announce_date）而非报告期（period_date）对齐进宽表，防止前视偏差
3. **频率打平**：低频数据（季度财报、月度宏观）通过 forward-fill 对齐到日频宽表
4. **全局数据广播**：无 symbol 维度的数据（宏观指标）广播到全 universe 的每个 symbol

### 5.2 Universe 构建 (oxq.universe)

Universe 决定"每个时间截面上，哪些 symbol 参与计算"。缺少显式 Universe 管理会导致 survivorship bias。

```python
class UniverseProvider(Protocol):
    def get_universe(self, as_of_date: str) -> UniverseSnapshot: ...
    def get_history(self, start: str, end: str) -> list[UniverseSnapshot]: ...
```

三种内置实现：

| 实现 | 说明 | 适用场景 |
|------|------|----------|
| `StaticUniverse` | 固定 symbol 列表 | 单标的策略、手动指定标的池 |
| `IndexUniverse` | 指数成分股，支持 Point-in-Time | 沪深300轮动、行业指数策略 |
| `FilterUniverse` | 基于 indicator 条件动态过滤 | 全市场因子策略 |

### 5.3 指标库 (oxq.indicators)

Indicator 是路径无关的纯函数：输入单个 symbol 的 DataFrame，输出等长 Series，引擎追加为宽表新列。

```python
class Indicator(Protocol):
    name: str
    def compute(self, mktdata: pd.DataFrame, **params) -> pd.Series: ...
```

内置指标：

| 类别 | 指标 | 说明 |
|------|------|------|
| 趋势 | SMA, EMA, WMA, DEMA, TEMA | 移动平均系列 |
| 动量 | RSI, ROC, PPO, Momentum, NdayReturn, LogReturn | 动量与收益指标 |
| MACD | MACDLine, MACDSignal, MACDHistogram | MACD 拆分为三个独立指标 |
| 波动 | BollingerUpper, BollingerLower, ATR, RollingVolatility, RollingMDD | 波动率指标 |
| 成交量 | OBV, VWAP, MFI | 量价指标 |
| 方向 | ADX, AROON, StochK, CCI | 趋势强度指标 |
| 因子 | Ratio | 比率因子 |

指标拆分原则：`compute()` 返回单个 Series（一列）。多输出指标拆分为独立类：BBands 拆为 `BollingerUpper` + `BollingerLower`，MACD 拆为三个。

### 5.4 信号生成器 (oxq.signals)

Signal 和 Indicator 签名相同——都是逐 symbol 计算的纯函数。区别在于语义：Indicator 输出连续数值，Signal 输出离散的交易意图。Signal 描述"交易的欲望"，策略可能因风控规则或约束条件而不执行信号。

```python
class Signal(Protocol):
    name: str
    def compute(self, mktdata: pd.DataFrame, **params) -> pd.Series: ...
```

7 种内置信号：

| 信号类型 | 说明 |
|----------|------|
| `Crossover` | 两条线交叉（上穿检测） |
| `Threshold` | 超过/低于阈值 |
| `Comparison` | 两个值比较 |
| `Formula` | 自定义布尔公式 |
| `Peak` | 峰值/谷值检测 |
| `Timestamp` | 时间条件触发 |
| `Composite` | 多信号 AND/OR 组合 |

跨 symbol 的操作（排名、权重归一化）由 PortfolioOptimizer 负责，不属于 Signal 层。

### 5.5 组合优化器 (oxq.portfolio.optimizers)

PortfolioOptimizer 负责将 Signal 输出转化为目标组合权重。所有权重之和为 1.0（含 CASH）。

```python
class PortfolioOptimizer(Protocol):
    name: str
    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]: ...
```

4 种内置优化器：

| 优化器 | 逻辑 | 适用场景 |
|--------|------|----------|
| `EqualWeightOptimizer` | 等权分配 | 分散投资、快速验证 |
| `RiskParityOptimizer` | 按波动率倒数加权 | 风险均衡配置 |
| `KellyOptimizer` | Kelly 公式计算最优仓位 | 有历史胜率数据时 |
| `TopNRankingOptimizer` | 按评分排名取 Top N 归一化 | 截面动量/因子策略 |

当所需的 indicator 列缺失时，优化器跳过对应标的物并将权重分配给 CASH，而非抛出错误。

### 5.6 交易规则 (oxq.rules)

Rule 返回 RuleResult，而非 Order。RuleResult 携带 weights（权重覆盖）、constraints（约束条件）、target_positions（目标仓位）或 hold（冻结后续规则）。Rules 不属于 Strategy，通过 `Engine.run(rules=[...])` 传入。

在执行管道中 Rule 分为两个时机：

- **Pre-trade Rule** — 在 Portfolio 产出目标组合后、交易执行前检查。可调整权重、附加约束或冻结交易
- **Post-trade Rule** — 在交易执行后监控持仓。触发止损、止盈等减仓操作

已实现的 Rule：

| 规则 | 时机 | 返回 |
|------|------|------|
| `MaxDrawdownRisk` | Pre-trade | `RuleResult(target_positions={sym: 0.0}, hold=True)` |
| `DailyLossLimitRisk` | Pre-trade | `RuleResult(hold=True)` |
| `StopLossRule` | Post-trade | `RuleResult(target_positions={sym: 0.0})` |
| `TakeProfitRule` | Post-trade | `RuleResult(target_positions={sym: 0.0})` |
| `TrailingStopRule` | Post-trade | `RuleResult(target_positions={sym: 0.0})` |
| `ExitRule` | Post-trade | `RuleResult(target_positions={sym: 0.0})` |

Post-trade Rule 只产出减仓意图（target_positions），不产出加仓意图。加仓完全由 Signal + Portfolio 驱动。Engine 根据 target_positions 计算卖出股数并统一通过交易算法提交。

### 5.7 交易执行 (oxq.trade)

- **OrderGenerator**：将目标权重 + 当前持仓 → 订单列表（PlannedOrder），支持 lot_size 手数约束（美股默认 1，A 股 100）
- **SimBroker**：模拟撮合，实现 Broker Protocol，支持 market/limit/stop/trailing_stop 订单类型
- **LiveBroker**：实盘交易（Alpaca API）
- **FeeModel / SlippageModel**：PercentageFee（比例手续费 + 最低收费）、PercentageSlippage（百分比滑点）
- **多交易所**：SSE, SZSE, NYSE, NASDAQ, HKEX

### 5.8 组合管理 (oxq.portfolio)

- **OrderBook**：管理 ManagedOrder 集合，跟踪订单生命周期
- **Portfolio**：管理持仓状态（positions）、现金余额（cash）、当前价格快照（bar_prices）
- **RunResult**：回测结果，包含 portfolio、trades、equity_curve、mktdata、benchmark_prices，提供 total_return()、sharpe_ratio()、max_drawdown()、annualized_return()、annualized_volatility()、calmar_ratio()、sortino_ratio() 等方法
- **ExecutionReport**：对比模拟成交与实盘成交

### 5.9 执行引擎 (oxq.core.engine)

Engine 是通用策略执行引擎，provider-agnostic——不知道自己在运行回测还是实盘。

```python
def run(self, strategy, market, broker,
        start="", end="", initial_cash=100_000.0,
        rules=None, run_through=None, tracer=None) -> RunResult
```

`setup()` + `step()` 分离支持两种执行模式：

```python
# 批量执行（回测）
result = engine.run(strategy, market, broker, rules=[...], ...)

# 逐步执行（实盘/调试）
engine.setup(strategy=strategy, market=market, broker=broker, rules=[...], ...)
for date in engine.dates:
    engine.step(date)
result = engine.result
```

### 5.10 参数优化 (oxq.optimize)

```python
paramset = ParamSet("sma_tuning")
paramset.add_distribution("sma_fast", "period", values=range(5, 30, 5))
paramset.add_distribution("sma_slow", "period", values=range(20, 100, 10))
paramset.add_constraint("sma_fast.period < sma_slow.period")

results = grid_search(strategy, paramset, data, metric="sharpe_ratio")

wfa_results = walk_forward(strategy, paramset, data,
    train_period="2Y", test_period="6M", step="3M",
    optimize_metric="sharpe_ratio", anchored=False)
```

### 5.11 统计检验 (oxq.optimize.validation)

| 方法 | 用途 |
|------|------|
| `deflated_sharpe()` | 校正多重比较后的 Sharpe Ratio |
| `haircut_sharpe()` | 对 Sharpe Ratio 施加 haircut |
| `profit_hurdle()` | 最低利润门槛检验 |
| `white_reality_check()` | Bootstrap 检验策略收益是否显著 |
| `k_fold_cv()` | 时间序列 k 折交叉验证 |
| `cscv()` | 组合对称交叉验证 |
| `oos_deterioration()` | 样本外退化度量 |

### 5.12 可观测性 (oxq.observe)

- **DefaultTracer**：执行追踪，生命周期钩子（on_run_start、on_indicator、on_signal、on_rule、on_run_end），生成 TraceSpan
- **AuditRecord**：审计日志，包含策略配置快照 + 四维哈希（mktdata_hash、trades_hash、equity_hash、result_hash）用于确定性验证
- **StrategyMonitor**：监控策略运行状态，检测绩效偏离和异常期
- **MarketStateDetector**：基于波动率检测市场状态（高波/低波/正常）
- **ExperimentLog**：结构化实验日志，记录假设、观察、结论

---

## 6. Tool 定义与分发

### 6.1 Tool 定义（oxq.tools）

Tool 定义是框架的核心资产之一，与传输协议无关。每个 Tool 是 SDK 的薄封装：参数解析 → 调用 `oxq` SDK → 格式化返回。所有计算和逻辑在 SDK 中实现。

| 工具组 | 工具名 | 说明 |
|--------|--------|------|
| **strategy** | `strategy_create` | 创建策略 |
| | `strategy_add_indicator` | 添加指标 |
| | `strategy_add_signal` | 添加信号 |
| | `strategy_inspect` | 查看策略详情 |
| | `strategy_list` | 列出所有策略 |
| | `indicator_list` | 列出可用指标类型 |
| | `indicator_describe` | 查看指标参数说明 |
| | `signal_list` | 列出可用信号类型 |
| | `signal_describe` | 查看信号参数说明 |
| | `rule_list` | 列出可用规则类型 |
| | `rule_describe` | 查看规则参数说明 |
| **data** | `data_load_symbols` | 加载标的行情数据 |
| | `data_list_symbols` | 列出已有标的 |
| | `data_inspect` | 查看数据摘要 |
| | `factor_download` | 下载宏观因子 |
| | `factor_list` / `factor_inspect` | 因子查询 |
| **universe** | `universe_set` | 设置 Universe |
| | `universe_inspect` | 查看成分快照 |
| | `universe_history` | 查看成分变动 |
| **engine** | `engine_run` | 运行策略 |
| | `engine_results` | 获取绩效指标 |
| | `engine_trade_list` | 查看交易记录 |
| | `run_list` | 列出运行历史 |
| **optimize** | `paramset_create` / `paramset_list` | 参数空间管理 |
| | `grid_search` / `walk_forward` | 搜索方法 |
| | `cross_validate` / `overfit_analysis` | 统计检验 |
| **observe** | `observe_trace` / `observe_audit_*` | 追踪与审计 |
| | `observe_monitor_*` | 策略监控 |
| | `observe_experiment_*` | 实验日志 |
| **live** | `live_connect` / `live_account` | 券商连接 |
| | `live_positions` / `live_bars` | 实盘数据 |
| | `live_generate_orders` / `live_submit_order` | 订单管理 |

### 6.2 MCP Server（可选分发层）

MCP Server 是 `oxq.tools` 的 MCP 协议适配，用于支持不能执行代码的 AI 客户端。MCP Server 不包含业务逻辑，只做协议适配和自动注册。Coding Agent 直接 `import oxq` 即可，不需要 MCP Server。

---

## 7. Agent Skills

每个 skill.md 描述一个完整的 Agent 工作流，指导 AI Agent 如何组合 tools 完成任务。

| Skill | 状态 | 核心工作流 |
|-------|------|-----------|
| `strategy-builder.md` | 已实现 | 约束 → 目标 → 假设 → 数据 → 逐层构建 → 回测 |
| `data-explorer.md` | 已实现 | 检查数据 → 下载行情/因子 → 质量检查 |
| `backtest-runner.md` | 已重定向 | → strategy-builder |
| `parameter-tuner.md` | 模板 | 参数优化 + 统计检验 |
| `performance-reviewer.md` | 模板 | 绩效分析 + 归因 |
| `risk-analyzer.md` | 模板 | 回撤分析 + 压力测试 |
| `trade-executor.md` | 模板 | 订单生成 → 确认 → 监控 |
| `strategy-monitor.md` | 模板 | 实盘监控 + 偏离检测 |
| `universe-builder.md` | 模板 | Universe 构建 |
| `live-trader.md` | 模板 | 实盘交易 |

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
| 可选依赖 | mcp (Python) | MCP Server 分发时需要 |
| 构建工具 | uv | 现代 Python 项目管理 |
| 测试 | pytest | 标准选择 |

---

## 9. 实现路线

### Phase 1: 核心引擎 + SDK (MVP) ✅ 已完成
- `oxq.core`: Strategy (Universe + Signal + Portfolio), Engine, types (Order, Fill, Portfolio, Position, RuleResult, 全部 Protocol)
- `oxq.universe`: UniverseProvider Protocol, StaticUniverse, FilterUniverse
- `oxq.indicators`: 27 个内置指标
- `oxq.signals`: 7 种 per-symbol 信号（Crossover, Threshold, Comparison, Formula, Composite, Peak, Timestamp）
- `oxq.rules`: ExitRule, StopLossRule, TakeProfitRule, TrailingStopRule, MaxDrawdownRisk, DailyLossLimitRisk
- `oxq.portfolio`: Portfolio, Position, RunResult + analytics, PortfolioOptimizer（EqualWeight, RiskParity, Kelly, TopNRanking）
- `oxq.trade`: SimBroker, FeeModel, SlippageModel, OrderGenerator
- `oxq.data`: LocalMarketDataProvider, YFinanceDownloader, AkShareDownloader, WorldBank 因子
- `oxq.tools`: 40+ 个协议无关 Tool
- `mcp_server`: FastMCP 适配层
- `skills/`: strategy-builder, data-explorer 等 10 个 skill

### Phase 2: 参数优化 + 统计检验 ✅ 已完成
- `oxq.optimize`: ParameterSet, GridSearch, WalkForward, TimeSeriesCV, 过拟合分析
- 未完成: `IndexUniverse`（Point-in-Time）

### Phase 3: 交易执行 + 可观测性 🔄 部分完成
- `oxq.observe`: DefaultTracer, AuditRecord, StrategyMonitor, MarketStateDetector, ExperimentLog
- `oxq.trade`: LiveBroker (Alpaca), FillPriceMode
- `oxq.contrib.alpaca`: AlpacaClient, AlpacaMarketDataProvider
- `oxq.portfolio`: ExecutionReport, OrderBook
- 未完成: observe_replay, EventBus

### Phase 4: 多策略 + 高级特性
- `oxq.orchestrator`: 多策略编排 + 资金分配
- 更多指标/信号
- 机构级多策略管理能力

---

## 参考

- **quantstrat (R)**: indicator → signal → rule 分层模型、paramset 参数优化、walk-forward analysis、Deflated Sharpe Ratio 统计检验、order book 管理
- **xquant.shop**: agent pipeline 架构、immutable specs、provider injection
- **Peterson, Brian G. (2017)**: *"Developing & Backtesting Systematic Trading Strategies"* — 假设驱动开发、逐组件评估、信号预测力评估、MAE/MFE 分析、Rule Burden、Walk Forward Analysis、统计检验方法论
