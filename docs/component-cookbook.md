# 组件手册：48 指标 / 8 信号 / 10 规则 / 6 优化器

> 配套脚本:`examples/modules/12_component_gallery.py` —— 一次运行,72 个内置
> 组件全部产出真实输出。这份文档是它的讲解版:每条都标注是否实测,
> 参数契约以脚本里跑通的为准,不是凭记忆写的。
>
> 想跑通的策略模板见 `examples/modules/13_strategy_combos.py`(把这些组件
> 组合成 5 种完整可用的 `StrategySpec`)。

---

## 先读这个:四类组件的调用方式完全不同

这是本篇最容易踩坑的地方——四类组件**不是同一套调用约定**:

| 组件类型 | 调用方式 | 输入 | 输出 |
|---|---|---|---|
| Indicator | `cls().compute(bars, **params)` | 一只标的的 OHLCV DataFrame | `pd.Series` |
| Signal | `cls().compute(df_with_indicators, **params)` | 含指标列的 DataFrame | `pd.Series`(bool 或分类值) |
| Rule | `cls(**init_params).evaluate(symbol, row, portfolio, prices)` | 单行 `row` + `Portfolio` 对象 | `RuleResult` |
| PortfolioOptimizer | `cls(**init_params).optimize(signals, indicators)` | `dict[symbol, DataFrame]` × 2 | `dict[symbol, weight]` |

在 `strategy_spec.yaml` / `StrategySpec` 里,这些差异被 `IndicatorDef` /
`SignalRuleDef` / `PortfolioRuleDef` / `portfolio.type+params` 统一包装掉了——
但如果你想在 spec 之外直接调用某个组件做单元测试,就必须知道上面这张表。

---

## 1. 指标(Indicator)—— 48 个

按注册表的 `category` 元数据分组(用 `get_indicator_metadata(name)["category"]`
取得,不是凭印象归类的):

| 分类 | 数量 | 组件 |
|---|---|---|
| trend(趋势) | 10 | SMA, EMA, DEMA, TEMA, WMA, IchimokuTenkan/Kijun/SenkouA/SenkouB/Chikou |
| momentum(动量) | 9 | ROC, RSI, Momentum, SimpleMomentum, NdayReturn, LogReturn, PPO, StochK, **RPS** |
| volatility(波动) | 8 | ATR, BollingerUpper/Lower, GarchVolatility, HurstExponent, RollingVolatility, RollingMDD, AnnualizedVolatility |
| valuation(估值) | 5 | PE, PB, EP, BP, MarketCap |
| quality(质量) | 4 | AccrualRatio, CashFlowRatio, NetProfitMargin, ROEChange |
| volume(成交量) | 4 | OBV, MFI, VWAP, TurnoverRate |
| direction(方向) | 3 | ADX, AROON, CCI |
| macd | 3 | MACDLine, MACDSignal, MACDHistogram |
| factor(合成) | 2 | Ratio, PowerRatio |

### 通用调用(45/48 走这条路)

```python
from oxq.data.market import LocalMarketDataProvider
from oxq.indicators import SMA, RSI

market = LocalMarketDataProvider(data_dir="~/.oxq/data/market")
bars = market.get_bars("510300", "2023-01-01", "2024-12-31")

sma = SMA().compute(bars, column="close", period=20)
rsi = RSI().compute(bars, column="close", period=14)
```

**参数因指标而异**,不能假设都是 `(column, period)`:

```python
ATR().compute(bars, period=14)                              # 没有 column 参数
MACDLine().compute(bars, column="close", fast_period=12, slow_period=26)
Ratio().compute(bars, col_a="mom_20", col_b="vol_20")        # 完全不同的参数名
```

调用前用 `inspect.signature(cls.compute)` 核实,别猜。

### 3 个特殊情况(实测过,和"通用调用"不一样)

**① `RPS` 是唯一横截面指标**,吃的是全市场的 dict,不是单只标的的 DataFrame:

```python
from oxq.indicators import RPS

bars_by_symbol = {"510300": bars_1, "510050": bars_2, "510500": bars_3}
result = RPS().compute_cross_section(bars_by_symbol, column="close", period=60)
# result: dict[str, pd.Series],每只标的一个百分位排名序列
```

**② `MACDSignal` / `MACDHistogram` 读的是别的指标算出来的列**,不会自己重算
MACD 线:

```python
bars["macd"] = MACDLine().compute(bars, column="close")
bars["macd_signal"] = MACDSignal().compute(bars, macd_col="macd")          # 读 "macd" 列
histogram = MACDHistogram().compute(bars, macd_col="macd", signal_col="macd_signal")
```

**③ 估值/质量/换手率类(10 个)读的是基本面列,不是行情列**——本地 parquet
的 OHLCV 数据没有这些列,真实使用需要能提供基本面数据的 provider:

| 指标 | 默认读取列 |
|---|---|
| PE | `eps`, `close` |
| PB | `book_value_per_share`, `close` |
| EP | `eps`, `close` |
| BP | `book_value_per_share`, `close` |
| MarketCap | `total_shares`, `close` |
| AccrualRatio | `net_income`, `operating_cash_flow`, `total_assets` |
| CashFlowRatio | `operating_cash_flow`, `total_assets` |
| NetProfitMargin | `net_income`, `revenue` |
| ROEChange | `roe` |
| TurnoverRate | `volume`, `total_shares` |

`12_component_gallery.py` 用构造出的常数列(`eps=0.35` 等)跑通这些指标的
公式,**不代表这是真实财务数据**,只是证明调用契约正确。

---

## 2. 信号(Signal)—— 8 个

统一签名:`compute(mktdata: pd.DataFrame, **params) -> pd.Series`。全部实测:

| 组件 | 参数 | 输出域 | 实测(510300, 2023-2024, 484 交易日) |
|---|---|---|---|
| Comparison | `left, right, relationship` | bool | `close > sma_slow`: 173 True / 311 False |
| Crossover | `fast, slow` | bool(仅穿越当天为 True) | 6 True / 478 False |
| Threshold | `column, threshold, relationship` | bool | `rsi_14>70`: 10 True / 474 False |
| Composite | `signals: list[str], logic` | bool | AND 组合:152 True / 332 False |
| Formula | `expr`(pandas `.eval()` 表达式) | bool | 与等价 Composite 结果完全一致 |
| Peak | `column, kind, order` | bool | 局部极值点:46 True / 438 False |
| ROCTiming | `column, mode, bottom, top`(或 `mode="rolling_quantile"` + `q_window/q_bottom/q_top`) | **三态字符串** `{"BUY","SELL","HOLD"}` | 441 HOLD / 27 SELL / 16 BUY |
| Timestamp | `rule`(`month_start`/`month_end`/`quarter_start`/`quarter_end`/`weekday:N`) | bool | 月初标记:24 True / 460 False |

只有 `ROCTiming` 输出的不是布尔值,而是 `"BUY"/"SELL"/"HOLD"` 三态字符串——
这一点很重要,因为**只有它能直接喂给 `SignalToPosition` 优化器**(见下)。

---

## 3. 规则(Rule)—— 10 个

统一签名:`evaluate(symbol: str, row: pd.Series, portfolio: Portfolio, prices: dict|None) -> RuleResult`。

**关键认知(本轮实测才搞清楚,容易想当然搞错)**:不同规则读取"当前价格"的
来源不一样,不是都从 `prices` 参数或 `portfolio.bar_prices` 读:

| 规则 | 价格/日期来源 | 是否跨调用有状态 |
|---|---|---|
| StopLossRule | `row["close"]` vs `portfolio.positions[symbol].avg_cost` | 无状态 |
| TakeProfitRule | `row["close"]` vs `avg_cost` | 无状态 |
| TrailingStopRule | `row["close"]`,内部记录 high-water mark | **有状态**(第二次调用起才可能触发) |
| ExitRule | `row[fast]` / `row[slow]`(任意列名,不一定叫 close) | 无状态 |
| MaxDrawdownRisk | `portfolio.total_value(prices)`,内部记录峰值 | **有状态** |
| DailyLossLimitRisk | `portfolio.total_value(prices)` + `row.name`(日期),内部记录当日起始值 | **有状态**(按日期换新起点) |
| MaxHoldingsRule | 不看价格,只看 `len(portfolio.positions)` | 无状态(但不阻止已持仓标的) |
| BlacklistRule | 不看价格 | 无状态 |
| RebalanceFrequencyRule | `row.name`(bar 的时间戳),内部计数已过多少个 bar | **有状态** |
| CalendarRebalanceRule | `row.name`,内部记录当前周期是否已放行过 | **有状态** |

**"有状态"意味着单次调用往往看不出效果**——`MaxDrawdownRisk` 第一次调用只是
记录峰值,不可能同时触发;必须先用一次高价格调用"建立峰值",再用低价格
调用才会触发。`12_component_gallery.py` 里每个有状态规则都按"先建立状态、
再触发"的两步调用顺序写的,直接照抄那个模式。

### 实测示例(节选自 gallery 脚本)

```python
from decimal import Decimal
import pandas as pd
from oxq.core.types import Portfolio, Position
from oxq.rules import StopLossRule, MaxDrawdownRisk

def held(entry_price, cash="10000"):
    return Portfolio(cash=Decimal(cash),
                      positions={"510300": Position("510300", shares=100, avg_cost=Decimal(str(entry_price)))})

def row(close, date="2024-01-15"):
    return pd.Series({"close": close}, name=pd.Timestamp(date))

# StopLoss:只看 row["close"] vs avg_cost,跟 prices 参数无关
stop = StopLossRule(threshold=0.05)
pf = held(4.0)
stop.evaluate("510300", row(3.90), pf)   # holds:跌 2.5%,阈值 5%
stop.evaluate("510300", row(3.50), pf)   # fires:跌 12.5%,触发止损

# MaxDrawdownRisk:cash 要设成 0,否则现金会稀释掉价格波动的占比
mdd = MaxDrawdownRisk(max_drawdown=0.15)
pf = held(4.0, cash="0")
mdd.evaluate("510300", row(4.5), pf, prices={"510300": Decimal("4.5")})  # 建立峰值
mdd.evaluate("510300", row(3.7), pf, prices={"510300": Decimal("3.7")})  # fires:17.8% 回撤
```

`MaxHoldingsRule` 还有第二个方法 `evaluate_batch(target_weights, portfolio,
pending_orders)`,用于对一整批目标权重做统一限仓(而不是逐标的判断),
实际回测引擎内部用的是这个批量版本。

---

## 4. 组合优化器(PortfolioOptimizer)—— 6 个

统一签名:`optimize(signals: dict[str, DataFrame], indicators: dict[str, DataFrame]) -> dict[str, float]`。

**⚠️ 已发现的目录纠错**:早前一份手工整理的组件目录文档把这类组件写作
`Kelly(win_rate: float, win_loss_ratio: float)`,**是错的**。本轮实测确认:

- 类名带 `Optimizer` 后缀:`oxq.portfolio.optimizers.KellyOptimizer`,不是 `Kelly`
  (其余 5 个同理:`EqualWeightOptimizer`、`TopNRankingOptimizer` 等;注册表
  名字 `"Kelly"`/`"EqualWeight"` 是给 `spec.portfolio.type` 用的,不是类名)
- **真实参数是列名,不是数值**:
  `KellyOptimizer(win_rate_col: str, avg_win_col: str, avg_loss_col: str, fraction: float=1.0)`——
  它从 `indicators` 里对应的列读数值,不是让你直接传胜率数字

| 组件 | 类名 | 构造参数 | 实测输出(3 只标的) |
|---|---|---|---|
| EqualWeight | `EqualWeightOptimizer` | 无 | 等权:各 0.333 |
| TopNRanking | `TopNRankingOptimizer` | `score_col, n, filter_negative, max_weight, pre_filter_signal, weighting, ascending` | 按分数加权选 2 只:0.53/0.47 |
| PctEquity | `PctEquityOptimizer` | `pct` | 每只固定 20%,剩余归 CASH |
| RiskParity | `RiskParityOptimizer` | `volatility_col` | 按波动率倒数加权 |
| **Kelly** | `KellyOptimizer` | `win_rate_col, avg_win_col, avg_loss_col, fraction=1.0` | 按 Kelly 公式,剩余归 CASH |
| SignalToPosition | `SignalToPositionOptimizer` | `signal, buy_weight=1.0, sell_weight=0.0, hold_behavior="maintain"` | 单标的择时:BUY→1.0 |

```python
from oxq.portfolio.optimizers import TopNRankingOptimizer, KellyOptimizer

signals = {"510300": pd.DataFrame({"entry_gate": [True]}), ...}
indicators = {
    "510300": pd.DataFrame({"rps_60": [80.0], "win_rate": [0.55], "avg_win": [0.02], "avg_loss": [0.01]}),
    ...
}

TopNRankingOptimizer(score_col="rps_60", n=2, pre_filter_signal="entry_gate").optimize(signals, indicators)
KellyOptimizer(win_rate_col="win_rate", avg_win_col="avg_win", avg_loss_col="avg_loss", fraction=0.5).optimize(signals, indicators)
```

`SignalToPosition` 的 `signal` 列必须是 `"BUY"/"SELL"/"HOLD"` 三态字符串——
正好是 `ROCTiming` 信号的输出格式,这不是巧合,两者就是配套设计的
(见 `examples/modules/13_strategy_combos.py` 组合 5)。

---

## 哪些组件需要额外前提,不能开箱即用

- **估值/质量类指标**(PE/PB/AccrualRatio 等 10 个):需要能提供基本面数据列
  的 provider,本地 OHLCV parquet 没有这些列
- **Rule 全部**:需要构造 `oxq.core.types.Portfolio`/`Position` 才能调用,
  不能直接喂 DataFrame
- **有状态 Rule**(6 个,见上表):单次调用看不出触发效果,必须按"建立状态→
  触发"两步调用
- **Optimizer 全部**:需要 `signals`/`indicators` 都是 `dict[symbol, DataFrame]`
  形状,不是拼好的宽表

在 `strategy_spec.yaml` / `StrategySpec` 里用这些组件时,以上细节全部由
编译器(`oxq/spec/compiler.py`)处理好了——只有想绕开 spec、直接调组件做
单元测试时才需要关心这些。
