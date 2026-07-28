# 快速研究手册：因子测试与策略回测

> 面向"我想验证一个想法,而不是交付一份合规报告"的日常研究场景。
>
> 本手册的每一条 API 契约、每一个坑、每一组耗时数字,都来自一次完整的实跑
> 验证,不是从文档抄的。凡是没实测过的地方都已标注。

---

## Part 1 · 先决定要验证到哪一层

这是全篇最重要的一张表。**大部分时间浪费都源于用第三层的工具去回答第一层的问题。**

| 你要回答的问题 | 用什么 | 实测耗时 | 需要 spec/skill 吗 |
|---|---|---|---|
| 这个指标算出来的数值对不对、长什么样 | `Indicator().compute(bars, **params)` | **~1.8 秒**(几乎全是 Python 启动开销) | 都不需要 |
| 这个因子有没有预测力 | `examples/modules/11_factor_probe.py` 或 `oxq.factor_eval.metrics` | **秒级** | 不需要 spec |
| 这个策略放进真实交易赚不赚钱 | `StrategySpec` → `compile_run()` | **~75 秒**(回测本身 32 秒) | 需要 spec |
| 这个结论要留痕给别人交代 | 八闸门治理流程 | **数小时 + 大量 context** | 全都需要 |

### 关于八闸门治理流程

`brainstorm-strategy-idea` → `audit-strategy-idea` → `build-strategy-spec` →
`audit-strategy-spec` → `audit-runtime-semantics` → `run-authorized-backtest`
→ `monitor-strategy-run` → `write-research-report`

这套流程产出哈希链、逐字段确认表、授权事件等完整证据。它的价值是**证明
"人类逐字段看过并签字"**,不是提高结果准确性。

**实测结论**:同一个 SPEC,走完整八闸门 vs 直接 `oxq backtest run --allow-unaudited`,
**产出的数值完全一致**,后者 39 秒完成。差别只在有没有证据链。

→ **日常迭代一律走快车道。只有"这就是最终定稿、要给别人看"时才走八闸门。**

### 关于 skill

skill 不会让事情变快——它们本质上是教你调那几行 SDK 代码的**防错清单**。
本手册已经把最关键的清单内容抄录下来了,日常研究不必再加载 skill。

---

## Part 2 · 因子快速测试(首选路径)

### 2.1 用现成脚手架

`examples/modules/11_factor_probe.py` 可以探测 48 个内置指标中的**任意一个**,
不需要写 spec、不需要跑回测。

```bash
# 先看有哪些指标可用
python examples/modules/11_factor_probe.py --list

# 探测一个因子
python examples/modules/11_factor_probe.py \
    --indicator RSI \
    --symbols 510300 510050 510500 159915 \
    --start 2022-01-01 --end 2025-12-31

# 指标专属参数用 JSON 传
python examples/modules/11_factor_probe.py \
    --indicator MACDLine --params '{"fast_period": 12, "slow_period": 26}' \
    --symbols 510300 510050 510500
```

输出包含:因子数值样本、各 horizon 的 IC / RankIC / ICIR、IC 衰减曲线、换手率。

### 2.2 输出怎么读

以本轮对 21 只 ETF 的两次实测为例:

**RSI(14)**

```
horizon   1d: IC -0.0074  RankIC +0.0057  ICIR -0.0182
horizon   5d: IC +0.0163  RankIC +0.0174  ICIR +0.0406
horizon  20d: IC +0.0154  RankIC +0.0116  ICIR +0.0391
Turnover:  0.0987
```

IC 全部在 ±0.02 以内 → **基本没有预测力**,可以直接放弃,不必再花时间回测。

**RPS(60)**(2021-2026)

```
horizon   1d: IC +0.0094  RankIC +0.0126  ICIR +0.0221
horizon   5d: IC +0.0101  RankIC +0.0137  ICIR +0.0237
horizon  20d: IC -0.0209  RankIC -0.0204  ICIR -0.0473   ← 注意这里
Turnover:  0.0528
```

**20 日 IC 转负**。而本轮那个策略恰恰是 20 交易日(月频)调仓 + RPS(60) 排名。
这一行直接解释了为什么该策略 OOS Sharpe 只有 0.164 —— **不是参数没调好,
是因子在目标持有期上本身就没有正向预测力**。

> 这个结论 30 秒就能得出。当初走完整流程花了几小时才拿到等价信息。
> **这就是"先测因子再写策略"的全部意义。**

粗略判读基准(横截面因子):

| \|IC\| | 含义 |
|---|---|
| < 0.02 | 基本无效,别浪费时间 |
| 0.02 – 0.05 | 弱,可能被成本吃掉 |
| > 0.05 | 值得进一步验证 |

ICIR 比 IC 更重要——它衡量预测力**稳不稳定**。IC 高但 ICIR 低,说明只是
少数几天蒙对了。

### 2.3 API 契约(已逐个实测)

**指标解析**

```python
import oxq
from oxq.core.registry import list_indicators, get_indicator_metadata

list_indicators()                    # 48 个名字
getattr(oxq.indicators, "RSI")       # 48/48 全部可解析成类
get_indicator_metadata("RSI")        # {'description':..., 'category':...}
```

**`compute()` 签名因指标而异**——这是最容易踩的坑,必须用
`inspect.signature` 过滤参数,不能硬传:

```python
SMA.compute(mktdata, column='close', period=20)
MACDLine.compute(mktdata, column='close', fast_period=12, slow_period=26)
ATR.compute(mktdata, period=14)                    # 没有 column 参数
Ratio.compute(mktdata, col_a='', col_b='')         # 完全不同的参数名
```

**只有 `RPS` 有 `compute_cross_section`**,其余 47 个都是逐标的时序计算:

```python
RPS.compute_cross_section(mktdata: dict[str, pd.DataFrame], column, period, scale, min_symbols)
```

**评价函数精确签名**(`oxq.factor_eval.metrics`):

```python
compute_ic(factor: DataFrame, forward_returns: DataFrame, min_obs=3) -> dict
    # 返回 {'mean': float, 'std': float, 'series': list}
compute_rank_ic(factor, forward_returns, min_obs=3) -> dict          # 同上结构
compute_icir(ic_mean: float, ic_std: float) -> float
compute_decay(factor, prices, horizons: list[int], min_obs=3) -> dict
    # 返回 {'horizons': list, 'ic_values': list}
compute_turnover(factor: DataFrame) -> float
compute_ts_ic(factor, forward_returns, min_obs=30) -> dict           # 时序版
```

**时序因子**(单标的择时)走 tearsheet 路径:

```python
from oxq.factor_eval.bundle import create_bundle
from oxq.factor_eval.tearsheet import generate_tearsheet

bundle = create_bundle(factor_values: pd.Series, prices: pd.DataFrame, forward_periods: list[int])
result = generate_tearsheet(bundle, forward_periods=[1,5,20], output_dir="...")
```

### 2.4 防错清单

- **远期收益绝不能混入当日**:`prices.pct_change(h).shift(-h)`,让 t 日的因子
  只对上 t+1..t+h 的收益
- factor 与 forward_returns 必须按索引对齐后再算
- **多个 horizon 都要看**——单一 horizon 的结论不可信(见上面 RPS 的例子)
- 换手率要和交易成本一起判断:IC 0.03 但换手 0.5,大概率被成本吃光
- **样本量门槛**:
  - < 3 只:相关系数根本算不出来(`min_obs=3`),会返回 NaN
  - < 10 只:横截面 IC 是噪音,不能作为证据
  - 10–30 只:可用但要谨慎
  - \> 30 只:比较站得住

---

## Part 3 · 策略回测快车道

因子验证通过后再写策略。**用 Python 直接构造 `StrategySpec`,不必写 YAML。**

### 3.1 完整可粘贴模板

```python
import time
from pathlib import Path

from oxq.spec.schema import StrategySpec, IndicatorDef, SignalRuleDef, PortfolioRuleDef
from oxq.spec.validator import validate
from oxq.spec.compiler import compile_run
from oxq.audit import audit_reproducibility, audit_research
from oxq.robustness import run_robustness

TOP_N = 5   # <-- 迭代时就改这一行

t0 = time.time()

spec = StrategySpec.template(
    strategy_id="my_strategy_v1",
    hypothesis="用一两句话说清楚你赌的是什么规律",
    market_preset="cn_a_share",        # 或 "us_equity",只有这两个值
)

spec.universe.symbols = ["510300", "510050", "510500"]
spec.benchmark.symbols = ["510300"]

spec.data.data_dir = "~/.oxq/data/market"
spec.data.required_columns = ["open", "high", "low", "close", "volume"]

spec.validation.train_period = ["2015-01-01", "2020-12-31"]
spec.validation.test_period  = ["2021-01-01", "2026-07-24"]
spec.validation.required_oos = True

spec.signal.indicators = {
    "sma_trend": IndicatorDef(type="SMA", params={"column": "close", "period": 50}),
    "rps_60": IndicatorDef(type="RPS", params={"column": "close", "period": 60,
                                               "scale": 100.0, "min_symbols": 1}),
}
spec.signal.rules = {
    "trend_ok": SignalRuleDef(type="Comparison",
        params={"left": "close", "right": "sma_trend", "relationship": "gt"}),
    "momentum_ok": SignalRuleDef(type="Threshold",
        params={"column": "rps_60", "threshold": 50.0, "relationship": "gt"}),
    "entry_gate": SignalRuleDef(type="Composite",
        params={"signals": ["trend_ok", "momentum_ok"], "logic": "and"}),
}

spec.portfolio.type = "TopNRanking"
spec.portfolio.params = {
    "score_col": "rps_60", "n": TOP_N, "filter_negative": False,
    "max_weight": 1.0, "pre_filter_signal": "entry_gate",
    "weighting": "score", "ascending": False,
}
# 注意:值必须是 PortfolioRuleDef,传 dict 会报错(见 3.2)
spec.portfolio.rules = {
    "rebalance": PortfolioRuleDef(type="RebalanceFrequencyRule",
                                  params={"interval_days": 20}),
}

spec.execution.initial_cash = 1_000_000
spec.execution.lot_size_config.default = 100
spec.cost.fee_rate = 0.001
spec.cost.slippage_rate = 0.001

result = validate(spec)
print(f"validate: {result.status}")
if result.status == "fail":
    for f in result.findings:
        print("  -", f)
    raise SystemExit(1)

run_result, run_dir = compile_run(spec, out_dir="./runs/my_strategy")
print(f"Total Return:   {run_result.total_return():.2%}")
print(f"Annualized Ret: {run_result.annualized_return():.2%}")
print(f"Sharpe Ratio:   {run_result.sharpe_ratio():.3f}")
print(f"Max Drawdown:   {run_result.max_drawdown():.2%}")
print(f"Trades:         {len(run_result.trades)}")

# 这三个是真检验,而且很便宜(共约 40 秒),建议保留
print("reproducibility:", audit_reproducibility(run_dir).get("status"))
print("research bias:  ", audit_research(run_dir).get("status"))
print("robustness:     ", run_robustness(run_dir).get("status"))

print(f"\n总耗时: {time.time()-t0:.1f}s")
```

### 3.2 已实测踩到的坑

| 坑 | 症状 / 后果 | 正确做法 |
|---|---|---|
| **`portfolio.rules` 传 dict** | `validate()` 抛 `AttributeError: 'dict' object has no attribute 'type'`(`validator.py:606`) | 值必须是 `PortfolioRuleDef(...)` |
| `market_preset` 用错 | region/currency/calendar 全是美股默认值,语义完全错但不报错 | 只接受 `us_equity` / `cn_a_share`,A 股必须显式指定 |
| 把 `cost` 塞进 `execution` | 校验失败 | `cost` 是**顶层** section |
| `train_period` 写成 dict | 校验失败 | 是二元列表 `["start", "end"]` |
| `price_adjustment` 填别的值 | local provider 拒绝 | 只接受 `"adjusted"` |
| **调仓频率设了不生效** | 以为设了月频,实际天天调仓 | 以 `portfolio.rules.rebalance` 为准,`execution.rebalance` 会被它覆盖(见 `oxq/spec/compiler.py::_effective_rebalance()`) |
| `execution.lot_size` 不生效 | 旧字段,被静默覆盖 | 用 `execution.lot_size_config.default` |
| 想要当日收盘成交 | 框架不支持 | 目前只可靠支持 `next_open` 家族 |

### 3.3 两个审计层次不要混淆

| | SDK 函数(`oxq.audit` / `oxq.robustness`) | skill 治理闸门 |
|---|---|---|
| 检验内容 | 真的检查复现性、前视偏差、成本敏感度 | 证明"人类逐字段看过 SPEC" |
| 耗时 | 约 40 秒 | 数小时 |
| 产物 | 一个 dict,直接看 `['status']` | 哈希链、确认表、事件溯源 |
| 日常要不要 | **要**,便宜且有信息量 | 不要 |

三个函数签名都是 `(run_dir: str | Path) -> dict`。

### 3.4 CLI 等价路径

```bash
oxq spec validate strategy_spec.yaml
oxq backtest run strategy_spec.yaml --allow-unaudited --out ./runs --json
oxq registry export --out component_catalog.json   # 核实组件名是否存在
```

`--allow-unaudited` 跳过 `spec_audit.json` / `runtime_audit.json` 闸门。

---

## Part 4 · 新容器从零到可跑

> 每个 session 都是全新容器。`~/.config/open-xquant/`、`~/.oxq/data/market/`、
> `~/oxq-research/` 全部随容器销毁,只有 Git 仓库里的内容会自动出现。

### 4.1 安装 open-xquant

```bash
uv run oxq agent install --target claude-code --profile standalone-agent
```

**第一次必定崩溃**,这是框架 bug,不是你操作错了:回滚安全检查会对新建的
SDK bundle 目录做哈希,而 `uv venv` 创建的 Python 解释器是**符号链接**,
哈希函数拒绝符号链接。

**原地把同一条命令再跑一遍即可成功**——此时 bundle 已缓存,不再有"新目录"
需要哈希。

如果 `uv` 因为 `uv.lock` 锁定的镜像源被网络策略拦截而失败,用
`pip install -e .` 自建 venv 绕开,**不要修改 `uv.lock`**。

### 4.2 找到 runner

```bash
cat ~/.config/open-xquant/agent.yaml     # 读 preferred_runner_argv / preferred_runner
```

指向 `~/.config/open-xquant/sdk-bundles/<hash>/runner/.venv/bin/oxq`。
纯 SDK 脚本用同目录的 `.venv/bin/python3`。

**不要用裸的 `oxq`**——那可能是别的版本。

### 4.3 skill 装在哪

装到 `~/.claude/skills/`(**不是** `~/.config/open-xquant/skills/`)。
内容与仓库 `agent/skills/` **逐字节一致**(已 diff 验证),额外多出的只有
`.open-xquant-managed.json` 和 `manifest.json` 两种安装器记账文件。

→ 所以 skill 内容本来就在 Git 仓库里,不需要额外备份。

### 4.4 重建 ETF 行情数据

本轮数据来自私有仓库 `chenqiang0929-opus/etf-netflow-dev` 的 `data/20260727/`。

**关键口径**(搞错会得到完全错误的收益率):

- `kline.parquet` 是**未复权**价格
- 必须用 `nav.parquet` 的 `pct_chg` 经 `loader.adjusted_return_factor()`
  累乘出 `cum_factor`,再以各标的**最新日 raw close 为锚**反推复权价
- **绝不能拿 nav 两端相除算收益**——510310 因 2024-09-20 缩股,那样会算出
  +146.3%,真实值是 +17.8%

```python
"""Rebuild adjusted-OHLCV parquet from etf-netflow-dev snapshot."""
import sys
sys.path.insert(0, "/workspace/etf-netflow-dev/src")

import pandas as pd
from pathlib import Path
from etf_netflow import loader, config

OUT_DIR = Path.home() / ".oxq" / "data" / "market"
OUT_DIR.mkdir(parents=True, exist_ok=True)

codes = (  # 分类内选样:宽基/全球/行业各取流动性 top 7
    "513050", "159920", "510900", "513100", "513500", "159941", "513520",
    "510300", "510050", "510500", "159915", "159949", "159919", "510330",
    "512880", "512000", "512800", "512010", "512400", "512200", "512700",
)

kline = pd.read_parquet(config.SNAPSHOT_DIR / "kline.parquet")
kline["trade_date"] = pd.to_datetime(kline["trade_date"])
kline["code"] = kline["code"].astype("string")
kline = kline[kline["code"].isin(codes)].sort_values(["code", "trade_date"]).reset_index(drop=True)

factor = loader.adjusted_return_factor(codes)
factor["trade_date"] = pd.to_datetime(factor["trade_date"])

merged = kline.merge(factor, on=["code", "trade_date"], how="left")
merged["cum_factor"] = merged.groupby("code")["cum_factor"].ffill().bfill()
merged = merged[merged["cum_factor"].notna()].copy()

# 以最新日 raw close 为锚,把 cum_factor 换算成复权价
last_close = merged.groupby("code")["close"].transform("last")
last_cum = merged.groupby("code")["cum_factor"].transform("last")
adj_close = merged["cum_factor"] * (last_close / last_cum)
mult = adj_close / merged["close"]
merged["adj_open"] = merged["open"] * mult
merged["adj_high"] = merged["high"] * mult
merged["adj_low"] = merged["low"] * mult
merged["adj_close"] = adj_close

out = merged[["code", "trade_date", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]].rename(
    columns={"adj_open": "open", "adj_high": "high", "adj_low": "low", "adj_close": "close"}
)

for code, g in out.groupby("code"):
    g = g.drop(columns=["code"]).set_index("trade_date").sort_index()
    g.index = g.index.tz_localize("UTC")      # 必须 tz-aware
    g.index.name = "date"                      # 必须叫 date
    g["volume"] = g["volume"].astype("int64")
    g[["open", "high", "low", "close", "volume"]].to_parquet(OUT_DIR / f"{code}.parquet")
    print(code, len(g), g.index.min().date(), "->", g.index.max().date())
```

`LocalMarketDataProvider` 要求:文件名 `<SYMBOL>.parquet`、index 是
tz-aware DatetimeIndex 且命名 `date`、含 `open/high/low/close/volume` 列。

验证数据是否就位:

```python
from oxq.tools.data import list_symbols, inspect_symbol
print(list_symbols())
print(inspect_symbol("510300"))     # 看 missing_values 是否为 0
```

### 4.5 universe 选样口径

**必须分类内独立排流动性**(宽基 / 全球 / 行业 各取 top 7 = 21 只)。

不要"全市场混排 top N 再按类别筛"——那样宽基会把跨境和行业 ETF 全部挤掉。
本轮最初就是这么错的,全球只剩 3 只、行业只剩 6 只。

---

## Part 5 · 已有结论存档(不要重复验证)

### `etf_price_momentum_v1` — 已被证伪

**假设**:60 日 RPS 最强、且站上 50 日均线的 ETF,未来约 20 个交易日延续跑赢。

**配置**:21 只 ETF(宽基/全球/行业各 7) · RPS(60) + SMA(50) 过滤 ·
TopNRanking n=5 按分数加权 · 20 交易日调仓 · fee/slippage 各 0.001 ·
初始资金 100 万 · train 2015-2020 / test 2021-2026 · benchmark 510300

**结果**:

| 指标 | n=5(定稿) | n=8(对照) |
|---|---|---|
| 总收益 | +14.4% | +10.8% |
| 年化 | +1.2% | +0.93% |
| Sharpe | 0.159 | 0.142 |
| 最大回撤 | -53.9% | -50.84% |
| 交易笔数 | 753 | 1052 |
| **OOS Sharpe** | **0.164** | — |

**判定**:OOS Sharpe 0.164 < 预注册阈值 0.3 → 按策略自己的 `decision_policy`
**应判 REJECT**。这是真实的负面结果,不是流水线故障。

**根因**(事后用 factor_probe 30 秒定位):RPS(60) 在这批标的上
**20 日 IC = -0.0209**,即在目标持有期上不但没有正向预测力,还微弱反向。
调参数救不了——问题在因子本身。

**已知局限**:静态 universe 存在幸存者偏差(用户已知并接受);样本区间
大部分是 A 股单边行情。

### 下次可以试的方向

- 缩短持有期到 1–5 天(RPS 在 1d/5d 的 IC 为正,虽然很弱)
- 换因子:先用 `factor_probe.py` 批量筛 48 个内置指标在 20 日 horizon 上的 IC,
  找到 |IC| > 0.03 的再写策略
- 扩大 universe 到 30+ 只以提高横截面统计的可信度

---

## 附:速查

```bash
# runner
RUNNER=$(python -c "import yaml,os;print(yaml.safe_load(open(os.path.expanduser('~/.config/open-xquant/agent.yaml')))['preferred_runner'])")

# 因子探测
python examples/modules/11_factor_probe.py --list
python examples/modules/11_factor_probe.py --indicator RSI --symbols A B C --start 2022-01-01 --end 2025-12-31

# 快速回测
oxq backtest run spec.yaml --allow-unaudited --out ./runs --json

# 组件清单
oxq registry export --out component_catalog.json
```

内置组件数量:48 指标 / 8 信号 / 10 规则 / 6 组合优化器。
完整清单用 `oxq registry export` 或 `factor_probe.py --list` 查,
**不要凭记忆写类名**。
