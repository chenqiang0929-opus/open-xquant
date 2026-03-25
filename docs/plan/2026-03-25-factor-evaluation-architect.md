## 系统架构方案：因子评估工具（Factor Evaluation）

### 一句话定义

开发者引入一个 Indicator 后，通过一个标准化工具评估其对未来收益的预测力（IC/ICIR/RankIC/衰减/换手率），输出结构化报告，帮助决定是否采用。

### 核心组件

1. **factor_eval 计算模块** (`src/oxq/factor_eval/metrics.py`)
   纯函数模块，包含所有评估指标的计算逻辑。输入 factor values + forward returns（已按截面对齐），输出指标数值。不依赖 Engine。

2. **factor_evaluate Tool** (`src/oxq/tools/factor_eval.py`)
   面向开发者和 Agent 的入口。负责编排完整流程：加载数据 → per-symbol 计算 Indicator → 构建截面 → 计算 forward returns → 调用计算模块 → 组装结构化报告。通过 `@registry.tool` 注册，自动暴露为 MCP 工具。

### 数据流

```
开发者传入: indicator_name + symbols + date_range + eval_config
    |
    v
[1] 加载市场数据: MarketDataProvider.get_bars() per symbol
    → dict[str, pd.DataFrame]  (与 Engine 相同的 wide table 数据模型)
    |
    v
[2] 计算因子值: indicator.compute(mktdata[symbol], **params) per symbol
    → 每个 symbol 的 DataFrame 新增一列因子值
    |
    v
[3] 构建截面数据: 按日期对齐所有 symbol 的因子值 + close 价格
    → pd.DataFrame (index=date, columns=symbols) for factor
    → pd.DataFrame (index=date, columns=symbols) for forward returns
    |
    v
[4] 计算评估指标: compute_ic(), compute_rank_ic(), compute_decay(), compute_turnover()
    → 每个函数是纯函数，输入截面数据，输出数值
    |
    v
[5] 组装结构化报告: dict[str, Any]
    → 返回给开发者 / Agent
```

### 结构化报告格式

```python
{
    "indicator": "SMA",
    "params": {"period": 20},
    "symbols_count": 300,
    "date_range": {"start": "2024-01-01", "end": "2025-12-31"},
    "warnings": ["symbols_count < 30: statistical significance is weak"],

    "metrics": {
        "ic": {
            "mean": 0.035,
            "std": 0.12,
            "description": "per-period Pearson correlation, averaged"
        },
        "icir": {
            "value": 0.29,
            "description": "IC mean / IC std, measures stability"
        },
        "rank_ic": {
            "mean": 0.041,
            "description": "per-period Spearman rank correlation, averaged"
        },
        "decay": {
            "horizons": [1, 5, 10, 20],
            "ic_values": [0.035, 0.028, 0.015, 0.005],
            "description": "IC at different forward return horizons"
        },
        "turnover": {
            "mean": 0.32,
            "description": "average rank change ratio per period"
        }
    },

    "ic_series": {
        "dates": ["2024-01-02", "2024-01-03", "..."],
        "values": [0.05, -0.02, "..."],
        "description": "per-period IC time series, for plotting or further analysis"
    }
}
```

### 关键架构决策

**决策 1：计算模块独立于 Engine，不复用 Engine 代码**

选择：factor_eval 计算模块直接调用 `indicator.compute()`，自己管理 per-symbol 循环和截面构建。

为什么：Engine 的 Indicator 计算与 Strategy 收集逻辑、Tracer、Signal/Portfolio 流程耦合。因子评估是一个独立的分析场景，不应依赖回测 pipeline 的任何状态。

放弃了什么：不复用 Engine 的 Indicator 循环代码（约 20 行），接受少量重复换取解耦。

**决策 2：计算逻辑与 Tool 入口分层**

选择：纯函数计算模块 (`src/oxq/factor_eval/metrics.py`) + Tool 编排层 (`src/oxq/tools/factor_eval.py`) 分离。

为什么：计算模块可被 notebook、脚本、测试直接 import 使用，不绑定 Tool 协议。Tool 层负责参数解析、数据加载、错误处理、报告格式化。

放弃了什么：把所有逻辑写在一个 Tool 函数里会更简单，但会导致计算逻辑只能通过 Tool 调用，不利于 ML pipeline 等场景复用。

**决策 3：forward returns 在工具内部计算，不由用户传入**

选择：用户只传 indicator + symbols + date_range，工具内部用 close 价格计算 forward returns。

为什么：减少用户出错的可能（forward return 计算涉及对齐、NaN 处理），保证评估流程的一致性。衰减分析需要多个 horizon 的 forward returns，全部内部生成。

放弃了什么：灵活性——用户不能传入自定义的 forward returns（比如行业中性化收益）。如果未来有此需求，可以加一个可选参数。

**决策 4：单 symbol 支持但给出警告**

选择：不拒绝单 symbol 输入，但在报告的 `warnings` 字段中标明统计意义不足。

为什么：有些场景（如 ETF、期货）确实只有一个标的。此时 IC 退化为时序相关性，仍有参考价值，但用户需要知道局限性。

阈值：symbols < 30 时触发警告。

**决策 5：eval_config 控制评估行为**

选择：通过配置控制 forward return 周期、衰减分析的 horizons 等参数，提供合理默认值。

```python
eval_config = {
    "forward_days": 5,                  # default forward return horizon
    "decay_horizons": [1, 5, 10, 20],   # horizons for decay analysis
    "min_periods": 3,                   # min observations per period for IC
}
```

为什么：不同因子适用不同的评估周期（动量因子可能看 20 日，短期反转看 1 日）。默认值覆盖最常见场景，高级用户可调整。

**决策 6：同时支持截面评估和时序评估**

选择：在报告中同时输出截面 IC（cross-sectional IC）和时序 IC（time-series IC）。

为什么：oxq 的架构天然偏爱轮动类策略（多资产 Universe → per-symbol Indicator → 截面排序 → 轮动）。轮动策略的因子有效性应通过时序 IC 评估——「某资产因子值高时，它自己未来是否涨」，而非截面 IC——「因子高的资产是否比因子低的涨得多」。两者衡量的是不同能力：

| | 截面 IC (cross-sectional) | 时序 IC (time-series) |
|---|---|---|
| 问什么 | 因子能区分哪个好哪个差吗？ | 因子能预测某个资产自身的涨跌吗？ |
| 适用策略 | 个股选股、多空对冲 | 轮动、趋势跟踪、择时 |
| 需要 symbol 数 | 多（>30 才有统计意义） | 少也行（单 symbol 即可） |

计算方式：对每个 symbol 单独算因子值与自身未来收益的 Pearson 相关系数（跨时间），然后取所有 symbol 的平均。

```python
def compute_ts_ic(factor, forward_returns, min_obs=30):
    """Time-series IC: per-symbol Pearson(factor, fwd_return) across time, averaged."""
    for symbol in factor.columns:
        corr = Pearson(factor[symbol], forward_returns[symbol])  # 跨时间
        ...
    return mean(per_symbol_corrs)
```

放弃了什么：不做更复杂的时序评估（如分位数回测、多空收益曲线）。这些可以通过后续的 Engine 回测完成。

### 已知风险与边界

1. **不处理截面标准化**：评估工具直接用 Indicator 原始输出值计算 IC。如果因子值在不同股票间尺度差异大（如市值），IC 可能失真。这属于因子设计问题，不在评估工具的职责范围内。未来可以考虑加一个可选的 `normalize` 参数。

2. **不处理停牌/缺失数据的复杂场景**：如果某些 symbol 在某些日期无数据，截面构建时会自然产生 NaN，IC 计算时跳过。但不做主动的停牌标记或复权处理——这是数据层的责任。

3. **不包含可视化**：报告是结构化数据，不生成图表。IC 时序、衰减曲线等可视化可以后续作为 `chart_factor_eval` Tool 单独实现，消费结构化报告的数据。

### 新增文件

| 文件 | 职责 |
|------|------|
| `src/oxq/factor_eval/__init__.py` | 导出公共 API |
| `src/oxq/factor_eval/metrics.py` | 纯函数计算模块：compute_ic, compute_rank_ic, compute_icir, compute_decay, compute_turnover |
| `src/oxq/tools/factor_eval.py` | Tool 入口：factor_evaluate, 编排数据加载→计算→报告 |
| `tests/factor_eval/test_metrics.py` | 计算模块测试：手算验证 IC/ICIR/RankIC |
| `tests/tools/test_factor_eval_tool.py` | Tool 集成测试：端到端评估流程 |

### 下一步建议

1. **先实现计算模块** (`src/oxq/factor_eval/metrics.py`)，用 TDD——构造已知相关性的合成数据，手算 IC/RankIC 期望值，写测试先行。
2. **再实现 Tool 层** (`src/oxq/tools/factor_eval.py`)，复用现有 `data_load_symbols` 的数据加载模式，注册为 `factor_evaluate` Tool。
3. **暂不做可视化**——结构化报告已经足够 Agent 和开发者消费，图表可以后续用 `chart_factor_eval` 独立实现。
