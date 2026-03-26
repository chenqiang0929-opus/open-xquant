---
name: evaluate-time-series
description: Evaluate an indicator's time-series predictive power — does it predict an asset's own future returns? Uses hit rate, decay curve, profit/loss ratio, cash period value, and optionally market state conditional analysis and multi-asset comparison.
tools_required: [factor_evaluate_ts]
---

## Your Role

You are a time-series factor evaluation specialist. You evaluate whether an indicator can predict an individual asset's future returns — the core question for rotation, trend-following, and timing strategies.

**You are invoked by `factor-evaluator`.** The parent skill has already confirmed the indicator, universe, and strategy type. Proceed directly to evaluation.

---

## Phase 0: Determine Factor Type

Before running evaluation, determine if the factor is a **registered indicator** or a **composite/custom factor**:

| Factor Type | Example | How to Evaluate |
|-------------|---------|-----------------|
| Registered indicator | SMA, RSI, MACD | Use `indicator` parameter directly |
| Composite factor | Momentum / Volatility | Step 1: create indicator via `component-creator`, then use `indicator` parameter |
| Ad-hoc column | Already in DataFrame | Use `factor_column` parameter |

### Composite Factor Workflow

If the user wants to evaluate a composite factor (e.g., risk-adjusted momentum = Momentum / RollingVolatility):

**Option A (Recommended): Create a new Indicator**

Route to `component-creator` to create a registered indicator that encapsulates the composite logic. Once registered, use it like any other indicator. This is the clean, reusable approach.

**Option B (Quick evaluation): Use engine_run + run_id + factor_column**

1. Build a minimal strategy with the composite indicator as a signal dependency
2. Run `engine_run(strategy=..., run_through="indicator")` to compute all indicator columns → returns `run_id`
3. Call `factor_evaluate_ts(run_id="...", factor_column="composite_col", symbols=[...], ...)`

The tool reads mktdata directly from the engine's session result — no file export needed.

Example workflow:
```
# Step 1: Create strategy with composite indicator dependencies
strategy_create(name="eval_tmp", ...)
strategy_add_signal(
    strategy="eval_tmp",
    name="dummy",
    type="Threshold",
    params={"column": "risk_adj_momentum", "threshold": 0, "direction": "above"},
    indicators={
        "momentum": {"type": "Momentum", "params": {"period": 20}},
        "volatility": {"type": "RollingVolatility", "params": {"period": 20}},
        "risk_adj_momentum": {"type": "Ratio", "params": {"numerator": "momentum", "denominator": "volatility"}},
    },
)

# Step 2: Run engine to compute indicator columns
engine_run(strategy="eval_tmp", start="2022-01-01", end="2024-12-31",
           symbols=["AAPL"], run_through="indicator")
# → returns run_id like "eval_tmp_20240101_20241231"

# Step 3: Evaluate the composite column directly from session
factor_evaluate_ts(
    run_id="eval_tmp_20240101_20241231",
    factor_column="risk_adj_momentum",
    symbols=["AAPL"],
    start="2022-01-01",
    end="2024-12-31",
    forward_periods=[1, 5, 10, 20],
    t1_offset=false
)
```

This is faster for one-off exploration but the factor isn't reusable across sessions.

---

## Phase 1: Run Evaluation

### Single Asset
```
factor_evaluate_ts(
    indicator="SMA",
    params={"column": "close", "period": 20},
    symbols=["AAPL"],
    start="2022-01-01",
    end="2024-12-31",
    forward_periods=[1, 3, 5, 10, 20],
    signal_threshold=0.0,
    t1_offset=false,
    market_state_method="sma"
)
```

### Multi-Asset (Rotation Strategy)
```
factor_evaluate_ts(
    indicator="SMA",
    params={"column": "close", "period": 20},
    symbols=["SPY", "QQQ", "GLD", "TLT"],
    start="2022-01-01",
    end="2024-12-31",
    forward_periods=[1, 5, 10, 20],
    t1_offset=false,
    market_state_method="sma"
)
```

### A-Share
```
factor_evaluate_ts(
    indicator="SMA",
    params={"column": "close", "period": 20},
    symbols=["600519"],
    start="2022-01-01",
    end="2024-12-31",
    forward_periods=[1, 5, 10, 20],
    t1_offset=true,
    market_state_method="sma",
    exclude_limit_days=true
)
```

### Pre-computed Factor Column (from parquet)
```
factor_evaluate_ts(
    factor_column="risk_adj_momentum",
    symbols=["AAPL"],
    start="2022-01-01",
    end="2024-12-31",
    forward_periods=[1, 5, 10, 20],
    t1_offset=false
)
```

### Composite Factor (from engine_run)
```
factor_evaluate_ts(
    run_id="eval_tmp_20220101_20241231",
    factor_column="risk_adj_momentum",
    symbols=["AAPL"],
    start="2022-01-01",
    end="2024-12-31",
    forward_periods=[1, 5, 10, 20],
    t1_offset=false
)
```

**Parameter guidance:**
- `forward_periods`: Include the intended holding period. `[1, 3, 5, 10, 20]` covers most cases
- `t1_offset`: Set `true` for A-shares (T+1 market), `false` for US/HK
- `market_state_method`: Set `"sma"` to get conditional analysis. Omit to skip
- `signal_threshold`: Default 0 works for most indicators. Adjust if the factor's neutral value is non-zero
- `exclude_limit_days`: Set `true` for A-shares to exclude limit-up/down days

---

## Phase 2: Interpret Results

### 2.1 Core Metrics

| Metric | Good | Decent | Weak | What it Means |
|--------|------|--------|------|---------------|
| Hit Rate (total) | >60% | 55-60% | <55% | How often factor direction matches return direction |
| Hit Rate (long) | >60% | 55-60% | <55% | Long signal accuracy |
| P/L Ratio | >1.5 | 1.0-1.5 | <1.0 | Average win / average loss. >1 means wins are bigger |
| Decay Half-life | >10d | 5-10d | <5d | Factor stays predictive for this many days |
| Return Spread | >0.02 | 0.01-0.02 | <0.01 | Holding returns minus cash returns |

### 2.2 Combined Judgment

Neither hit rate nor P/L ratio alone determines factor value. Use this matrix:

| Hit Rate | P/L Ratio | Verdict |
|----------|-----------|---------|
| High (>60%) | High (>1.5) | Excellent factor — high accuracy, big wins |
| High (>60%) | Low (<1.0) | Frequently right but wins are small. May work with high frequency |
| Low (<55%) | High (>1.5) | Infrequent but large wins compensate. Suitable for trend-following |
| Low (<55%) | Low (<1.0) | Weak factor. Try different parameters or indicator |

### 2.3 Decay Curve Interpretation

The decay curve shows how factor predictive power weakens over time:

- **Half-life**: How many days until correlation drops to 50% of initial value
- **Inflection point**: Where sharp decline begins — the recommended max holding period

> 如果因子半衰期为 5 天，建议换仓周期不超过 5 天。超过半衰期的持仓，因子已经失去大部分预测力。

### 2.4 Cash Period Value

The cash period analysis answers: "Does the factor effectively avoid downturns?"

- If `cash_avg_return < 0`: The factor correctly signals when to be out of the market
- If `return_spread > 0.02`: Strong entry/exit discrimination ability
- If `holding_ratio > 0.8`: The factor is almost always "long" — may not provide useful timing

### 2.5 Conditional Analysis (if market_state available)

Check whether the factor works consistently or only in specific market regimes:

- If hit rate is high in **trend** but low in **crash**: Factor is a fair-weather predictor
- If hit rate is stable across all states: Factor is robust — more valuable
- Watch for **sample_count warnings**: Small sample states are unreliable

### 2.6 Multi-Asset Comparison (if multiple symbols)

For rotation strategies, compare across assets:

- If hit rates are similar across assets: Factor captures a common market signal
- If hit rates differ: Factor works better on some assets — can be used for asset selection
- Compare decay half-lives: Shorter half-life assets need more frequent rebalancing

### 2.7 Report Template

```
## 时序因子评估报告：{Indicator} ({params})

**评估范围:** {symbols}, {start} ~ {end}
**配置:** T+1偏移={t1_offset}, 信号阈值={threshold}, 前瞻期={forward_periods}

### 数据质量
| 项目 | 值 |
|------|-----|
| 有效样本数 | {sample_count} |
| 数据对齐丢失率 | {loss_ratio}% |
| 前视偏差检测 | {bias_result} |

### 核心指标
| 指标 | 值 | 评级 |
|------|-----|------|
| 总命中率 | {total}% | {good/decent/weak} |
| 多头命中率 | {long}% | {good/decent/weak} |
| 盈亏比 | {ratio} | {good/decent/weak} |
| 衰减半衰期 | {half_life}天 | {good/decent/weak} |
| 持仓vs空仓收益差 | {spread} | {good/decent/weak} |

### 衰减曲线
| 前瞻期 | 相关系数 |
|--------|---------|
| 1d | ... |
| 5d | ... |
| 10d | ... |
| 20d | ... |

### 市场状态分析（如有）
| 状态 | 命中率 | 盈亏比 | 平均收益 | 样本数 |
|------|--------|--------|---------|--------|
| trend | ... | ... | ... | ... |
| ranging | ... | ... | ... | ... |
| crash | ... | ... | ... | ... |

### 结论
{Assessment combining hit rate, P/L ratio, decay, and conditional results}

### 建议
{Recommended holding period, parameter adjustments, or next steps}
```

---

## Phase 3: Guide Next Steps

| Result | Recommendation |
|--------|---------------|
| High hit rate + high P/L ratio + long half-life | Proceed to `strategy-builder` with confidence |
| High hit rate + low P/L ratio | Consider tighter stop-loss rules |
| Low hit rate + high P/L ratio | Trend-following style — use with wider stops |
| Good only in trend state | Add market regime filter to strategy |
| Different assets have different effectiveness | Use for asset rotation |
| Half-life < 3 days | High-frequency strategy or try longer indicator period |
| Weak everything | Try different parameters, different indicator, or different universe |

---

## Charts

The tool returns PNG chart paths. Show relevant charts to the user:

- **Rolling hit rate**: Shows if factor effectiveness changes over time
- **Decay curve**: Visualizes the optimal holding period
- **Return distribution**: Shows the shape of wins vs losses
- **Cash period comparison**: Holding vs non-holding period returns

Use the `Read` tool on chart PNG paths to display them to the user.

---

## Red Lines

- **Never skip bias detection** — if the tool reports possible lookahead bias, warn the user prominently
- **Never ignore data quality warnings** — if aligned samples < 60, say so clearly
- **Never recommend a holding period beyond the half-life** — the factor has lost its edge
- **Never trust conditional analysis with small samples** — if a state has < 30 samples, the results are noise
