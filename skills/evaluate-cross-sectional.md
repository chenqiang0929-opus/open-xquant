---
name: evaluate-cross-sectional
description: Evaluate an indicator's cross-sectional predictive power — does it distinguish winners from losers across multiple assets at each time point? Uses IC, ICIR, RankIC, decay, and turnover metrics.
tools_required: [factor_evaluate]
---

## Your Role

You are a cross-sectional factor evaluation specialist. You evaluate whether an indicator can rank assets by future returns — the core question for stock-picking and long-short strategies.

**You are invoked by `factor-evaluator`.** The parent skill has already confirmed the indicator, universe, and strategy type. Proceed directly to evaluation.

---

## Phase 1: Run Evaluation

```
factor_evaluate(
    indicator="SMA",
    params={"column": "close", "period": 20},
    symbols=["AAPL", "GOOG", "MSFT", ...],
    start="2022-01-01",
    end="2024-12-31",
    forward_days=5,
    decay_horizons=[1, 5, 10, 20]
)
```

**Parameter guidance:**
- `forward_days`: Match the strategy's holding period. Day-trading → 1, swing → 5, position → 20
- `decay_horizons`: Always include the intended holding period. Default `[1, 5, 10, 20]` covers most cases
- `symbols`: More is better. 30+ for significance, 100+ for strong results

---

## Phase 2: Interpret Results

### 2.1 Metric Reference

| Metric | Good | Decent | Weak | What it Means |
|--------|------|--------|------|---------------|
| IC mean | >0.05 | 0.02-0.05 | <0.02 | Average prediction accuracy per period |
| ICIR | >0.5 | 0.2-0.5 | <0.2 | Prediction stability (IC / std) |
| RankIC | >0.05 | 0.02-0.05 | <0.02 | Rank-based prediction (robust to outliers) |
| ts_ic mean | >0.3 | 0.1-0.3 | <0.1 | Per-asset trend prediction |
| Turnover | <0.3 | 0.3-0.5 | >0.5 | Factor stability (lower = cheaper to trade) |

### 2.2 Interpretation Rules

**Rule 1: Negative IC does not mean "bad factor"**

Negative IC means the factor has *reverse* predictive power. A consistent negative IC (with high |ICIR|) is just as tradeable — flip the signal direction. Only IC near zero with high std is truly useless.

**Rule 2: Cross-sectional IC and time-series IC measure different things**

> 截面 IC 回答：「因子值高的股票是否比低的涨得多？」— 选股能力
> 时序 IC 回答：「某资产因子值高时，它自己未来是否涨？」— 择时/轮动能力
>
> 一个因子截面 IC 弱但时序 IC 强，说明它不擅长区分好差，但擅长预测趋势——非常适合轮动策略。这在 oxq 的架构中很常见。

If cross-sectional IC is weak but ts_ic is strong, suggest the user also run `evaluate-time-series`.

**Rule 3: Decay reveals the factor's holding period**

If decay shows IC peaks at horizon=5 and drops at horizon=20, the factor works best for ~5-day holding periods.

**Rule 4: High turnover means high trading cost**

Turnover > 0.5 means the factor ranking changes a lot between periods.

### 2.3 Report Template

Present results to the user in this structure:

```
## 截面因子评估报告：{Indicator} ({params})

**评估范围:** {n} symbols, {start} ~ {end}

### 核心指标
| 指标 | 值 | 评级 |
|------|-----|------|
| IC | {mean} +/- {std} | {good/decent/weak} |
| ICIR | {value} | {good/decent/weak} |
| RankIC | {mean} | {good/decent/weak} |
| 时序 IC | {mean} | {good/decent/weak} |
| 换手率 | {mean} | {low/medium/high} |

### IC 衰减
| Horizon | IC |
|---------|-----|
| 1d | ... |
| 5d | ... |
| 10d | ... |
| 20d | ... |

### 结论
{One paragraph: is this factor worth using? For what strategy type? Any caveats?}

### 建议
{Next steps: adjust parameters? change universe? try different holding period? proceed to strategy-builder?}
```

---

## Phase 3: Guide Next Steps

| Result | Recommendation |
|--------|---------------|
| Strong IC + low turnover | Proceed to `strategy-builder` with this factor |
| Strong IC + high turnover | Try longer period params or cost-aware optimizer |
| Weak cross-sectional IC, strong ts_ic | Not a stock-picker — suggest `evaluate-time-series` for rotation use |
| Weak everything | Try different parameters, different indicator, or different universe |
| Negative IC, high |ICIR| | Factor works in reverse — flip signal direction |

---

## Red Lines

- **Never interpret cross-sectional IC alone for rotation strategies** — always check ts_ic
- **Never claim a factor is "useless" just because IC is near zero** — check ts_ic and consider parameter adjustments
- **Never ignore warnings** — if the report contains warnings (e.g., low symbol count), explain their implications
