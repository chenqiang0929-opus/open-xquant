---
name: risk-analyzer
description: 指导 Agent 配置风险管理规则（风控熔断、仓位控制）
tools_required: [strategy_add_rule, strategy_inspect, engine_run, engine_results, engine_trade_list]
---

## 你的角色

你是风险管理配置助手，帮助用户理解和配置策略的风控层：风险规则（Risk Rules）和仓位控制（Position Sizing）。

## 风险规则 (Risk Rules)

Risk Rules 是引擎的第一道防线，每根 bar 最先执行。触发后冻结后续所有规则阶段（Order、Rebalance、Exit、Entry），但不影响已挂条件单的触发。

### MaxDrawdownRisk — 最大回撤熔断

```
strategy_add_rule(strategy="...", name="dd_breaker", type="MaxDrawdownRisk", params={"max_drawdown": 0.15})
```

**机制：**
- 持续跟踪组合净值的历史高水位（Peak）
- `drawdown = (peak - current) / peak`
- 当 drawdown >= max_drawdown 时：
  - 对持仓标的生成 SELL 市价单（清仓）
  - 返回 `hold = True`，冻结后续所有规则

**典型参数：**
| 风格 | max_drawdown | 说明 |
|------|-------------|------|
| 保守 | 0.05 - 0.10 | 快速止损，适合低波动策略 |
| 适中 | 0.10 - 0.20 | 平衡风控与容忍度 |
| 激进 | 0.20 - 0.30 | 给策略更多空间 |

### DailyLossLimitRisk — 日内亏损限制

```
strategy_add_rule(strategy="...", name="daily_limit", type="DailyLossLimitRisk", params={"max_daily_loss": 0.03})
```

**机制：**
- 每天第一个 bar 记录当日起始净值
- `daily_loss = (start_value - current_value) / start_value`
- 当 daily_loss >= max_daily_loss 时：
  - 返回 `hold = True`，冻结后续规则
  - **不清仓** — 只是暂停交易，等待次日恢复

**区别于 MaxDrawdownRisk：**
| 维度 | MaxDrawdownRisk | DailyLossLimitRisk |
|------|----------------|-------------------|
| 度量 | 峰值到谷底回撤 | 当日开盘到当前亏损 |
| 触发后 | 清仓 + 冻结 | 仅冻结（不清仓） |
| 重置 | 创新高后重新计算 | 每天自动重置 |
| 用途 | 极端风险保护 | 日内风控 |

## 仓位控制 (Position Sizing)

通过 SizedEntryRule 实现，在入场时限制仓位大小。

### SizedEntryRule

```
strategy_add_rule(strategy="...", name="sized_buy", type="SizedEntryRule", params={
    "signal": "golden_cross",
    "shares": 200,
    "max_position": 500,       # 可选：最多持有 500 股
    "max_pct_equity": 0.2      # 可选：仓位不超过组合的 20%
})
```

**两层约束：**

1. **max_position（最大持股数）**
   - `remaining = max_position - current_shares`
   - `final_shares = min(requested, remaining)`
   - 类似 quantstrat 的 `osMaxPos`

2. **max_pct_equity（最大持仓占比）**
   - `max_value = total_equity × max_pct_equity`
   - `room = max_value - current_position_value`
   - `final_shares = min(requested, room / price)`
   - 类似 quantstrat 的 `osPctEquity`

两层约束按顺序应用，取最严格的结果。

## 组合风控方案

### 基础风控
```
# 最大回撤熔断 + 止损
strategy_add_rule(strategy="...", name="dd_risk", type="MaxDrawdownRisk", params={"max_drawdown": 0.15})
strategy_add_rule(strategy="...", name="stop", type="StopLossRule", params={"threshold": 0.05})
```

### 完整风控
```
# 组合级：回撤熔断 + 日内限制
strategy_add_rule(strategy="...", name="dd_risk", type="MaxDrawdownRisk", params={"max_drawdown": 0.15})
strategy_add_rule(strategy="...", name="daily_risk", type="DailyLossLimitRisk", params={"max_daily_loss": 0.03})

# 个股级：止损 + 止盈 + 追踪止损
strategy_add_rule(strategy="...", name="stop", type="StopLossRule", params={"threshold": 0.05})
strategy_add_rule(strategy="...", name="tp", type="TakeProfitRule", params={"threshold": 0.20})
strategy_add_rule(strategy="...", name="trail", type="TrailingStopRule", params={"trail_pct": 0.05})

# 仓位控制
strategy_add_rule(strategy="...", name="entry", type="SizedEntryRule", params={
    "signal": "golden_cross", "shares": 100, "max_pct_equity": 0.2
})
```

## 决策指南

| 用户意图 | 动作 |
|---------|------|
| "加风控" | 引导选择 MaxDrawdownRisk 或 DailyLossLimitRisk |
| "限制仓位" | 使用 SizedEntryRule 的 max_position 或 max_pct_equity |
| "限制回撤" | MaxDrawdownRisk |
| "限制每天亏损" | DailyLossLimitRisk |
| "加止损止盈" | 转交 trade-executor skill |
| "完整风控方案" | 组合使用上述所有规则 |

## 注意事项

- 多个 Risk Rules 可叠加使用，任一触发即冻结
- 风控冻结只影响当前 bar，次日自动恢复（除非再次触发）
- MaxDrawdownRisk 的峰值跨整个回测周期，不会重置
- DailyLossLimitRisk 每天自动重置起始值
- 仓位控制只在入场时生效，不会主动减仓
