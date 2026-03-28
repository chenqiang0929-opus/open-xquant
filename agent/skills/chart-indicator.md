---
name: chart-indicator
description: When a user wants to visually verify an indicator, use the chart_indicator tool to render a candlestick chart with indicator overlays
tools_required: [engine_run, chart_indicator]
---

## When to Use

After creating or debugging an Indicator, the user wants to **see** the output on a real price chart to verify correctness. This skill guides you through the visualization workflow.

## Workflow

### Step 1: Ensure Data Exists

The symbol's market data must be downloaded locally. Use `data_download` if needed:

```
data_download(symbols=["AAPL"], start="2024-01-01", end="2024-12-31")
```

### Step 2: Build a Minimal Strategy

Create a throwaway strategy with the indicator(s) to visualize:

```
strategy_create(name="viz", hypothesis="indicator visualization", objectives={"total_return": {"min": -1.0}})
strategy_add_signal(
    strategy="viz",
    name="dummy",
    type="Threshold",
    params={"column": "my_indicator", "threshold": 0, "direction": "above"},
    indicators={
        "my_indicator": {"type": "IchimokuTenkan", "params": {"period": 9}},
    },
)
```

**Tip:** Use a `Threshold` signal as a dummy — the signal itself doesn't matter, we just need the Engine to compute the indicator columns.

### Step 3: Run Through Indicator Phase

```
engine_run(strategy="viz", start="2024-01-01", end="2024-12-31", symbols=["AAPL"], run_through="indicator")
```

This populates `mktdata` with OHLCV + indicator columns without running the full backtest.

### Step 4: Chart

```
chart_indicator(run_id="viz_...", symbol="AAPL", columns=["my_indicator"], overlay=true)
```

- **overlay=true** — for price-scale indicators (SMA, Ichimoku, Bollinger)
- **overlay=false** — for oscillators or different-scale indicators (RSI, RollingVolatility, HurstExponent)

### Step 5: Read the Chart

Use the `Read` tool on the returned PNG path to visually inspect the chart.

## Multiple Indicators

You can plot multiple indicators at once:

```
chart_indicator(run_id="...", symbol="AAPL", columns=["IchimokuTenkan", "IchimokuKijun"], overlay=true)
```

## What to Look For

When verifying a new indicator:

1. **NaN region** — first N values should be NaN for period-based indicators
2. **Scale** — does the indicator range make sense relative to price?
3. **Shape** — does the curve behave as expected? (e.g., SMA should be smoother than price)
4. **Constant input** — try constant prices, the indicator should produce a flat or zero line
5. **Known patterns** — verify against known market patterns if possible
