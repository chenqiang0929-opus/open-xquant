# Factor Screening Skill

## Purpose

Screen stocks by indicator conditions from a specified index or custom universe.

## When to Use

User wants to filter/screen stocks by financial or technical indicator conditions.
Examples:
- "从沪深300中选择ROE大于15的股票"
- "Select CSI500 stocks with PB < 5 and momentum > 10"
- "筛选上证50中市盈率低于20的股票"

## Workflow

### Step 1: Clarify Universe

If user specifies an index (沪深300, CSI300, etc.):
```
→ universe_set(type="index", code="csi300")
```

If user provides a symbol list:
```
→ universe_set(type="static", symbols=[...])
```

If neither specified, ask: "Which universe? (e.g., csi300, csi500, sse50, or a list of symbols)"

### Step 2: Resolve Indicator Names

Use `resolve_alias()` to map Chinese names to canonical English:
- 市净率 → pb
- 市盈率 → pe_ttm
- 动量 → momentum
- 净资产收益率 → roe

### Step 3: Ensure Data Availability

For each indicator needed:

**Download-type indicators (roe, pe_ttm, pb, roa, peg):**
```
→ financial_download(symbol=sym, start=start, end=end, source="eastmoney",
                     indicators=["roe", "pe_ttm", "pb"])
```

**Compute-type indicators (momentum, volatility):**
1. Ensure OHLCV data is available: `data_load_symbols(symbols=..., source=...)`
2. Indicators will be computed by Engine from OHLCV data.

**Important:** Downloaded financial data needs to be merged into the mktdata wide table
before screening. Use `read_factor()` to load financial data, then merge relevant
columns into each symbol's OHLCV DataFrame.

### Step 4: Screen

```
→ universe_set(type="filter", symbols=symbols,
    filters=[
        {"column": "roe", "op": ">", "value": 15},
        {"column": "pb", "op": "<", "value": 10},
    ])
```

### Step 5: Present Results

Show the audit table (details) to the user for verification.
Offer next steps:
- "构建等权组合并回测？"
- "调整筛选条件？"
- "导出符合条件的股票列表？"
