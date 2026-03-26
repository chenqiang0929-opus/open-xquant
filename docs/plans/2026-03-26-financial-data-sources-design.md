# Financial Data Sources Design

## Goal

Add financial statement data fetching for A-shares (via EastMoney/akshare) and US stocks (via yfinance), following the same pattern as existing World Bank macro data. Design a unified Protocol so future data sources can be added with minimal effort.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| A-share data source | EastMoney via akshare | akshare already a dependency, stable wrapper |
| US stock data source | yfinance | Consistent with existing market data |
| US publish_date | None (Optional) | yfinance doesn't provide it; user can supplement |
| Time granularity | Both quarterly and annual via `period` param | Quarterly is finest grain; annual derived by filtering |
| report_date vs publish_date | Strictly separated | Prevent look-ahead bias in backtesting |
| Storage format | Wide table per symbol, parquet | Schema evolution friendly; `df["eps"]` direct access |
| Protocol placement | `data/providers.py` | Data protocols separate from engine pipeline protocols in `core/types.py` |
| File organization | Extend `factors.py` | Financial data and macro data are both factor data sources; differ only by source |

## Financial Indicators (8 total)

| Indicator | Source (A-share) | Source (US) |
|---|---|---|
| `eps` | stock_financial_abstract: 基本每股收益 | financials: Basic EPS |
| `revenue` | stock_financial_abstract: 营业总收入 | financials: Total Revenue |
| `net_income` | stock_financial_abstract: 净利润 | financials: Net Income |
| `roe` | stock_financial_abstract: 净资产收益率(ROE) | computed: net_income / equity |
| `book_value_per_share` | stock_financial_abstract: 每股净资产 | computed: equity / shares |
| `operating_cash_flow` | stock_financial_abstract: 经营现金流量净额 | cashflow: Operating Cash Flow |
| `total_assets` | computed: 股东权益 × 权益乘数 | balance_sheet: Total Assets |
| `total_shares` | computed: 股东权益 / 每股净资产 | balance_sheet: Ordinary Shares Number |

## Architecture

### Protocol Layer (`src/oxq/data/providers.py`)

```python
@runtime_checkable
class FactorFetcher(Protocol):
    """Unified factor data source interface.

    Each data source (WorldBank, EastMoney, yfinance) implements one Fetcher.
    Swapping data sources only requires implementing this interface.
    """
    def fetch(
        self,
        target: str,           # indicator name or symbol
        start: str,
        end: str,
        **kwargs: Any,
    ) -> pd.DataFrame: ...

    def list_indicators(self) -> list[str]: ...
```

Existing `Downloader` Protocol moves here from `loaders.py` (re-exported for compatibility).

### Factor Module (`src/oxq/data/factors.py`)

Extended from "WorldBank only" to "all factor data sources":

**WorldBankFetcher** (refactored from WorldBankDownloader):
- Implements `FactorFetcher`
- Returns `DataFrame(index=year, columns=countries)`
- `WorldBankDownloader` kept as alias for backward compatibility

**EastMoneyFetcher** (new):
- Implements `FactorFetcher`
- A-share financial data via `stock_financial_abstract` (per-symbol, single API call)
- `EASTMONEY_FIELD_MAP`: indicator -> chinese metric label (or None for computed)
- `total_assets` computed from 股东权益 × 权益乘数; `total_shares` computed from 股东权益 / 每股净资产
- `publish_date` = NaT (not available from this API; user can supplement manually)
- Returns wide DataFrame: `index=report_date, columns=[publish_date, period, eps, revenue, ...]`

**YFinanceFinancialFetcher** (new):
- Implements `FactorFetcher`
- US financial data via yfinance Ticker
- `YFINANCE_FIELD_MAP`: indicator -> (report_type, field_name) or None (computed)
- `roe` and `book_value_per_share` computed from other fields
- `publish_date` = NaT (not available from yfinance)

**FactorDownloader** (new):
- Accepts any `FactorFetcher` + `sub` category string
- Handles: fetch -> validate -> store parquet -> incremental merge/dedup
- Storage: `~/.oxq/data/factor/{sub}/{target}.parquet`
  - macro: `~/.oxq/data/factor/macro/gdp.parquet`
  - financial: `~/.oxq/data/factor/financial/600519.parquet`

**read_factor()** (enhanced):
- New params: `sub="macro"` (default, backward compatible), `point_in_time=False`
- `point_in_time=True`: filters by `publish_date` instead of `report_date` to prevent look-ahead bias

### Adapters (`src/oxq/data/adapters.py`)

```python
class MarketDataAdapter:
    """Adapts YFinanceDownloader/AkShareDownloader to FactorFetcher Protocol."""
```

> **Design note:** The original design included a `WorldBankAdapter`, but during implementation
> the `factor_download` tool was migrated directly to `FactorDownloader(WorldBankFetcher(), sub="macro")`,
> making the adapter unnecessary. The adapter was dropped in favor of direct usage.

### Tools Layer (`src/oxq/tools/data.py`)

Three new tools:
- `financial_download` — download financial data, routes to EastMoneyFetcher or YFinanceFinancialFetcher
- `financial_list` — list local financial data files
- `financial_inspect` — inspect a symbol's financial data (date range, indicators, sample)

Existing `factor_download` tool: signature unchanged, internal implementation migrated to `FactorDownloader(WorldBankFetcher(), sub="macro")` directly (no adapter needed).

### Parquet Schema (financial)

| Column | Type | Description |
|---|---|---|
| `report_date` (index) | datetime | Reporting period end (e.g. 2024-06-30) |
| `publish_date` | datetime \| NaT | Financial report release date |
| `period` | str | "quarterly" \| "annual" |
| `total_shares` | float | |
| `eps` | float | |
| `book_value_per_share` | float | |
| `net_income` | float | |
| `operating_cash_flow` | float | |
| `total_assets` | float | |
| `revenue` | float | |
| `roe` | float | |

Wide table — new indicators in the future simply add columns. Parquet schema evolution handles missing columns as NaN when reading old files.

## Backward Compatibility

- `WorldBankDownloader` = alias for `WorldBankFetcher`
- `INDICATOR_MAP` = re-export of `MACRO_INDICATOR_MAP`
- `Downloader` Protocol re-exported from `loaders.py`
- `read_factor()` default `sub="macro"` — existing calls unchanged
- `factor_download` tool signature unchanged

## File Change Summary

| File | Action | Description |
|---|---|---|
| `src/oxq/data/providers.py` | Modify | Add `Downloader` (moved from loaders) + new `FactorFetcher` Protocol |
| `src/oxq/data/factors.py` | Modify | Refactor to unified factor module: WorldBankFetcher + EastMoneyFetcher + YFinanceFinancialFetcher + FactorDownloader + read_factor enhancement |
| `src/oxq/data/adapters.py` | Create | MarketDataAdapter (WorldBankAdapter dropped — direct migration) |
| `src/oxq/data/loaders.py` | Minor | Downloader re-exported from providers |
| `src/oxq/data/__init__.py` | Modify | New exports + backward compat aliases |
| `src/oxq/tools/data.py` | Modify | Add financial_download/list/inspect; migrate factor_download internals |
| `tests/data/test_fetchers.py` | Create | EastMoneyFetcher + YFinanceFinancialFetcher unit tests |
| `tests/data/test_factor_downloader.py` | Create | FactorDownloader storage/merge tests |
| `tests/data/test_adapters.py` | Create | Adapter tests |

3 new files, 5 modified files. No breaking changes.

## Test Strategy

- **Fetcher tests**: mock akshare/yfinance, verify field mapping, report_date/publish_date extraction, period filtering
- **FactorDownloader tests**: inject mock Fetcher, verify parquet write, incremental merge/dedup, subdirectory paths
- **read_factor tests**: write temp parquet, verify `point_in_time=True` filters by publish_date, indicator column selection, backward compat (`sub="macro"` default)
- **Adapter tests**: verify old Downloader satisfies FactorFetcher Protocol via adapter
