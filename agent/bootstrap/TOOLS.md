# TOOLS.md - Environment & Tool Reference

Skills define _how_ tools work. This file documents _this specific setup_ —
what's available, where things live, and how to reach them.

---

## open-xquant MCP Server

The primary way to interact with open-xquant is via its built-in MCP server,
which exposes the full `oxq` SDK as callable tools.

- **Repo**: github.com/xingwudao/open-xquant
- **PyPI package**: `open-xquant` (import as `oxq`)
- **Python requirement**: >= 3.12
- **Package manager**: `uv` (preferred), `pip` as fallback
- **MCP server entry**: `agent/mcp_server/server.py`

Verify the MCP server is reachable before starting any research session.
If tools are unavailable, check whether the MCP server process is running.

---

## oxq Core Pipeline

完整执行管道：

```
Indicator → Universe → Signal → Portfolio → Pre-trade Rule
    → Trading Algorithm → Broker → Post-trade Rule
```

两个阶段：
- **向量化阶段** (setup): Indicator + Signal 对全量时间序列一次计算
- **逐 bar 阶段** (step): Portfolio → Rule → Trading → Broker 逐步推进

Key interfaces (all use Protocol over ABC — prefer structural typing):
- `Indicator.compute(df) → Series` — 纯函数，输出连续数值
- `Signal.compute(df) → Series` — 纯函数，输出离散标签（buy/hold/sell）
- `PortfolioOptimizer.optimize(signals, indicators) → dict[str, float]` — 截面优化，输出目标权重
- `Rule.evaluate(symbol, row, portfolio) → RuleResult` — 逐 bar 有状态，输出约束/减仓意图

Universe: `StaticUniverse` (fixed pool), `FilterUniverse` (dynamic screening)
Data providers: `LocalMarketDataProvider` (unified read interface)
Data sources: YFinance (US equities), AkShare (A-shares), WorldBank (macro factors)

---

## SDK Tools (62 tools across 10 modules)

All tools are defined in `oxq.tools`, protocol-agnostic. MCP Server auto-registers them.

### strategy (15 tools)

| Tool | Description |
|------|-------------|
| `strategy_create` | Create a new strategy with hypothesis and objectives |
| `strategy_list` | List all strategies in the current session |
| `strategy_inspect` | Inspect a strategy definition (indicators, signals, rules) |
| `strategy_add_signal` | Add a signal with its dependent indicators |
| `strategy_add_rule` | Add a rule with its dependent indicators |
| `strategy_set_universe` | Set universe (static list or filter-based screening) |
| `strategy_set_portfolio` | Set portfolio optimizer (EqualWeight, RiskParity, Kelly, TopNRanking, PctEquity) |
| `indicator_list` | List all available indicator types with formulas |
| `indicator_describe` | Describe an indicator: formula, parameters, category |
| `signal_list` | List all available signal types |
| `signal_describe` | Describe a signal type: parameters and usage |
| `rule_list` | List all available rule types |
| `rule_describe` | Describe a rule type: parameters and usage |
| `portfolio_list` | List all available portfolio optimizer types |
| `portfolio_describe` | Describe a portfolio optimizer: parameters and usage |

### data (9 tools)

| Tool | Description |
|------|-------------|
| `data_load_symbols` | Download market data for given symbols |
| `data_list_symbols` | List locally available market data symbols |
| `data_inspect` | Inspect data summary (rows, date range, missing values) |
| `factor_download` | Download macro indicator from World Bank (gdp, cpi, etc.) |
| `factor_list` | List locally available factor files |
| `factor_inspect` | Inspect a factor file (year range, countries, samples) |
| `financial_download` | Download financial statement data (eps, roe, revenue, etc.) |
| `financial_list` | List locally available financial data files |
| `financial_inspect` | Inspect financial data (date range, indicators, samples) |

### universe (4 tools)

| Tool | Description |
|------|-------------|
| `universe_set` | Create a universe (static or filter-based) |
| `universe_list_indexes` | List available index-based universes (S&P 500, CSI 300, etc.) |
| `universe_inspect` | Inspect symbols: data availability, date range, latest price/volume |
| `universe_history` | Get universe snapshots over a date range |

### engine (4 tools)

| Tool | Description |
|------|-------------|
| `engine_run` | Run strategy backtest with fee/slippage models |
| `run_list` | List all runs with key metrics |
| `engine_results` | Get performance metrics and objectives check |
| `engine_trade_list` | Get trade list from a backtest run |

### factor_eval (2 tools)

| Tool | Description |
|------|-------------|
| `factor_evaluate` | Cross-sectional evaluation: IC, ICIR, RankIC, decay, turnover |
| `factor_evaluate_ts` | Time-series evaluation: hit rate, decay curve, P/L ratio, tearsheet PNG |

### optimize (7 tools)

| Tool | Description |
|------|-------------|
| `paramset_create` | Create parameter search space |
| `paramset_list` | List all parameter sets |
| `paramset_inspect` | Inspect distributions, constraints, sample combinations |
| `grid_search` | Exhaustive grid search, return ranked results |
| `walk_forward` | Walk-forward analysis with rolling/anchored windows |
| `cross_validate` | Time-series cross-validation (expanding/sliding) |
| `overfit_analysis` | Compare in-sample GridSearch vs walk-forward OOS |

### observe (11 tools)

| Tool | Description |
|------|-------------|
| `observe_trace` | View execution trace (TraceSpans) for a run |
| `observe_audit_log` | Create/view audit record (config snapshot + layered hashes) |
| `observe_audit_compare` | Compare two audit records to find divergence layer |
| `observe_monitor_create` | Create StrategyMonitor for health checking |
| `observe_monitor_summary` | Get health summary and bad periods |
| `observe_detect_market_state` | Detect volatility regimes (high/normal/low) |
| `observe_performance_by_state` | Performance grouped by market state |
| `observe_experiment_create` | Create experiment log for iteration tracking |
| `observe_experiment_add` | Add experiment record manually |
| `observe_experiment_add_from_strategy` | Auto-extract experiment from strategy + run |
| `observe_experiment_list` | List all experiments as formatted table |

### live (9 tools)

| Tool | Description |
|------|-------------|
| `live_connect` | Connect to Alpaca (paper or live) |
| `live_account` | Get account info: equity, buying power, cash |
| `live_positions` | Get current portfolio positions |
| `live_bars` | Fetch historical OHLCV bars from Alpaca |
| `live_generate_orders` | Generate trade plan from target weights |
| `live_submit_order` | Submit order (market, limit, stop, trailing_stop) |
| `live_order_status` | Get order status by ID |
| `live_open_orders` | List open orders, optionally by symbol |
| `live_cancel_order` | Cancel open orders for a symbol |

### chart (1 tool)

| Tool | Description |
|------|-------------|
| `chart_indicator` | Render candlestick chart with indicator overlays, return PNG path |

### MCP-only tools (not part of oxq SDK)

| Tool | Description |
|------|-------------|
| `get_current_date` | Get today's date (call first when user mentions relative dates) |
| `skill_list` | List all available agent skills |
| `skill_load` | Load a skill's full instructions by name |

---

## Skills Directory

Skills live in `agent/skills/` within the repo.
Installed skills for this agent live in `~/.openclaw/workspace/skills/`.

Before using any skill, read its content to understand:
- what tools it calls
- what inputs it expects
- what outputs it produces

---

## Python Environment

- Preferred runner: `uv run python` or `uv run pytest`
- Linter/formatter: `ruff` (rules: E, F, I, N, W, UP)
- Type checker: `mypy` (strict mode)
- Test runner: `uv run pytest` (unit tests by default; e2e and integration tests excluded unless flagged)

Install variants:
```bash
pip install open-xquant                          # core only
pip install open-xquant[yfinance,akshare]        # with data sources
pip install open-xquant[mcp]                     # with MCP server
pip install open-xquant[live]                    # Alpaca live trading
pip install open-xquant[chart]                   # chart visualization
pip install open-xquant[agent]                   # full agent stack
```

---

## Examples & Tutorials

Located in `examples/` within the repo:

| Path | Content |
|------|---------|
| **Tutorials** | |
| `tutorials/data_module.ipynb` | Data download & reading |
| `tutorials/universe_module.ipynb` | Universe construction |
| `tutorials/engine_module.ipynb` | Indicator → Signal → Rule pipeline |
| `tutorials/signal_comparison.ipynb` | Signal type comparison |
| `tutorials/rules_module.ipynb` | Rule configuration & evaluation |
| `tutorials/optimize_module.ipynb` | Parameter optimization |
| `tutorials/observe_module.ipynb` | Tracing, audit, monitoring |
| `tutorials/financial_factors.ipynb` | Financial statement factors |
| `tutorials/portfolio_snapshots.ipynb` | Portfolio state inspection |
| `tutorials/rotation_strategy.ipynb` | Rotation strategy walkthrough |
| `tutorials/alpaca_paper_trading.ipynb` | Alpaca paper trading |
| **Strategies** | |
| `strategies/sma_crossover.py` | SMA crossover backtest |
| `strategies/momentum_rotation.py` | Momentum rotation strategy |
| `strategies/ma_crossover.py` | Moving average crossover |
| `strategies/mean_reversion.py` | Mean reversion strategy |
| `strategies/global_rotation_etf.py` | Global ETF rotation |
| `strategies/signal_comparison.py` | Signal comparison script |
| `strategies/multi_strategy.py` | Multi-strategy orchestration |
| **Apps** | |
| `app/agent_demo.py` | MCP-based agent demo (Streamlit) |
| `app/live_trading_demo.py` | Live trading demo |
| **Factor Evaluation** | |
| `factor_eval_demo.py` | Factor evaluation walkthrough |
| `factor_eval_multi_period.py` | Multi-period factor decay analysis |

---

## Session State

Tool calls share mutable state through `oxq.tools.session`. State is persisted
to a temp file so it survives MCP server restarts. This holds:
- Created strategies
- Run results
- Parameter sets
- Experiment logs
- Live broker connections

---

## Research Output Conventions

- Research scripts: `~/.openclaw/workspace/research/`
- Backtest results: log parameters to memory before running
- Factor evaluation: always compute IC + quantile returns + turnover together
- Reproducibility check: re-run with identical inputs before recording any result

---

## Framework Feedback Log

Running log of friction points and improvement suggestions:
`~/.openclaw/workspace/memory/framework-feedback.md`

Format per entry:
```
## [YYYY-MM-DD] <topic>
**Friction**: what was hard or missing
**Suggestion**: what would fix it
**Priority**: P0 / P1 / P2
```

---

## This Server

<!-- Replace with your actual deployment details -->
- Host: {{hostname}}
- OS: {{os_version}}
- OpenClaw: {{openclaw_version}}
- Gateway: {{gateway_url}}
- Channels: {{connected_channels}}
