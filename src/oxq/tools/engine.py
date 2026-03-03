"""Engine tools — run strategies and inspect results."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

from oxq.core.engine import Engine
from oxq.data.loaders import resolve_data_dir
from oxq.data.market import LocalMarketDataProvider
from oxq.tools import session
from oxq.tools.registry import registry
from oxq.trade.sim_broker import SimBroker
from oxq.universe.static import StaticUniverse


@registry.tool(
    name="engine_run",
    description="Run a strategy backtest and return summary results",
)
def engine_run(
    strategy: str,
    symbols: list[str],
    start: str,
    end: str,
    initial_cash: float = 100_000.0,
    run_through: Literal["indicator", "signal"] | None = None,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Run a strategy through the engine and store the result."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    # Set universe from symbols param
    strat.universe = StaticUniverse(tuple(symbols))

    path = resolve_data_dir(Path(data_dir) if data_dir else None)
    market = LocalMarketDataProvider(path)
    broker = SimBroker()

    try:
        result = Engine().run(
            strat,
            market=market,
            router=broker,
            receiver=broker,
            start=start,
            end=end,
            initial_cash=initial_cash,
            run_through=run_through,
        )
    except Exception as e:
        return {"error": str(e)}

    run_id = f"{strategy}_{int(time.time())}"
    session._run_results[run_id] = result

    positions = {
        sym: {"shares": pos.shares, "avg_cost": pos.avg_cost}
        for sym, pos in result.portfolio.positions.items()
    }

    return {
        "run_id": run_id,
        "portfolio": {"cash": result.portfolio.cash, "positions": positions},
        "total_trades": len(result.trades),
        "equity_curve_length": len(result.equity_curve),
    }


@registry.tool(
    name="engine_results",
    description="Get performance metrics and objectives check for a run",
)
def engine_results(run_id: str) -> dict[str, Any]:
    """Compute metrics and check against strategy objectives."""
    result = session._run_results.get(run_id)
    if result is None:
        return {"error": f"Run '{run_id}' not found"}

    metrics = {
        "total_return": result.total_return(),
        "sharpe_ratio": result.sharpe_ratio(),
        "max_drawdown": result.max_drawdown(),
    }

    # Check objectives from the strategy that produced this result
    objectives_check: list[dict[str, Any]] = []
    # Find the strategy name from the run_id prefix
    strategy_name = run_id.rsplit("_", 1)[0]
    strat = session._strategies.get(strategy_name)

    if strat and strat.objectives:
        for metric_name, bounds in strat.objectives.items():
            actual = metrics.get(metric_name)
            if actual is None:
                continue
            check: dict[str, Any] = {"metric": metric_name, "actual": actual}
            passed = True
            if "min" in bounds:
                check["min"] = bounds["min"]
                passed = passed and actual >= bounds["min"]
            if "max" in bounds:
                check["max"] = bounds["max"]
                passed = passed and actual <= bounds["max"]
            if "target" in bounds:
                check["target"] = bounds["target"]
            check["pass"] = passed
            objectives_check.append(check)

    return {
        "run_id": run_id,
        "metrics": metrics,
        "objectives_check": objectives_check,
    }


@registry.tool(
    name="engine_trade_list",
    description="Get the list of trades from a backtest run",
)
def engine_trade_list(run_id: str) -> dict[str, Any]:
    """Return all trades from a run."""
    result = session._run_results.get(run_id)
    if result is None:
        return {"error": f"Run '{run_id}' not found"}

    trades = [
        {
            "symbol": fill.order.symbol,
            "side": fill.order.side,
            "shares": fill.order.shares,
            "price": fill.filled_price,
            "date": fill.filled_at,
        }
        for fill in result.trades
    ]

    return {
        "run_id": run_id,
        "total_trades": len(trades),
        "trades": trades,
    }
