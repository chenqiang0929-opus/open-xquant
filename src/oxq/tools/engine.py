"""Engine tools — run strategies and inspect results."""

from __future__ import annotations

import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from oxq.core.engine import Engine
from oxq.data.loaders import resolve_data_dir
from oxq.data.market import LocalMarketDataProvider
from oxq.tools import session
from oxq.tools.registry import registry
from oxq.trade.fees import PercentageFee
from oxq.trade.sim_broker import SimBroker
from oxq.trade.slippage import PercentageSlippage
from oxq.universe.static import StaticUniverse


@registry.tool(
    name="engine_run",
    description="Run a strategy backtest and return summary results. Supports fee and slippage models.",
)
def engine_run(
    strategy: str,
    symbols: list[str],
    start: str,
    end: str,
    initial_cash: float = 100_000.0,
    run_through: Literal["indicator", "signal"] | None = None,
    data_dir: str | None = None,
    fee_rate: float | None = None,
    fee_min: float | None = None,
    slippage_rate: float | None = None,
) -> dict[str, Any]:
    """Run a strategy through the engine and store the result."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    # Set universe from symbols param
    strat.universe = StaticUniverse(tuple(symbols))

    path = resolve_data_dir(Path(data_dir) if data_dir else None)
    market = LocalMarketDataProvider(path)

    # Build fee and slippage models from params
    fee_model = None
    if fee_rate is not None:
        fee_model = PercentageFee(
            rate=Decimal(str(fee_rate)),
            min_fee=Decimal(str(fee_min)) if fee_min is not None else Decimal("0"),
        )

    slippage_model = None
    if slippage_rate is not None:
        slippage_model = PercentageSlippage(rate=Decimal(str(slippage_rate)))

    broker = SimBroker(fee_model=fee_model, slippage_model=slippage_model)

    try:
        result = Engine().run(
            strat,
            market=market,
            broker=broker,
            start=start,
            end=end,
            initial_cash=initial_cash,
            run_through=run_through,
        )
    except Exception as e:
        return {"error": str(e)}

    run_id = f"{strategy}_{int(time.time())}"
    session._run_results[run_id] = result
    session._save()

    # Get last market prices from mktdata for position valuation
    last_prices: dict[str, float] = {}
    for sym in result.mktdata:
        df = result.mktdata[sym]
        if not df.empty and "close" in df.columns:
            last_prices[sym] = float(df["close"].iloc[-1])

    positions = {
        sym: {
            "shares": int(pos.shares),
            "avg_cost": float(pos.avg_cost),
            "market_price": last_prices.get(sym, float(pos.avg_cost)),
        }
        for sym, pos in result.portfolio.positions.items()
    }

    total_value = float(
        result.equity_curve[-1][1] if result.equity_curve else result.portfolio.cash,
    )

    return {
        "run_id": run_id,
        "portfolio": {
            "cash": float(result.portfolio.cash),
            "positions": positions,
            "total_value": total_value,
        },
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
        "total_return": float(result.total_return()),
        "annualized_return": float(result.annualized_return()),
        "annualized_volatility": float(result.annualized_volatility()),
        "max_drawdown": float(result.max_drawdown()),
        "sharpe_ratio": float(result.sharpe_ratio()),
        "calmar_ratio": float(result.calmar_ratio()),
        "sortino_ratio": float(result.sortino_ratio()),
    }

    # Check objectives from the strategy that produced this result
    objectives_check: list[dict[str, Any]] = []
    # Find the strategy name from the run_id prefix
    strategy_name = run_id.rsplit("_", 1)[0]
    strat = session._strategies.get(strategy_name)

    # Metrics where the value is negative and "max" means abs(actual) <= abs(max)
    _abs_compare_metrics = {"max_drawdown"}

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
                if metric_name in _abs_compare_metrics:
                    # For drawdown: -0.08 is better than -0.15,
                    # so abs(actual) <= abs(max) means pass
                    passed = passed and abs(actual) <= abs(bounds["max"])
                else:
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
            "shares": int(fill.order.shares),
            "order_type": fill.order.order_type,
            "price": float(fill.filled_price),
            "fee": float(fill.fee),
            "date": fill.filled_at,
        }
        for fill in result.trades
    ]

    return {
        "run_id": run_id,
        "total_trades": len(trades),
        "trades": trades,
    }
