"""Observe tools — strategy monitoring, market state detection, experiment tracking."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from oxq.observe.detector import MarketStateDetector
from oxq.observe.experiment import ExperimentLog
from oxq.observe.monitor import StrategyMonitor
from oxq.tools import session
from oxq.tools.registry import registry


@registry.tool(
    name="observe_monitor_create",
    description="Create a StrategyMonitor for a run result to check strategy health. "
    "Returns monitor_id and initial summary.",
)
def observe_monitor_create(
    run_id: str,
    benchmark: str | None = None,
    roll_window: int = 63,
    min_bad_days: int = 20,
    gap_days: int = 5,
) -> dict[str, Any]:
    """Create a StrategyMonitor from a run result."""
    result = session._run_results.get(run_id)
    if result is None:
        return {"error": f"Run '{run_id}' not found"}

    try:
        monitor = StrategyMonitor(
            result,
            benchmark=benchmark,
            roll_window=roll_window,
            min_bad_days=min_bad_days,
            gap_days=gap_days,
        )
    except Exception as e:
        return {"error": str(e)}

    monitor_id = f"mon_{run_id}_{int(time.time())}"
    session._monitors[monitor_id] = monitor
    session._save()

    return {
        "monitor_id": monitor_id,
        **monitor.summary(),
    }


@registry.tool(
    name="observe_monitor_summary",
    description="Get health summary and bad periods for a strategy monitor.",
)
def observe_monitor_summary(monitor_id: str) -> dict[str, Any]:
    """Get summary and bad periods from a monitor."""
    monitor = session._monitors.get(monitor_id)
    if monitor is None:
        return {"error": f"Monitor '{monitor_id}' not found"}

    bad_periods = [
        {
            "start": str(bp.start),
            "end": str(bp.end),
            "days": bp.days,
            "avg_sharpe": bp.avg_sharpe,
        }
        for bp in monitor.bad_periods
    ]

    return {
        "monitor_id": monitor_id,
        **monitor.summary(),
        "bad_periods": bad_periods,
    }


@registry.tool(
    name="observe_detect_market_state",
    description="Detect market volatility regimes (high/normal/low) from a run result's market data.",
)
def observe_detect_market_state(
    run_id: str,
    symbols: list[str] | None = None,
    vol_lookback: int = 20,
    high_vol_multiplier: float = 1.3,
    low_vol_multiplier: float = 0.7,
) -> dict[str, Any]:
    """Create a MarketStateDetector from a run result."""
    result = session._run_results.get(run_id)
    if result is None:
        return {"error": f"Run '{run_id}' not found"}

    try:
        detector = MarketStateDetector(
            result,
            symbols=tuple(symbols) if symbols else None,
            vol_lookback=vol_lookback,
            high_vol_multiplier=high_vol_multiplier,
            low_vol_multiplier=low_vol_multiplier,
        )
    except Exception as e:
        return {"error": str(e)}

    detector_id = f"det_{run_id}_{int(time.time())}"
    session._detectors[detector_id] = detector
    session._save()

    state_counts = detector.states.value_counts().to_dict()

    return {
        "detector_id": detector_id,
        "thresholds": {
            "vol_median": detector.vol_median,
            "high_vol_line": detector.high_vol_line,
            "low_vol_line": detector.low_vol_line,
        },
        "state_counts": {
            "high": int(state_counts.get("high", 0)),
            "normal": int(state_counts.get("normal", 0)),
            "low": int(state_counts.get("low", 0)),
        },
    }


@registry.tool(
    name="observe_performance_by_state",
    description="Evaluate strategy performance grouped by market state (high/normal/low volatility).",
)
def observe_performance_by_state(
    detector_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Get performance breakdown by market state."""
    detector = session._detectors.get(detector_id)
    if detector is None:
        return {"error": f"Detector '{detector_id}' not found"}

    result = session._run_results.get(run_id)
    if result is None:
        return {"error": f"Run '{run_id}' not found"}

    try:
        perf = detector.performance_by_state(result)
    except Exception as e:
        return {"error": str(e)}

    return {
        "detector_id": detector_id,
        "run_id": run_id,
        "performance": perf,
    }


@registry.tool(
    name="observe_experiment_create",
    description="Create an experiment log for tracking strategy iteration experiments.",
)
def observe_experiment_create(name: str = "") -> dict[str, Any]:
    """Create a new ExperimentLog."""
    log = ExperimentLog(name=name)
    log_id = f"explog_{int(time.time())}"
    session._experiment_logs[log_id] = log
    session._save()

    return {
        "log_id": log_id,
        "name": name,
    }


@registry.tool(
    name="observe_experiment_add",
    description="Add an experiment record manually to an experiment log.",
)
def observe_experiment_add(
    log_id: str,
    name: str,
    observation: str,
    hypothesis: str,
    criteria: dict[str, Any],
    result: dict[str, Any],
    conclusion: str,
    notes: str = "",
) -> dict[str, Any]:
    """Add a manual experiment record to a log."""
    log = session._experiment_logs.get(log_id)
    if log is None:
        return {"error": f"Experiment log '{log_id}' not found"}

    log.add(
        name=name,
        observation=observation,
        hypothesis=hypothesis,
        criteria=criteria,
        result=result,
        conclusion=conclusion,
        notes=notes,
    )
    session._save()

    return {
        "log_id": log_id,
        "experiment_count": len(log.experiments),
    }


@registry.tool(
    name="observe_experiment_add_from_strategy",
    description="Auto-extract experiment data from a strategy and run result, add to experiment log.",
)
def observe_experiment_add_from_strategy(
    log_id: str,
    strategy: str,
    run_id: str,
    observation: str,
    conclusion: str,
    notes: str = "",
) -> dict[str, Any]:
    """Add experiment from strategy + run result auto-extraction."""
    log = session._experiment_logs.get(log_id)
    if log is None:
        return {"error": f"Experiment log '{log_id}' not found"}

    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    run_result = session._run_results.get(run_id)
    if run_result is None:
        return {"error": f"Run '{run_id}' not found"}

    try:
        log.add_from_strategy(
            strategy=strat,
            result=run_result,
            observation=observation,
            conclusion=conclusion,
            notes=notes,
        )
    except Exception as e:
        return {"error": str(e)}

    session._save()

    # Return the last added experiment data
    last_exp = log.experiments[-1]
    return {
        "log_id": log_id,
        "experiment_count": len(log.experiments),
        "experiment": asdict(last_exp),
    }


@registry.tool(
    name="observe_experiment_list",
    description="List all experiments in a log as a formatted table.",
)
def observe_experiment_list(log_id: str) -> dict[str, Any]:
    """List experiments in a log."""
    log = session._experiment_logs.get(log_id)
    if log is None:
        return {"error": f"Experiment log '{log_id}' not found"}

    experiments = [asdict(e) for e in log.experiments]

    # Try to generate markdown table
    try:
        markdown = log.to_markdown()
    except ImportError:
        markdown = ""

    return {
        "log_id": log_id,
        "name": log.name,
        "experiment_count": len(experiments),
        "experiments": experiments,
        "markdown_table": markdown,
    }
