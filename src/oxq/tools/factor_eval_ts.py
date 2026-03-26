"""Time-series factor evaluation tool — assess indicator predictive power per asset."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oxq.core.registry import list_indicators
from oxq.data.loaders import resolve_data_dir
from oxq.factor_eval.bundle import create_bundle
from oxq.factor_eval.preprocessing import (
    apply_t1_offset,
    mark_limit_days,
    mark_suspension_days,
)
from oxq.factor_eval.tearsheet import generate_tearsheet
from oxq.tools.registry import registry


def _compute_market_state_sma(prices: pd.DataFrame, period: int = 60) -> pd.Series:
    """Classify market state using SMA crossover on mean prices.

    - trend: price > SMA (uptrend)
    - ranging: price near SMA (within 2%)
    - crash: price < SMA by more than 5%
    """
    mean_price = prices.mean(axis=1)
    sma = mean_price.rolling(period).mean()
    ratio = mean_price / sma - 1

    state = pd.Series("ranging", index=prices.index)
    state[ratio > 0.02] = "trend"
    state[ratio < -0.05] = "crash"
    # NaN for warmup period
    state[sma.isna()] = np.nan
    return state


def _infer_asset_meta(symbols: list[str]) -> dict:
    """Infer exchange info from symbol codes."""
    meta = {}
    for sym in symbols:
        code = sym.split(".")[0] if "." in sym else sym
        if code.isdigit() and len(code) == 6:
            if code.startswith(("60", "68")):
                meta[sym] = {"exchange": "SH", "asset_type": "stock"}
            elif code.startswith(("00", "30")):
                meta[sym] = {"exchange": "SZ", "asset_type": "stock"}
            else:
                meta[sym] = {"exchange": "US", "asset_type": "stock"}
        else:
            meta[sym] = {"exchange": "US", "asset_type": "stock"}
    return meta


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, pd.Series):
        return _make_json_safe(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _make_json_safe(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _make_json_safe(obj.tolist())
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    return obj


@registry.tool(
    name="factor_evaluate_ts",
    description=(
        "Evaluate an indicator's time-series predictive power for one or more assets. "
        "Computes hit rate, decay curve, profit/loss ratio, cash period value, "
        "and optionally market state conditional analysis and multi-asset comparison. "
        "Returns a structured evaluation report with PNG chart paths."
    ),
)
def factor_evaluate_ts(
    indicator: str,
    params: dict[str, Any],
    symbols: list[str],
    start: str,
    end: str,
    data_dir: str | None = None,
    forward_periods: list[int] | None = None,
    signal_threshold: float = 0.0,
    t1_offset: bool = True,
    market_state_method: str | None = None,
    market_state_params: dict[str, Any] | None = None,
    exclude_limit_days: bool = False,
) -> dict[str, Any]:
    """Evaluate an indicator's time-series predictive power."""
    if forward_periods is None:
        forward_periods = [1, 3, 5, 10, 20]

    # -- Resolve indicator class ------------------------------------------------
    indicators = list_indicators()
    if indicator not in indicators:
        return {"error": f"Unknown indicator '{indicator}'. Available: {sorted(indicators)}"}

    ind_cls = indicators[indicator]
    ind_instance = ind_cls()

    # -- Load market data -------------------------------------------------------
    # Load full data (not sliced by end) so prices extend beyond factor dates
    # for forward return computation.
    data_path = resolve_data_dir(Path(data_dir) if data_dir else None)
    mktdata_full: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for sym in symbols:
        parquet = data_path / f"{sym}.parquet"
        if not parquet.exists():
            errors.append(f"No data for '{sym}'")
            continue
        df = pd.read_parquet(parquet)
        mktdata_full[sym] = df.loc[start:]

    if not mktdata_full:
        return {"error": f"No data found. {'; '.join(errors)}"}

    # -- Compute indicator per symbol -------------------------------------------
    # Compute indicator on the user-specified date range only.
    factor_parts: list[pd.Series] = []
    for sym, df_full in mktdata_full.items():
        df_range = df_full.loc[:end]
        vals = ind_instance.compute(df_range, **params)
        # Tag with (date, asset) MultiIndex
        mi = pd.MultiIndex.from_arrays(
            [vals.index, [sym] * len(vals)],
            names=["date", "asset"],
        )
        factor_parts.append(pd.Series(vals.values, index=mi, dtype=float))

    factor_values = pd.concat(factor_parts).sort_index()

    # -- Build prices DataFrame -------------------------------------------------
    # Use full data (beyond end) so forward returns can be computed.
    prices_df = pd.DataFrame({sym: df["close"] for sym, df in mktdata_full.items()})

    # -- Warnings ---------------------------------------------------------------
    warnings: list[str] = []
    if errors:
        warnings.append(f"Skipped symbols: {'; '.join(errors)}")

    # -- Asset meta -------------------------------------------------------------
    asset_meta = _infer_asset_meta(list(mktdata_full.keys()))

    # -- Create FactorBundle ----------------------------------------------------
    try:
        bundle = create_bundle(
            factor_values=factor_values,
            prices=prices_df,
            forward_periods=forward_periods,
            asset_meta=asset_meta,
        )
    except ValueError as exc:
        return {"error": str(exc)}

    # -- Preprocessing ----------------------------------------------------------
    bundle.limit_days = mark_limit_days(bundle.prices, asset_meta=asset_meta)
    bundle.suspension_days = mark_suspension_days(bundle.prices)

    if t1_offset:
        bundle = apply_t1_offset(bundle)

    # -- Market state -----------------------------------------------------------
    if market_state_method == "sma":
        sma_period = (market_state_params or {}).get("period", 60)
        market_state = _compute_market_state_sma(bundle.prices, period=sma_period)
        bundle.market_state = market_state.dropna()

    # -- Generate tearsheet -----------------------------------------------------
    tearsheet = generate_tearsheet(
        bundle=bundle,
        forward_periods=forward_periods,
        signal_threshold=signal_threshold,
        exclude_limit_days=exclude_limit_days,
        start_date=start,
        end_date=end,
    )

    summary = tearsheet["summary"]
    charts = tearsheet["charts"]

    # -- Extract structured metrics ---------------------------------------------
    hr = summary["hit_rate"]
    dc = summary["decay_curve"]
    pl = summary["profit_loss"]
    cp = summary["cash_period"]

    result = {
        "indicator": indicator,
        "params": params,
        "symbols": list(mktdata_full.keys()),
        "symbols_count": len(mktdata_full),
        "date_range": {"start": start, "end": end},
        "warnings": warnings,
        "config": {
            "t1_offset": t1_offset,
            "signal_threshold": signal_threshold,
            "forward_periods": forward_periods,
            "market_state_method": market_state_method,
        },
        "metrics": {
            "alignment_report": summary["alignment_report"],
            "lookahead_bias": summary["lookahead_bias"],
            "hit_rate": {
                "total": hr["total_hit_rate"],
                "long": hr["long_hit_rate"],
                "short": hr["short_hit_rate"],
                "sample_count": hr["sample_count"],
            },
            "decay_curve": {
                "correlations": dc["correlations"],
                "half_life": dc["half_life"],
                "inflection_point": dc["inflection_point"],
            },
            "profit_loss": {
                "avg_win": pl["avg_win"],
                "avg_loss": pl["avg_loss"],
                "ratio": pl["ratio"],
            },
            "cash_period": {
                "holding_avg_return": cp["holding_avg_return"],
                "cash_avg_return": cp["cash_avg_return"],
                "return_spread": cp["return_spread"],
            },
            "conditional": summary["conditional"],
            "comparison": summary["comparison"],
        },
        "charts": charts,
    }

    return _make_json_safe(result)
