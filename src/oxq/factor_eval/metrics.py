"""Pure functions for factor evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_obs: int = 3,
) -> dict[str, object]:
    """Compute Information Coefficient (mean per-period Pearson correlation).

    Parameters
    ----------
    factor : pd.DataFrame
        Factor values. index=date, columns=symbols.
    forward_returns : pd.DataFrame
        Forward returns aligned with factor. Same shape.
    min_obs : int
        Minimum observations per period. Periods with fewer are skipped.

    Returns
    -------
    dict with keys: mean, std, series (per-period IC values as list).
    """
    ic_values: list[float] = []
    for date in factor.index:
        f = factor.loc[date]
        r = forward_returns.loc[date]
        mask = f.notna() & r.notna()
        if mask.sum() < min_obs:
            continue
        corr, _ = stats.pearsonr(f[mask], r[mask])
        ic_values.append(float(corr))

    if not ic_values:
        return {"mean": float("nan"), "std": float("nan"), "series": []}

    return {
        "mean": float(np.mean(ic_values)),
        "std": float(np.std(ic_values, ddof=1)),
        "series": ic_values,
    }


def compute_rank_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_obs: int = 3,
) -> dict[str, object]:
    """Compute Rank IC (mean per-period Spearman rank correlation)."""
    ic_values: list[float] = []
    for date in factor.index:
        f = factor.loc[date]
        r = forward_returns.loc[date]
        mask = f.notna() & r.notna()
        if mask.sum() < min_obs:
            continue
        corr, _ = stats.spearmanr(f[mask], r[mask])
        ic_values.append(float(corr))

    if not ic_values:
        return {"mean": float("nan"), "std": float("nan"), "series": []}

    return {
        "mean": float(np.mean(ic_values)),
        "std": float(np.std(ic_values, ddof=1)),
        "series": ic_values,
    }


def compute_icir(ic_mean: float, ic_std: float) -> float:
    """Compute ICIR = IC mean / IC std."""
    if ic_std == 0.0 or np.isnan(ic_mean) or np.isnan(ic_std):
        return float("nan")
    return ic_mean / ic_std


def compute_decay(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: list[int],
    min_obs: int = 3,
) -> dict[str, object]:
    """Compute IC at multiple forward return horizons.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor values. index=date, columns=symbols.
    prices : pd.DataFrame
        Close prices. Same shape as factor.
    horizons : list[int]
        Forward return horizons in days.
    min_obs : int
        Minimum observations per period.

    Returns
    -------
    dict with keys: horizons (list[int]), ic_values (list[float]).
    """
    raise NotImplementedError


def compute_turnover(factor: pd.DataFrame) -> float:
    """Compute average factor rank turnover.

    Turnover = mean of per-period rank change ratios.
    """
    raise NotImplementedError
