"""Portfolio optimizer implementations."""

from __future__ import annotations

import pandas as pd


class EqualWeightOptimizer:
    """Assigns equal weight to all symbols."""

    name: str = "EqualWeight"

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        if not signals:
            return {"CASH": 1.0}
        weight = 1.0 / len(signals)
        return {symbol: weight for symbol in signals}


class RiskParityOptimizer:
    """Weights inversely proportional to volatility."""

    name: str = "RiskParity"

    def __init__(self, volatility_col: str = "volatility") -> None:
        self.volatility_col = volatility_col

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        inv_vols: dict[str, float] = {}
        for symbol, df in indicators.items():
            vol = float(df[self.volatility_col].iloc[-1])
            if vol > 0:
                inv_vols[symbol] = 1.0 / vol

        if not inv_vols:
            return {"CASH": 1.0}

        total = sum(inv_vols.values())
        return {symbol: iv / total for symbol, iv in inv_vols.items()}


class KellyOptimizer:
    """Kelly criterion-based position sizing."""

    name: str = "Kelly"

    def __init__(
        self,
        win_rate_col: str = "win_rate",
        avg_win_col: str = "avg_win",
        avg_loss_col: str = "avg_loss",
        fraction: float = 1.0,
    ) -> None:
        self.win_rate_col = win_rate_col
        self.avg_win_col = avg_win_col
        self.avg_loss_col = avg_loss_col
        self.fraction = fraction

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        weights: dict[str, float] = {}

        for symbol, df in indicators.items():
            win_rate = float(df[self.win_rate_col].iloc[-1])
            avg_win = float(df[self.avg_win_col].iloc[-1])
            avg_loss = float(df[self.avg_loss_col].iloc[-1])

            if avg_loss <= 0:
                continue

            payoff_ratio = avg_win / avg_loss
            kelly_pct = win_rate - (1 - win_rate) / payoff_ratio
            kelly_pct = max(kelly_pct, 0.0) * self.fraction

            if kelly_pct > 0:
                weights[symbol] = kelly_pct

        if not weights:
            return {"CASH": 1.0}

        total = sum(weights.values())
        if total > 1.0:
            weights = {s: w / total for s, w in weights.items()}
        else:
            weights["CASH"] = 1.0 - total

        return weights
