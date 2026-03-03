"""Engine result and performance analytics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from oxq.core.types import Fill, Portfolio


@dataclass
class RunResult:
    """Container for engine output with basic performance metrics."""

    portfolio: Portfolio
    trades: list[Fill]
    equity_curve: list[tuple[object, float]]  # [(date, value), ...]
    mktdata: dict[str, pd.DataFrame] = field(repr=False)

    # -- Metrics --------------------------------------------------------------

    def total_return(self) -> float:
        """Total return as a fraction (e.g. 0.15 = 15%)."""
        if len(self.equity_curve) < 2:
            return 0.0
        first = self.equity_curve[0][1]
        last = self.equity_curve[-1][1]
        if first == 0.0:
            return 0.0
        return (last - first) / first

    def sharpe_ratio(self, trading_days: int = 252) -> float:
        """Annualized Sharpe ratio (assumes risk-free rate = 0)."""
        if len(self.equity_curve) < 2:
            return 0.0
        values = np.array([v for _, v in self.equity_curve], dtype=float)
        returns = np.diff(values) / values[:-1]
        if len(returns) == 0 or np.std(returns) == 0.0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(trading_days))

    def max_drawdown(self) -> float:
        """Maximum drawdown as a negative fraction (e.g. -0.10 = -10%)."""
        if len(self.equity_curve) < 2:
            return 0.0
        values = np.array([v for _, v in self.equity_curve], dtype=float)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return float(np.min(drawdown))

    def annualized_return(self, trading_days: int = 252) -> float:
        """Annualized return based on mean daily log return x T."""
        if len(self.equity_curve) < 2:
            return 0.0
        values = np.array([v for _, v in self.equity_curve], dtype=float)
        log_returns = np.diff(np.log(values))
        if len(log_returns) == 0:
            return 0.0
        return float(np.mean(log_returns) * trading_days)

    def annualized_volatility(self, trading_days: int = 252) -> float:
        """Annualized volatility: sigma_daily x sqrt(T)."""
        if len(self.equity_curve) < 2:
            return 0.0
        values = np.array([v for _, v in self.equity_curve], dtype=float)
        log_returns = np.diff(np.log(values))
        if len(log_returns) == 0:
            return 0.0
        daily_vol = float(np.std(log_returns, ddof=1))
        return daily_vol * np.sqrt(trading_days)

    def calmar_ratio(self, trading_days: int = 252) -> float:
        """Calmar ratio: annualized_return / |MDD|."""
        ann_ret = self.annualized_return(trading_days)
        mdd = self.max_drawdown()
        if mdd == 0.0:
            return 0.0
        return float(ann_ret / abs(mdd))

    def sortino_ratio(
        self, risk_free: float = 0.0, trading_days: int = 252,
    ) -> float:
        """Sortino ratio: (annualized return - r_f) / downside deviation."""
        if len(self.equity_curve) < 2:
            return 0.0
        values = np.array([v for _, v in self.equity_curve], dtype=float)
        log_returns = np.diff(np.log(values))
        if len(log_returns) == 0:
            return 0.0
        downside = log_returns[log_returns < 0]
        if len(downside) == 0:
            return 0.0
        downside_dev = float(np.sqrt(np.mean(downside**2)) * np.sqrt(trading_days))
        ann_ret = float(np.mean(log_returns) * trading_days)
        return float((ann_ret - risk_free) / downside_dev)
