"""Rolling Volatility indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RollingVolatility:
    """N-day rolling volatility of log returns.

    sigma_{N,t} = sqrt(1/(N-1) * sum(r_{t-i} - r_bar_N)^2)
    """

    name = "RollingVolatility"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", period: int = 20,
    ) -> pd.Series:
        """Return rolling std of log returns (uses ddof=1)."""
        log_returns = np.log(mktdata[column]).diff()
        return log_returns.rolling(period).std(ddof=1)
