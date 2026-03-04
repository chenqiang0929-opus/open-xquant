"""RiskParity signal — inverse-volatility weighted allocation."""

from __future__ import annotations

import pandas as pd


class RiskParity:
    """Assign weights inversely proportional to rolling volatility.

    For each bar:
    1. Read vol column from each symbol (pre-computed by indicator)
    2. Compute inv_vol = 1 / vol (skip NaN and vol <= 0)
    3. Normalize: weight = inv_vol / sum(inv_vol)
    4. Cap at max_weight (excess goes to cash, not redistributed)
    """

    name = "RiskParity"

    def compute(
        self,
        mktdata: dict[str, pd.DataFrame],
        vol: str = "",
        max_weight: float = 1.0,
    ) -> dict[str, pd.Series]:
        """Return inverse-vol target_weight Series for every symbol."""
        symbols = list(mktdata.keys())
        if not symbols:
            return {}

        all_dates = mktdata[symbols[0]].index
        for s in symbols[1:]:
            all_dates = all_dates.union(mktdata[s].index)

        result: dict[str, pd.Series] = {
            s: pd.Series(0.0, index=mktdata[s].index, dtype=float)
            for s in symbols
        }

        for date in all_dates:
            inv_vols: dict[str, float] = {}
            for s in symbols:
                if date not in mktdata[s].index:
                    continue
                v = mktdata[s].at[date, vol]
                if pd.isna(v) or v <= 0:
                    continue
                inv_vols[s] = 1.0 / float(v)

            if not inv_vols:
                continue

            total = sum(inv_vols.values())
            for s, iv in inv_vols.items():
                w = min(iv / total, max_weight)
                result[s].at[date] = w

        return result
