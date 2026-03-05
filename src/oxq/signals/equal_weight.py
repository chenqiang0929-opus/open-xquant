"""EqualWeight signal — equal 1/N allocation across valid symbols."""

from __future__ import annotations

import pandas as pd


class EqualWeight:
    """Assign equal weight 1/N to every symbol with valid close data.

    N is always the total number of symbols in the universe, regardless
    of how many have data on a given date.  When a symbol is missing,
    its share goes to cash (not redistributed to others).

    For each bar:
    1. N = total symbols in universe (fixed)
    2. Each valid symbol gets weight = min(1/N, max_weight)
    3. Missing / NaN-close symbols get weight 0
    4. Excess weight goes to cash, not redistributed
    """

    name = "EqualWeight"

    def compute(
        self,
        mktdata: dict[str, pd.DataFrame],
        max_weight: float = 1.0,
    ) -> dict[str, pd.Series]:
        """Return equal target_weight Series for every symbol."""
        symbols = list(mktdata.keys())
        if not symbols:
            return {}

        # Each symbol keeps its own index; iterate the union of all dates
        all_dates = mktdata[symbols[0]].index
        for s in symbols[1:]:
            all_dates = all_dates.union(mktdata[s].index)

        result: dict[str, pd.Series] = {
            s: pd.Series(0.0, index=mktdata[s].index, dtype=float)
            for s in symbols
        }

        for date in all_dates:
            valid: list[str] = []
            for s in symbols:
                if date not in mktdata[s].index:
                    continue
                if pd.isna(mktdata[s].at[date, "close"]):
                    continue
                valid.append(s)

            if not valid:
                continue

            w = min(1.0 / len(symbols), max_weight)
            for s in valid:
                result[s].at[date] = w

        return result
