"""TopNRanking signal — cross-sectional ranking with weight normalization."""

from __future__ import annotations

import pandas as pd


class TopNRanking:
    """Rank symbols by score, select top N, normalize to target weights.

    For each bar:
    1. Read score column from each symbol
    2. Filter out NaN and (optionally) negative scores
    3. Rank descending, keep top N
    4. Normalize remaining scores to sum to 1.0
    5. Cap any single weight at max_weight (excess goes to cash)

    Output: per-symbol Series of target weights (0.0 for non-selected).
    """

    name = "TopNRanking"

    def compute(
        self,
        mktdata: dict[str, pd.DataFrame],
        score: str = "",
        n: int = 5,
        filter_negative: bool = True,
        max_weight: float = 1.0,
    ) -> dict[str, pd.Series]:
        """Return target_weight Series for every symbol."""
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
            scores: dict[str, float] = {}
            for s in symbols:
                if date not in mktdata[s].index:
                    continue
                val = mktdata[s].at[date, score]
                if pd.isna(val):
                    continue
                if filter_negative and val <= 0:
                    continue
                scores[s] = float(val)

            if not scores:
                continue

            # Rank descending, take top N
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top = ranked[:n]

            # Normalize to sum to 1.0
            total = sum(v for _, v in top)
            if total <= 0:
                continue

            # Cap at max_weight — excess goes to cash (not redistributed)
            for s, v in top:
                w = min(v / total, max_weight)
                result[s].at[date] = w

        return result
