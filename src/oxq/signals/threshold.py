"""Threshold signal — fires when a column crosses above or below a value."""

from __future__ import annotations

import operator

import pandas as pd


class Threshold:
    """True when ``column`` satisfies ``relationship`` relative to ``threshold``."""

    name = "Threshold"

    def compute(
        self,
        mktdata: pd.DataFrame,
        column: str = "",
        threshold: float = 0.0,
        relationship: str = "gt",
    ) -> pd.Series:
        ops = {
            "gt": operator.gt,
            "lt": operator.lt,
            "gte": operator.ge,
            "lte": operator.le,
        }
        op = ops[relationship]
        return op(mktdata[column], threshold)
