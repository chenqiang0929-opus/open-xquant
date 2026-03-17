"""Comparison signal — fires when one column satisfies a relationship with another."""

from __future__ import annotations

import operator

import pandas as pd


class Comparison:
    """True when ``left`` column satisfies ``relationship`` relative to ``right`` column."""

    name = "Comparison"

    def compute(
        self,
        mktdata: dict[str, pd.DataFrame],
        left: str = "",
        right: str = "",
        relationship: str = "gt",
    ) -> dict[str, pd.Series]:
        ops = {
            "gt": operator.gt,
            "lt": operator.lt,
            "gte": operator.ge,
            "lte": operator.le,
            "eq": operator.eq,
            "ne": operator.ne,
        }
        op = ops[relationship]
        return {s: op(df[left], df[right]) for s, df in mktdata.items()}
