"""Plan 027 (xquant-studio): the consumers in oxq/tools/strategy.py must
read from the live registry on every call, not from a module-level
snapshot taken at import time. A snapshot would hide any indicator,
signal, optimizer, or rule registered after import — which is exactly
what xquant-studio's component_create does at runtime.

These tests pin the invariant for indicators (the path where the bug
was first observed in xquant-studio session 2d5e93eb). The patch in
strategy.py replaces every snapshot reference with a live `list_*()`
call uniformly across all four slots, so the indicator coverage here
is sufficient evidence that the pattern is right.

Ref: xquant-studio/docs/plans/2026-04-07-027-impl-component-create-registration-visibility.md
"""

from __future__ import annotations

import pandas as pd
import pytest

from oxq.core.registry import register_indicator
from oxq.tools import session
from oxq.tools.strategy import (
    indicator_describe,
    indicator_list,
    strategy_add_signal,
    strategy_create,
)


@pytest.fixture(autouse=True)
def _reset_session():
    session.clear()


class _MockBetaInd:
    """A minimal Indicator-shape class registered AFTER strategy.py import."""

    name = "Plan027MockBeta"
    formula = "beta"

    def compute(self, mktdata, period: int = 60, **p):  # noqa: D401
        return pd.Series([0.0] * len(mktdata))


def test_indicator_list_reflects_post_import_registration():
    """A post-import register_indicator() must be visible to indicator_list()."""
    register_indicator(_MockBetaInd)
    names = {ind["name"] for ind in indicator_list()["indicators"]}
    assert "Plan027MockBeta" in names, sorted(names)


def test_indicator_describe_reflects_post_import_registration():
    """indicator_describe() must resolve a post-import indicator."""
    register_indicator(_MockBetaInd)
    out = indicator_describe(type="Plan027MockBeta")
    assert "error" not in out, out
    assert out["name"] == "Plan027MockBeta"
    assert "period" in out["params"]


def test_strategy_add_signal_resolves_post_import_indicator():
    """The composition path: _build_required_indicators must accept a
    post-import indicator type. This is the exact code path that produced
    "Unknown indicator type 'RollingBeta'" in xquant-studio session 2d5e93eb,
    row 392."""
    register_indicator(_MockBetaInd)

    create_result = strategy_create(
        name="t027",
        hypothesis="post-import indicator must resolve",
        objectives={"sharpe": {"min": 0.0}},
    )
    assert "error" not in create_result, create_result

    result = strategy_add_signal(
        strategy="t027",
        name="probe",
        type="Threshold",
        params={"column": "beta_col", "threshold": 0, "direction": "above"},
        indicators={
            "beta_col": {"type": "Plan027MockBeta", "params": {"period": 60}},
        },
    )
    assert "error" not in result, result
    assert result["signal"] == "probe"
