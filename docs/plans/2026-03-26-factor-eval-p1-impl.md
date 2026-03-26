# Factor Eval P1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add P1 analysis capabilities: profit/loss ratio, cash period value, market state conditional analysis, multi-asset comparison, and extend tearsheet to include all P1 content.

**Architecture:** Same as P0 — flat files under `src/oxq/factor_eval/`, pure functions, dict output. Tearsheet extended (not replaced) with P1 sections. Multi-asset comparison uses single FactorBundle with MultiIndex, grouping by asset internally.

**Tech Stack:** pandas, numpy, scipy.stats, matplotlib

---

### Task 1: Profit/Loss Ratio (`profit_loss.py`)

**Files:**
- Create: `src/oxq/factor_eval/profit_loss.py`
- Create: `tests/factor_eval/test_profit_loss.py`

**Step 1: Write failing tests**

```python
# tests/factor_eval/test_profit_loss.py
"""Tests for profit/loss ratio computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oxq.factor_eval.profit_loss import compute_profit_loss_ratio


class TestComputeProfitLossRatio:
    """Test profit/loss ratio with hand-calculated values."""

    @pytest.fixture()
    def data(self) -> dict:
        """10-point factor + return series.

        factor:  [0.5, -0.3, 0.8, -0.1, 0.6, -0.4, 0.2, 0.9, -0.5, 0.3]
        returns: [0.02, -0.01, 0.03, 0.01, -0.02, -0.03, 0.01, 0.04, -0.02, 0.01]

        Signal threshold = 0 (default).
        Long signals (factor > 0): indices 0,2,4,6,7,9
          returns: [0.02, 0.03, -0.02, 0.01, 0.04, 0.01]
          wins (>0): [0.02, 0.03, 0.01, 0.04, 0.01] → mean = 0.022
          losses (<0): [-0.02] → mean abs = 0.02
          P/L ratio = 0.022 / 0.02 = 1.1
        """
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        factor_values = pd.Series(
            [0.5, -0.3, 0.8, -0.1, 0.6, -0.4, 0.2, 0.9, -0.5, 0.3],
            index=pd.MultiIndex.from_arrays(
                [dates, ["a"] * 10], names=["date", "asset"],
            ),
        )
        forward_returns = pd.DataFrame(
            {"a": [0.02, -0.01, 0.03, 0.01, -0.02, -0.03, 0.01, 0.04, -0.02, 0.01]},
            index=dates,
        )
        return {"factor_values": factor_values, "forward_returns": forward_returns}

    def test_avg_win(self, data: dict) -> None:
        result = compute_profit_loss_ratio(**data)
        assert result["avg_win"] == pytest.approx(0.022)

    def test_avg_loss(self, data: dict) -> None:
        result = compute_profit_loss_ratio(**data)
        assert result["avg_loss"] == pytest.approx(0.02)

    def test_profit_loss_ratio(self, data: dict) -> None:
        result = compute_profit_loss_ratio(**data)
        assert result["ratio"] == pytest.approx(1.1)

    def test_return_distribution(self, data: dict) -> None:
        """Should return all long-signal returns for histogram data."""
        result = compute_profit_loss_ratio(**data)
        assert len(result["return_distribution"]) == 6  # 6 long signals

    def test_rolling_ratio_length(self, data: dict) -> None:
        result = compute_profit_loss_ratio(**data, rolling_window=5)
        assert len(result["rolling_ratio"]) == 10

    def test_warning_on_small_sample(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        fv = pd.Series(
            [0.1, 0.2, 0.3],
            index=pd.MultiIndex.from_arrays(
                [dates, ["a"] * 3], names=["date", "asset"],
            ),
        )
        fr = pd.DataFrame({"a": [0.01, 0.02, 0.03]}, index=dates)
        result = compute_profit_loss_ratio(fv, fr)
        assert result["warning"] is not None

    def test_no_losses_returns_inf(self, data: dict) -> None:
        """If all returns positive, ratio should be inf."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        fv = pd.Series(
            [0.5, 0.6, 0.7],
            index=pd.MultiIndex.from_arrays(
                [dates, ["a"] * 3], names=["date", "asset"],
            ),
        )
        fr = pd.DataFrame({"a": [0.01, 0.02, 0.03]}, index=dates)
        result = compute_profit_loss_ratio(fv, fr)
        assert result["ratio"] == float("inf")

    def test_date_range_filter(self, data: dict) -> None:
        full = compute_profit_loss_ratio(**data)
        sliced = compute_profit_loss_ratio(
            **data, start_date="2024-01-08", end_date="2024-01-12",
        )
        assert sliced["sample_count"] < full["sample_count"]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/factor_eval/test_profit_loss.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/oxq/factor_eval/profit_loss.py
"""Profit/loss ratio computation for time-series factor evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_SAMPLE_WARN = 60


def compute_profit_loss_ratio(
    factor_values: pd.Series,
    forward_returns: pd.DataFrame,
    signal_threshold: float = 0.0,
    rolling_window: int = 60,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Compute profit/loss ratio for long signals.

    Parameters
    ----------
    factor_values
        Series with MultiIndex (date, asset).
    forward_returns
        DataFrame with index=date, columns=asset.
    signal_threshold
        Factor value threshold for long signals. Default 0.
    rolling_window
        Window for rolling P/L ratio.
    start_date, end_date
        Optional date range filter.

    Returns
    -------
    dict with avg_win, avg_loss, ratio, rolling_ratio (Series),
    return_distribution (list), sample_count, warning.
    """
    dates = factor_values.index.get_level_values("date")
    assets = factor_values.index.get_level_values("asset")

    factors = []
    returns = []
    ret_dates = []

    for date, asset, fval in zip(dates, assets, factor_values.values):
        if start_date and str(date.date()) < start_date:
            continue
        if end_date and str(date.date()) > end_date:
            continue
        if asset not in forward_returns.columns or date not in forward_returns.index:
            continue
        ret = forward_returns.loc[date, asset]
        if np.isnan(fval) or np.isnan(ret):
            continue
        factors.append(fval)
        returns.append(ret)
        ret_dates.append(date)

    factors_arr = np.array(factors)
    returns_arr = np.array(returns)
    n = len(factors_arr)

    warning = None
    if n < _MIN_SAMPLE_WARN:
        warning = f"Sample size {n} < {_MIN_SAMPLE_WARN}, results for reference only."

    # Filter to long signals only
    long_mask = factors_arr > signal_threshold
    long_returns = returns_arr[long_mask]
    long_count = len(long_returns)

    if long_count == 0:
        return {
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "ratio": float("nan"),
            "rolling_ratio": pd.Series(dtype=float),
            "return_distribution": [],
            "sample_count": 0,
            "warning": warning,
        }

    wins = long_returns[long_returns > 0]
    losses = long_returns[long_returns < 0]

    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0

    if avg_loss == 0.0:
        ratio = float("inf") if avg_win > 0 else float("nan")
    else:
        ratio = avg_win / avg_loss

    # Rolling P/L ratio
    long_ret_series = pd.Series(returns_arr, index=ret_dates)
    long_flag_series = pd.Series(long_mask.astype(float), index=ret_dates)

    def _rolling_pl(window):
        rets = window.values
        flags = long_flag_series.loc[window.index].values
        masked = rets[flags > 0]
        if len(masked) == 0:
            return float("nan")
        w = masked[masked > 0]
        l = masked[masked < 0]
        aw = float(np.mean(w)) if len(w) > 0 else 0.0
        al = float(np.mean(np.abs(l))) if len(l) > 0 else 0.0
        if al == 0.0:
            return float("inf") if aw > 0 else float("nan")
        return aw / al

    rolling_ratio = long_ret_series.rolling(
        window=rolling_window, min_periods=1,
    ).apply(_rolling_pl, raw=False)

    return {
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "ratio": ratio,
        "rolling_ratio": rolling_ratio,
        "return_distribution": long_returns.tolist(),
        "sample_count": long_count,
        "warning": warning,
    }
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/factor_eval/test_profit_loss.py -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add src/oxq/factor_eval/profit_loss.py tests/factor_eval/test_profit_loss.py
git commit -m "feat(factor_eval): add profit/loss ratio computation"
```

---

### Task 2: Cash Period Value Analysis (`cash_period.py`)

**Files:**
- Create: `src/oxq/factor_eval/cash_period.py`
- Create: `tests/factor_eval/test_cash_period.py`

**Step 1: Write failing tests**

```python
# tests/factor_eval/test_cash_period.py
"""Tests for cash period value analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oxq.factor_eval.cash_period import compute_cash_period_value


class TestComputeCashPeriodValue:
    """Test cash period analysis with hand-calculated values."""

    @pytest.fixture()
    def data(self) -> dict:
        """10-point factor + return series.

        factor:  [0.5, -0.3, 0.8, -0.1, 0.6, -0.4, 0.2, 0.9, -0.5, 0.3]
        returns: [0.02, -0.01, 0.03, 0.01, -0.02, -0.03, 0.01, 0.04, -0.02, 0.01]

        threshold = 0:
        Holding (factor > 0): indices 0,2,4,6,7,9 → returns [0.02, 0.03, -0.02, 0.01, 0.04, 0.01]
          mean = 0.015
        Cash (factor <= 0): indices 1,3,5,8 → returns [-0.01, 0.01, -0.03, -0.02]
          mean = -0.0125
        """
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        factor_values = pd.Series(
            [0.5, -0.3, 0.8, -0.1, 0.6, -0.4, 0.2, 0.9, -0.5, 0.3],
            index=pd.MultiIndex.from_arrays(
                [dates, ["a"] * 10], names=["date", "asset"],
            ),
        )
        forward_returns = pd.DataFrame(
            {"a": [0.02, -0.01, 0.03, 0.01, -0.02, -0.03, 0.01, 0.04, -0.02, 0.01]},
            index=dates,
        )
        return {"factor_values": factor_values, "forward_returns": forward_returns}

    def test_holding_avg_return(self, data: dict) -> None:
        result = compute_cash_period_value(**data)
        assert result["holding_avg_return"] == pytest.approx(0.015)

    def test_cash_avg_return(self, data: dict) -> None:
        result = compute_cash_period_value(**data)
        assert result["cash_avg_return"] == pytest.approx(-0.0125)

    def test_return_spread(self, data: dict) -> None:
        """Spread = holding - cash = 0.015 - (-0.0125) = 0.0275."""
        result = compute_cash_period_value(**data)
        assert result["return_spread"] == pytest.approx(0.0275)

    def test_cash_max_loss(self, data: dict) -> None:
        """Max loss in cash period returns: min(-0.01, 0.01, -0.03, -0.02) = -0.03."""
        result = compute_cash_period_value(**data)
        assert result["cash_max_loss"] == pytest.approx(-0.03)

    def test_period_ratios(self, data: dict) -> None:
        """Holding: 6/10 = 60%, Cash: 4/10 = 40%."""
        result = compute_cash_period_value(**data)
        assert result["holding_ratio"] == pytest.approx(0.6)
        assert result["cash_ratio"] == pytest.approx(0.4)

    def test_no_overlap(self, data: dict) -> None:
        """Holding + cash should cover all sample points."""
        result = compute_cash_period_value(**data)
        assert result["holding_count"] + result["cash_count"] == result["total_count"]

    def test_warning_on_small_sample(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        fv = pd.Series(
            [0.1, 0.2, 0.3],
            index=pd.MultiIndex.from_arrays(
                [dates, ["a"] * 3], names=["date", "asset"],
            ),
        )
        fr = pd.DataFrame({"a": [0.01, 0.02, 0.03]}, index=dates)
        result = compute_cash_period_value(fv, fr)
        assert result["warning"] is not None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/factor_eval/test_cash_period.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/oxq/factor_eval/cash_period.py
"""Cash period value analysis for time-series factor evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_SAMPLE_WARN = 60


def compute_cash_period_value(
    factor_values: pd.Series,
    forward_returns: pd.DataFrame,
    signal_threshold: float = 0.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Analyze returns during holding vs cash periods.

    Holding period: factor > signal_threshold.
    Cash period: factor <= signal_threshold.

    Parameters
    ----------
    factor_values
        Series with MultiIndex (date, asset).
    forward_returns
        DataFrame with index=date, columns=asset.
    signal_threshold
        Threshold separating holding from cash. Default 0.
    start_date, end_date
        Optional date range filter.

    Returns
    -------
    dict with holding_avg_return, cash_avg_return, return_spread,
    cash_max_loss, holding_ratio, cash_ratio, holding_count,
    cash_count, total_count, warning.
    """
    dates = factor_values.index.get_level_values("date")
    assets = factor_values.index.get_level_values("asset")

    factors = []
    returns = []

    for date, asset, fval in zip(dates, assets, factor_values.values):
        if start_date and str(date.date()) < start_date:
            continue
        if end_date and str(date.date()) > end_date:
            continue
        if asset not in forward_returns.columns or date not in forward_returns.index:
            continue
        ret = forward_returns.loc[date, asset]
        if np.isnan(fval) or np.isnan(ret):
            continue
        factors.append(fval)
        returns.append(ret)

    factors_arr = np.array(factors)
    returns_arr = np.array(returns)
    n = len(factors_arr)

    warning = None
    if n < _MIN_SAMPLE_WARN:
        warning = f"Sample size {n} < {_MIN_SAMPLE_WARN}, results for reference only."

    if n == 0:
        return {
            "holding_avg_return": float("nan"),
            "cash_avg_return": float("nan"),
            "return_spread": float("nan"),
            "cash_max_loss": float("nan"),
            "holding_ratio": float("nan"),
            "cash_ratio": float("nan"),
            "holding_count": 0,
            "cash_count": 0,
            "total_count": 0,
            "warning": warning,
        }

    holding_mask = factors_arr > signal_threshold
    cash_mask = ~holding_mask

    holding_returns = returns_arr[holding_mask]
    cash_returns = returns_arr[cash_mask]

    holding_avg = float(np.mean(holding_returns)) if len(holding_returns) > 0 else float("nan")
    cash_avg = float(np.mean(cash_returns)) if len(cash_returns) > 0 else float("nan")

    if np.isnan(holding_avg) or np.isnan(cash_avg):
        spread = float("nan")
    else:
        spread = holding_avg - cash_avg

    cash_max_loss = float(np.min(cash_returns)) if len(cash_returns) > 0 else float("nan")

    holding_count = int(np.sum(holding_mask))
    cash_count = int(np.sum(cash_mask))

    return {
        "holding_avg_return": holding_avg,
        "cash_avg_return": cash_avg,
        "return_spread": spread,
        "cash_max_loss": cash_max_loss,
        "holding_ratio": holding_count / n,
        "cash_ratio": cash_count / n,
        "holding_count": holding_count,
        "cash_count": cash_count,
        "total_count": n,
        "warning": warning,
    }
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/factor_eval/test_cash_period.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/oxq/factor_eval/cash_period.py tests/factor_eval/test_cash_period.py
git commit -m "feat(factor_eval): add cash period value analysis"
```

---

### Task 3: Market State Conditional Analysis (`conditional.py`)

**Files:**
- Create: `src/oxq/factor_eval/conditional.py`
- Create: `tests/factor_eval/test_conditional.py`

**Step 1: Write failing tests**

```python
# tests/factor_eval/test_conditional.py
"""Tests for market state conditional analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oxq.factor_eval.conditional import compute_conditional_analysis


class TestComputeConditionalAnalysis:
    """Test conditional analysis with hand-calculated values."""

    @pytest.fixture()
    def data(self) -> dict:
        """12-point dataset with 3 market states.

        States: trend(4), ranging(4), crash(4)
        factor:  [0.5, 0.8, 0.3, 0.6,  0.2, -0.1, 0.4, -0.3,  -0.5, -0.8, 0.1, -0.2]
        returns: [0.03, 0.04, 0.02, 0.05,  0.01, -0.01, 0.02, -0.02,  -0.05, -0.03, -0.01, -0.04]
        states:  [trend, trend, trend, trend, ranging, ranging, ranging, ranging, crash, crash, crash, crash]

        Trend: factor [0.5,0.8,0.3,0.6] all > 0 → long signals
          returns [0.03,0.04,0.02,0.05] all > 0 → hit rate = 4/4 = 1.0
        Ranging: factor [0.2,-0.1,0.4,-0.3]
          long [0.2,0.4] returns [0.01,0.02] → 2 hits
          short [-0.1,-0.3] returns [-0.01,-0.02] → 2 hits
          hit rate = 4/4 = 1.0
        Crash: factor [-0.5,-0.8,0.1,-0.2]
          long [0.1] returns [-0.01] → 0 hits
          short [-0.5,-0.8,-0.2] returns [-0.05,-0.03,-0.04] → 3 hits (all < 0)
          hit rate = 3/4 = 0.75
        """
        dates = pd.date_range("2024-01-01", periods=12, freq="B")
        factor_values = pd.Series(
            [0.5, 0.8, 0.3, 0.6, 0.2, -0.1, 0.4, -0.3, -0.5, -0.8, 0.1, -0.2],
            index=pd.MultiIndex.from_arrays(
                [dates, ["a"] * 12], names=["date", "asset"],
            ),
        )
        forward_returns = pd.DataFrame(
            {"a": [0.03, 0.04, 0.02, 0.05, 0.01, -0.01, 0.02, -0.02, -0.05, -0.03, -0.01, -0.04]},
            index=dates,
        )
        market_state = pd.Series(
            ["trend"] * 4 + ["ranging"] * 4 + ["crash"] * 4,
            index=dates,
        )
        return {
            "factor_values": factor_values,
            "forward_returns": forward_returns,
            "market_state": market_state,
        }

    def test_returns_all_states(self, data: dict) -> None:
        result = compute_conditional_analysis(**data)
        assert set(result["by_state"].keys()) == {"trend", "ranging", "crash"}

    def test_trend_hit_rate(self, data: dict) -> None:
        result = compute_conditional_analysis(**data)
        assert result["by_state"]["trend"]["hit_rate"] == pytest.approx(1.0)

    def test_crash_hit_rate(self, data: dict) -> None:
        result = compute_conditional_analysis(**data)
        assert result["by_state"]["crash"]["hit_rate"] == pytest.approx(0.75)

    def test_sample_counts(self, data: dict) -> None:
        result = compute_conditional_analysis(**data)
        assert result["by_state"]["trend"]["sample_count"] == 4
        assert result["by_state"]["ranging"]["sample_count"] == 4
        assert result["by_state"]["crash"]["sample_count"] == 4

    def test_small_sample_warning(self) -> None:
        """State with < 30 samples gets a warning."""
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        fv = pd.Series(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            index=pd.MultiIndex.from_arrays(
                [dates, ["a"] * 6], names=["date", "asset"],
            ),
        )
        fr = pd.DataFrame({"a": [0.01, 0.02, 0.03, 0.01, 0.02, 0.03]}, index=dates)
        ms = pd.Series(["trend"] * 3 + ["crash"] * 3, index=dates)
        result = compute_conditional_analysis(fv, fr, ms)
        assert result["by_state"]["trend"]["warning"] is not None

    def test_none_market_state_returns_empty(self) -> None:
        """Graceful degradation when no market_state provided."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        fv = pd.Series(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            index=pd.MultiIndex.from_arrays(
                [dates, ["a"] * 5], names=["date", "asset"],
            ),
        )
        fr = pd.DataFrame({"a": [0.01, 0.02, 0.03, 0.01, 0.02]}, index=dates)
        result = compute_conditional_analysis(fv, fr, market_state=None)
        assert result["by_state"] == {}
        assert result["skipped"] is True

    def test_avg_return_per_state(self, data: dict) -> None:
        result = compute_conditional_analysis(**data)
        # Trend avg return: mean(0.03, 0.04, 0.02, 0.05) = 0.035
        assert result["by_state"]["trend"]["avg_return"] == pytest.approx(0.035)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/factor_eval/test_conditional.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/oxq/factor_eval/conditional.py
"""Market state conditional analysis for factor evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oxq.factor_eval.hit_rate import compute_hit_rate
from oxq.factor_eval.profit_loss import compute_profit_loss_ratio

_MIN_STATE_SAMPLE = 30


def compute_conditional_analysis(
    factor_values: pd.Series,
    forward_returns: pd.DataFrame,
    market_state: pd.Series | None = None,
    signal_threshold: float = 0.0,
    min_sample_warn: int = _MIN_STATE_SAMPLE,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Compute factor metrics grouped by market state.

    Parameters
    ----------
    factor_values
        Series with MultiIndex (date, asset).
    forward_returns
        DataFrame with index=date, columns=asset.
    market_state
        Series with index=date, values like 'trend'/'ranging'/'crash'.
        If None, returns empty result with skipped=True.
    signal_threshold
        Threshold for signals.
    min_sample_warn
        Minimum samples per state before warning. Default 30.
    start_date, end_date
        Optional date range filter.

    Returns
    -------
    dict with by_state (dict of state -> metrics), skipped (bool).
    """
    if market_state is None:
        return {"by_state": {}, "skipped": True}

    dates = factor_values.index.get_level_values("date")
    assets = factor_values.index.get_level_values("asset")

    # Build per-point records
    records = []
    for date, asset, fval in zip(dates, assets, factor_values.values):
        if start_date and str(date.date()) < start_date:
            continue
        if end_date and str(date.date()) > end_date:
            continue
        if asset not in forward_returns.columns or date not in forward_returns.index:
            continue
        if date not in market_state.index:
            continue
        ret = forward_returns.loc[date, asset]
        if np.isnan(fval) or np.isnan(ret):
            continue
        records.append({
            "date": date, "asset": asset, "factor": fval,
            "return": ret, "state": market_state[date],
        })

    if not records:
        return {"by_state": {}, "skipped": False}

    df = pd.DataFrame(records)
    states = df["state"].unique()
    by_state = {}

    for state in sorted(states):
        mask = df["state"] == state
        state_df = df[mask]
        n = len(state_df)

        factors_arr = state_df["factor"].values
        returns_arr = state_df["return"].values

        # Hit rate
        long_mask = factors_arr > signal_threshold
        short_mask = factors_arr < signal_threshold
        long_hits = np.sum(long_mask & (returns_arr > 0))
        short_hits = np.sum(short_mask & (returns_arr < 0))
        signal_count = int(np.sum(long_mask) + np.sum(short_mask))
        hit_rate = float((long_hits + short_hits) / signal_count) if signal_count > 0 else float("nan")

        # P/L ratio
        long_returns = returns_arr[long_mask]
        wins = long_returns[long_returns > 0]
        losses = long_returns[long_returns < 0]
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0
        if avg_loss == 0.0:
            pl_ratio = float("inf") if avg_win > 0 else float("nan")
        else:
            pl_ratio = avg_win / avg_loss

        avg_return = float(np.mean(returns_arr))

        warning = None
        if n < min_sample_warn:
            warning = f"Sample size {n} < {min_sample_warn}, results for reference only."

        by_state[state] = {
            "hit_rate": hit_rate,
            "profit_loss_ratio": pl_ratio,
            "avg_return": avg_return,
            "sample_count": n,
            "warning": warning,
        }

    return {"by_state": by_state, "skipped": False}
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/factor_eval/test_conditional.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/oxq/factor_eval/conditional.py tests/factor_eval/test_conditional.py
git commit -m "feat(factor_eval): add market state conditional analysis"
```

---

### Task 4: Multi-Asset Comparison (`comparison.py`)

**Files:**
- Create: `src/oxq/factor_eval/comparison.py`
- Create: `tests/factor_eval/test_comparison.py`

**Step 1: Write failing tests**

```python
# tests/factor_eval/test_comparison.py
"""Tests for multi-asset comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oxq.factor_eval.bundle import create_bundle
from oxq.factor_eval.comparison import compute_asset_comparison


class TestComputeAssetComparison:
    """Test multi-asset comparison."""

    @pytest.fixture()
    def multi_asset_bundle(self):
        """Bundle with 2 assets, 30 dates each, 80 price dates."""
        np.random.seed(42)
        factor_dates = pd.date_range("2024-01-01", periods=30, freq="B")
        price_dates = pd.date_range("2024-01-01", periods=80, freq="B")

        # Asset A: trending factor
        # Asset B: random factor
        fv_a = np.random.randn(30) * 0.5
        fv_b = np.random.randn(30) * 0.5

        factor_values = pd.Series(
            np.concatenate([fv_a, fv_b]),
            index=pd.MultiIndex.from_arrays(
                [np.tile(factor_dates, 2),
                 ["asset_a"] * 30 + ["asset_b"] * 30],
                names=["date", "asset"],
            ),
        )
        prices = pd.DataFrame(
            {
                "asset_a": 100 * np.exp(np.cumsum(np.random.randn(80) * 0.01)),
                "asset_b": 200 * np.exp(np.cumsum(np.random.randn(80) * 0.01)),
            },
            index=price_dates,
        )
        return create_bundle(factor_values, prices, forward_periods=[1, 5, 10, 20])

    def test_returns_both_assets(self, multi_asset_bundle) -> None:
        result = compute_asset_comparison(multi_asset_bundle, forward_periods=[1, 5])
        assert "asset_a" in result["by_asset"]
        assert "asset_b" in result["by_asset"]

    def test_each_asset_has_metrics(self, multi_asset_bundle) -> None:
        result = compute_asset_comparison(multi_asset_bundle, forward_periods=[1, 5])
        for asset_result in result["by_asset"].values():
            assert "hit_rate" in asset_result
            assert "profit_loss_ratio" in asset_result
            assert "decay_half_life" in asset_result
            assert "sample_count" in asset_result

    def test_single_asset_skips_comparison(self) -> None:
        """Single-asset bundle should return skipped=True."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        price_dates = pd.date_range("2024-01-01", periods=80, freq="B")
        fv = pd.Series(
            np.random.randn(30),
            index=pd.MultiIndex.from_arrays(
                [dates, ["a"] * 30], names=["date", "asset"],
            ),
        )
        prices = pd.DataFrame(
            {"a": 100 * np.exp(np.cumsum(np.random.randn(80) * 0.01))},
            index=price_dates,
        )
        bundle = create_bundle(fv, prices, forward_periods=[1, 5])
        result = compute_asset_comparison(bundle, forward_periods=[1, 5])
        assert result["skipped"] is True

    def test_sample_counts_are_correct(self, multi_asset_bundle) -> None:
        result = compute_asset_comparison(multi_asset_bundle, forward_periods=[1])
        assert result["by_asset"]["asset_a"]["sample_count"] == 30
        assert result["by_asset"]["asset_b"]["sample_count"] == 30

    def test_result_is_serializable(self, multi_asset_bundle) -> None:
        import json

        result = compute_asset_comparison(multi_asset_bundle, forward_periods=[1, 5])
        # Remove rolling series before serialization
        for asset_data in result["by_asset"].values():
            asset_data.pop("rolling_hit_rate", None)
            asset_data.pop("decay_correlations", None)
        json.dumps(result)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/factor_eval/test_comparison.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/oxq/factor_eval/comparison.py
"""Multi-asset comparison for factor evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oxq.factor_eval.bundle import FactorBundle
from oxq.factor_eval.decay_curve import compute_decay_curve
from oxq.factor_eval.hit_rate import compute_hit_rate
from oxq.factor_eval.profit_loss import compute_profit_loss_ratio
from oxq.factor_eval.returns import compute_forward_returns


def compute_asset_comparison(
    bundle: FactorBundle,
    forward_periods: list[int],
    signal_threshold: float = 0.0,
    method: str = "spearman",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Compare factor performance across assets in a multi-asset FactorBundle.

    Groups factor_values by asset and runs hit rate, P/L ratio, and
    decay analysis independently per asset.

    Parameters
    ----------
    bundle
        FactorBundle with MultiIndex (date, asset) factor_values.
    forward_periods
        Forward periods for decay analysis.
    signal_threshold
        Threshold for signals.
    method
        Correlation method for decay curve.
    start_date, end_date
        Optional date range filter.

    Returns
    -------
    dict with by_asset (dict of asset -> metrics), skipped (bool).
    Single-asset bundles return skipped=True.
    """
    assets = bundle.factor_values.index.get_level_values("asset").unique()

    if len(assets) < 2:
        return {"by_asset": {}, "skipped": True}

    # Compute forward returns once
    fwd_returns = compute_forward_returns(
        bundle.prices, forward_periods,
        suspension_days=bundle.suspension_days,
    )

    by_asset = {}
    for asset in sorted(assets):
        # Extract single-asset factor values
        asset_fv = bundle.factor_values.loc[
            bundle.factor_values.index.get_level_values("asset") == asset
        ]

        # Hit rate (using first period)
        primary_fwd = fwd_returns[forward_periods[0]]
        hr = compute_hit_rate(
            asset_fv, primary_fwd,
            signal_threshold=signal_threshold,
            start_date=start_date, end_date=end_date,
        )

        # P/L ratio
        pl = compute_profit_loss_ratio(
            asset_fv, primary_fwd,
            signal_threshold=signal_threshold,
            start_date=start_date, end_date=end_date,
        )

        # Decay curve
        decay = compute_decay_curve(
            asset_fv, fwd_returns,
            periods=forward_periods,
            method=method,
            start_date=start_date, end_date=end_date,
        )

        by_asset[asset] = {
            "hit_rate": hr["long_hit_rate"],
            "profit_loss_ratio": pl["ratio"],
            "decay_half_life": decay["half_life"],
            "sample_count": hr["sample_count"],
            "rolling_hit_rate": hr["rolling_hit_rate"],
            "decay_correlations": decay["correlations"],
        }

    return {"by_asset": by_asset, "skipped": False}
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/factor_eval/test_comparison.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/oxq/factor_eval/comparison.py tests/factor_eval/test_comparison.py
git commit -m "feat(factor_eval): add multi-asset comparison"
```

---

### Task 5: Extend Tearsheet with P1 Content

**Files:**
- Modify: `src/oxq/factor_eval/tearsheet.py`
- Modify: `tests/factor_eval/test_tearsheet.py`

**Step 1: Write failing tests (append to existing file)**

Add these tests to `tests/factor_eval/test_tearsheet.py`:

```python
class TestTearsheetP1:
    """Test P1 extensions to tearsheet."""

    @pytest.fixture()
    def bundle_with_state(self):
        """Bundle with market_state for conditional analysis."""
        np.random.seed(42)
        factor_dates = pd.date_range("2024-01-01", periods=50, freq="B")
        price_dates = pd.date_range("2024-01-01", periods=120, freq="B")

        factor_values = pd.Series(
            np.random.randn(50),
            index=pd.MultiIndex.from_arrays(
                [factor_dates, ["a"] * 50], names=["date", "asset"],
            ),
        )
        prices = pd.DataFrame(
            {"a": 100 * np.exp(np.cumsum(np.random.randn(120) * 0.01))},
            index=price_dates,
        )
        market_state = pd.Series(
            (["trend"] * 17 + ["ranging"] * 17 + ["crash"] * 16),
            index=factor_dates,
        )

        from oxq.factor_eval.bundle import create_bundle
        return create_bundle(
            factor_values, prices, forward_periods=[1, 5],
            market_state=market_state,
        )

    @pytest.fixture()
    def multi_asset_bundle(self):
        """Bundle with 2 assets."""
        np.random.seed(42)
        factor_dates = pd.date_range("2024-01-01", periods=50, freq="B")
        price_dates = pd.date_range("2024-01-01", periods=120, freq="B")

        factor_values = pd.Series(
            np.random.randn(100),
            index=pd.MultiIndex.from_arrays(
                [np.tile(factor_dates, 2),
                 ["a"] * 50 + ["b"] * 50],
                names=["date", "asset"],
            ),
        )
        prices = pd.DataFrame(
            {
                "a": 100 * np.exp(np.cumsum(np.random.randn(120) * 0.01)),
                "b": 200 * np.exp(np.cumsum(np.random.randn(120) * 0.01)),
            },
            index=price_dates,
        )

        from oxq.factor_eval.bundle import create_bundle
        return create_bundle(factor_values, prices, forward_periods=[1, 5])

    def test_includes_profit_loss(self, bundle_with_state) -> None:
        result = generate_tearsheet(bundle_with_state, forward_periods=[1, 5])
        assert "profit_loss" in result["summary"]
        assert "ratio" in result["summary"]["profit_loss"]

    def test_includes_cash_period(self, bundle_with_state) -> None:
        result = generate_tearsheet(bundle_with_state, forward_periods=[1, 5])
        assert "cash_period" in result["summary"]
        assert "return_spread" in result["summary"]["cash_period"]

    def test_includes_conditional_when_state_present(self, bundle_with_state) -> None:
        result = generate_tearsheet(bundle_with_state, forward_periods=[1, 5])
        assert "conditional" in result["summary"]
        assert result["summary"]["conditional"]["skipped"] is False

    def test_conditional_skipped_without_state(self) -> None:
        np.random.seed(42)
        factor_dates = pd.date_range("2024-01-01", periods=50, freq="B")
        price_dates = pd.date_range("2024-01-01", periods=120, freq="B")
        fv = pd.Series(
            np.random.randn(50),
            index=pd.MultiIndex.from_arrays(
                [factor_dates, ["a"] * 50], names=["date", "asset"],
            ),
        )
        prices = pd.DataFrame(
            {"a": 100 * np.exp(np.cumsum(np.random.randn(120) * 0.01))},
            index=price_dates,
        )
        from oxq.factor_eval.bundle import create_bundle
        bundle = create_bundle(fv, prices, forward_periods=[1, 5])
        result = generate_tearsheet(bundle, forward_periods=[1, 5])
        assert result["summary"]["conditional"]["skipped"] is True

    def test_includes_comparison_for_multi_asset(self, multi_asset_bundle) -> None:
        result = generate_tearsheet(multi_asset_bundle, forward_periods=[1, 5])
        assert "comparison" in result["summary"]
        assert result["summary"]["comparison"]["skipped"] is False

    def test_comparison_skipped_for_single_asset(self, bundle_with_state) -> None:
        result = generate_tearsheet(bundle_with_state, forward_periods=[1, 5])
        assert result["summary"]["comparison"]["skipped"] is True

    def test_p1_charts_exist(self, bundle_with_state, tmp_path) -> None:
        result = generate_tearsheet(
            bundle_with_state, forward_periods=[1, 5],
            output_dir=str(tmp_path),
        )
        assert "profit_loss_distribution" in result["charts"]
        assert os.path.exists(result["charts"]["profit_loss_distribution"])
        assert "cash_period" in result["charts"]
        assert os.path.exists(result["charts"]["cash_period"])
```

**Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/factor_eval/test_tearsheet.py::TestTearsheetP1 -v`
Expected: FAIL — missing keys in result

**Step 3: Extend `generate_tearsheet` implementation**

Add imports and P1 calls to `tearsheet.py`. After the existing P0 sections (bias, hit rate, decay), add:

```python
# After existing imports, add:
from oxq.factor_eval.cash_period import compute_cash_period_value
from oxq.factor_eval.comparison import compute_asset_comparison
from oxq.factor_eval.conditional import compute_conditional_analysis
from oxq.factor_eval.profit_loss import compute_profit_loss_ratio
```

In `generate_tearsheet()`, after the decay curve section and before building the return dict, add:

```python
    # 5. P1: Profit/loss ratio
    pl_result = compute_profit_loss_ratio(
        bundle.factor_values,
        fwd_returns[primary_period],
        signal_threshold=signal_threshold,
        start_date=start_date,
        end_date=end_date,
    )

    # 6. P1: Cash period value
    cp_result = compute_cash_period_value(
        bundle.factor_values,
        fwd_returns[primary_period],
        signal_threshold=signal_threshold,
        start_date=start_date,
        end_date=end_date,
    )

    # 7. P1: Conditional analysis (auto-skip if no market_state)
    cond_result = compute_conditional_analysis(
        bundle.factor_values,
        fwd_returns[primary_period],
        market_state=bundle.market_state,
        signal_threshold=signal_threshold,
        start_date=start_date,
        end_date=end_date,
    )

    # 8. P1: Multi-asset comparison (auto-skip if single asset)
    comp_result = compute_asset_comparison(
        bundle,
        forward_periods=forward_periods,
        signal_threshold=signal_threshold,
        method=method,
        start_date=start_date,
        end_date=end_date,
    )

    # 9. P1 charts
    pl_dist_path = _plot_return_distribution(
        pl_result["return_distribution"], output_dir,
    )
    cp_path = _plot_cash_period(cp_result, output_dir)
```

Add to the return dict under `summary`:
```python
        "profit_loss": pl_result,
        "cash_period": cp_result,
        "conditional": cond_result,
        "comparison": comp_result,
```

Add to charts:
```python
        "profit_loss_distribution": pl_dist_path,
        "cash_period": cp_path,
```

Add new plotting functions:

```python
def _plot_return_distribution(returns: list[float], output_dir: str) -> str:
    """Plot histogram of holding-period returns."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if returns:
        ax.hist(returns, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.set_title("Holding Period Return Distribution")
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "profit_loss_distribution.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_cash_period(cp_result: dict, output_dir: str) -> str:
    """Plot holding vs cash period average returns."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Holding", "Cash"]
    values = [
        cp_result.get("holding_avg_return", 0) or 0,
        cp_result.get("cash_avg_return", 0) or 0,
    ]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]
    ax.bar(labels, values, color=colors, edgecolor="black", alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_ylabel("Average Return")
    ax.set_title("Holding vs Cash Period Returns")
    ax.grid(True, alpha=0.3, axis="y")
    path = os.path.join(output_dir, "cash_period.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/factor_eval/test_tearsheet.py -v`
Expected: PASS (all P0 + P1 tests)

**Step 5: Commit**

```bash
git add src/oxq/factor_eval/tearsheet.py tests/factor_eval/test_tearsheet.py
git commit -m "feat(factor_eval): extend tearsheet with P1 content (profit/loss, cash period, conditional, comparison)"
```

---

### Task 6: Update `__init__.py` and Final Verification

**Files:**
- Modify: `src/oxq/factor_eval/__init__.py`

**Step 1: Add P1 exports**

Add imports (before the lazy `__getattr__`):
```python
from oxq.factor_eval.cash_period import compute_cash_period_value
from oxq.factor_eval.comparison import compute_asset_comparison
from oxq.factor_eval.conditional import compute_conditional_analysis
from oxq.factor_eval.profit_loss import compute_profit_loss_ratio
```

Add to `__all__`:
```python
    # P1 metrics
    "compute_profit_loss_ratio",
    "compute_cash_period_value",
    "compute_conditional_analysis",
    "compute_asset_comparison",
```

**Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass.

**Step 3: Run lint**

Run: `uv run ruff check src/oxq/factor_eval/ tests/factor_eval/`

**Step 4: Fix any issues and commit**

```bash
git add -u
git commit -m "feat(factor_eval): export all P1 modules, lint clean"
```
