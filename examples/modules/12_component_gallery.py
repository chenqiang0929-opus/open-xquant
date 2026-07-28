"""Run every built-in component once and print real output.

This is not a curated set of hand-picked examples -- it iterates the actual
component registry (``oxq.core.registry.list_indicators`` /
``list_signals`` / ``list_rules`` / ``list_portfolio_optimizers``) so it goes
stale loudly (an exception) rather than silently (a doc nobody re-reads).

Four sections, one per component family:

    1. Indicators (48)  -- oxq.indicators.*,            .compute(bars, **params)
    2. Signals    (8)   -- oxq.signals.*,                .compute(bars, **params)
    3. Rules      (10)  -- oxq.rules.*,                  .evaluate(symbol, row, portfolio, prices)
    4. Optimizers (6)   -- oxq.portfolio.optimizers.*,   .optimize(signals, indicators)

Each family has a different calling convention -- that is itself the main
thing worth learning here. See docs/component-cookbook.md for the narrative
version of this file plus corrections to component names/params that don't
match memory or older notes.

Run: python examples/modules/12_component_gallery.py
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

import oxq
from oxq.core.registry import (
    get_indicator_metadata,
    list_indicators,
    list_portfolio_optimizers,
    list_rules,
    list_signals,
)
from oxq.core.types import Portfolio, Position
from oxq.data.market import LocalMarketDataProvider
from oxq.portfolio import optimizers as opt_mod

pd.set_option("display.width", 120)

# ===========================================================================
# Shared fixtures
# ===========================================================================

DATA_DIR = "~/.oxq/data/market"
SYMBOLS = ["510300", "510050", "510500"]


def load_bars() -> dict[str, pd.DataFrame]:
    market = LocalMarketDataProvider(data_dir=DATA_DIR)
    return {s: market.get_bars(s, "2023-01-01", "2024-12-31") for s in SYMBOLS}


def with_fundamentals(bars: pd.DataFrame) -> pd.DataFrame:
    """Attach synthetic fundamental columns for valuation/quality indicators.

    Real usage needs a data provider that supplies these columns (eps,
    book_value_per_share, net_income, ...); the local OHLCV parquet files
    used elsewhere in this repo do not carry them. These numbers exist only
    to exercise the formula, not as anything resembling real fundamentals.
    """
    out = bars.copy()
    n = len(out)
    out["eps"] = 0.35
    out["book_value_per_share"] = 4.20
    out["total_shares"] = 1.2e9
    out["net_income"] = 5.0e8
    out["operating_cash_flow"] = 4.5e8
    out["total_assets"] = 8.0e9
    out["revenue"] = 3.0e9
    out["roe"] = 0.12
    return out


# ===========================================================================
# Section 1: Indicators (48)
# ===========================================================================

# Indicators whose compute() reads fundamental columns rather than OHLCV.
# Confirmed by inspect.signature(cls.compute) -- these are the only ones
# whose default kwargs reference something other than open/high/low/close/volume.
_FUNDAMENTAL_INDICATORS = {
    "PE", "PB", "EP", "BP", "MarketCap",
    "AccrualRatio", "CashFlowRatio", "NetProfitMargin", "ROEChange",
    "TurnoverRate",
}

# MACDSignal and MACDHistogram read columns named by their macd_col/signal_col
# defaults ("macd", "macd_signal") -- they don't compute MACDLine themselves.
_MACD_CHAIN_INDICATORS = {"MACDSignal", "MACDHistogram"}

# PowerRatio/Ratio need two arbitrary columns; there's no sensible default,
# so col_a/col_b are empty strings until the caller supplies them.
_RATIO_INDICATORS = {"Ratio", "PowerRatio"}


def run_indicators() -> None:
    print("=" * 70)
    print(f"SECTION 1: INDICATORS ({len(list_indicators())})")
    print("=" * 70)

    bars_by_symbol = load_bars()
    sample = with_fundamentals(bars_by_symbol["510300"])

    # MACDSignal/MACDHistogram need MACDLine's output already present under
    # its default read-column name, so compute the chain once up front.
    sample["macd"] = oxq.indicators.MACDLine().compute(sample, column="close")
    sample["macd_signal"] = oxq.indicators.MACDSignal().compute(sample, macd_col="macd")

    for name in sorted(list_indicators()):
        cls = getattr(oxq.indicators, name)
        meta = get_indicator_metadata(name) or {}
        category = meta.get("category", "?")

        if name == "RPS":
            # The only indicator that ranks symbols against each other on
            # the same date -- it needs the whole panel, not one DataFrame.
            result = cls().compute_cross_section(bars_by_symbol, column="close", period=60)
            last = {sym: round(series.iloc[-1], 2) for sym, series in result.items()}
            print(f"  {name:<18} [{category:<10}] compute_cross_section -> {last}")
            continue

        if name in _RATIO_INDICATORS:
            # Ratio/PowerRatio need mom_20 as an example numerator/denominator pair.
            sample["mom_20"] = oxq.indicators.NdayReturn().compute(sample, column="close", period=20)
            sample["vol_20"] = oxq.indicators.RollingVolatility().compute(sample, column="close", period=20)
            value = cls().compute(sample, col_a="mom_20", col_b="vol_20")
        elif name in _MACD_CHAIN_INDICATORS:
            value = cls().compute(sample)  # defaults already point at macd/macd_signal
        elif name in _FUNDAMENTAL_INDICATORS:
            value = cls().compute(sample)  # defaults already point at synthetic columns
        else:
            value = cls().compute(sample)

        last_valid = value.dropna().iloc[-1] if value.dropna().size else float("nan")
        print(f"  {name:<18} [{category:<10}] last={last_valid:.4f}")


# ===========================================================================
# Section 2: Signals (8)
# ===========================================================================


def run_signals() -> None:
    print()
    print("=" * 70)
    print(f"SECTION 2: SIGNALS ({len(list_signals())})")
    print("=" * 70)

    bars = load_bars()["510300"].copy()
    bars["sma_fast"] = oxq.indicators.SMA().compute(bars, column="close", period=10)
    bars["sma_slow"] = oxq.indicators.SMA().compute(bars, column="close", period=50)
    bars["roc_12"] = oxq.indicators.ROC().compute(bars, column="close", period=12)
    bars["rsi_14"] = oxq.indicators.RSI().compute(bars, column="close", period=14)
    bars["trend_ok"] = (bars["close"] > bars["sma_slow"]).astype(bool)
    bars["momentum_ok"] = (bars["rsi_14"] > 50).astype(bool)

    demos = {
        "Comparison": lambda: oxq.signals.Comparison().compute(
            bars, left="close", right="sma_slow", relationship="gt"
        ),
        "Crossover": lambda: oxq.signals.Crossover().compute(bars, fast="sma_fast", slow="sma_slow"),
        "Threshold": lambda: oxq.signals.Threshold().compute(
            bars, column="rsi_14", threshold=70.0, relationship="gt"
        ),
        "Composite": lambda: oxq.signals.Composite().compute(
            bars, signals=["trend_ok", "momentum_ok"], logic="and"
        ),
        "Formula": lambda: oxq.signals.Formula().compute(bars, expr="close > sma_slow and rsi_14 > 50"),
        "Peak": lambda: oxq.signals.Peak().compute(bars, column="close", kind="peak", order=3),
        "ROCTiming": lambda: oxq.signals.ROCTiming().compute(
            bars, column="roc_12", mode="fixed", bottom=-5.0, top=5.0
        ),
        "Timestamp": lambda: oxq.signals.Timestamp().compute(bars, rule="month_start"),
    }

    for name in sorted(list_signals()):
        assert name in demos, f"no demo wired up for signal {name!r}"
        series = demos[name]()
        counts = series.value_counts(dropna=False).to_dict()
        print(f"  {name:<12} value_counts={counts}")


# ===========================================================================
# Section 3: Rules (10)
# ===========================================================================
#
# Rules operate on portfolio state, not raw bars: evaluate(symbol, row,
# portfolio, prices) -> RuleResult. Below constructs a minimal held position
# and shows one scenario where the rule fires and, where meaningful, one
# where it doesn't. MaxDrawdownRisk / DailyLossLimitRisk are excluded from
# the "fires" comparison because they need multi-bar equity history to
# compute a drawdown/daily-loss -- a single synthetic snapshot can't
# demonstrate that honestly, only that the call doesn't raise.


def _held(entry_price: float, shares: int = 100, cash: str = "10000") -> Portfolio:
    """A portfolio holding one position in 510300 at the given entry price."""
    return Portfolio(
        cash=Decimal(cash),
        positions={"510300": Position("510300", shares=shares, avg_cost=Decimal(str(entry_price)))},
    )


def _row(close: float, date: str = "2024-01-15", **extra: float) -> pd.Series:
    """A single bar row with a real timestamp -- several rules key off row.name."""
    data = {"close": close, **extra}
    return pd.Series(data, name=pd.Timestamp(date))


def run_rules() -> None:  # noqa: PLR0915 - deliberately linear, one block per rule
    print()
    print("=" * 70)
    print(f"SECTION 3: RULES ({len(list_rules())})")
    print("=" * 70)

    # --- Price-exit rules read row["close"] and compare against the
    #     position's avg_cost from `portfolio` -- NOT from `prices` or
    #     `portfolio.bar_prices`. That was the actual bug in an earlier
    #     draft of this script: varying `prices` did nothing because these
    #     three rules never look at it.
    stop = oxq.rules.StopLossRule(threshold=0.05)
    pf = _held(entry_price=4.0)
    print("  StopLossRule           holds  (price -2.5%, threshold 5%)   ->", stop.evaluate("510300", _row(3.90), pf))
    print("  StopLossRule           fires  (price -12.5%, threshold 5%)  ->", stop.evaluate("510300", _row(3.50), pf))

    tp = oxq.rules.TakeProfitRule(threshold=0.15)
    pf = _held(entry_price=4.0)
    print("  TakeProfitRule         holds  (price +5%, target 15%)       ->", tp.evaluate("510300", _row(4.20), pf))
    print("  TakeProfitRule         fires  (price +20%, target 15%)      ->", tp.evaluate("510300", _row(4.80), pf))

    # TrailingStopRule tracks its own high-water mark across calls -- a
    # single evaluate() can't show a retracement, so this feeds it a rally
    # to set the high, then a pullback that breaches trail_pct off that high.
    trail = oxq.rules.TrailingStopRule(trail_pct=0.05)
    pf = _held(entry_price=4.0)
    trail.evaluate("510300", _row(4.50), pf)  # sets high-water mark to 4.50
    print("  TrailingStopRule       fires  (5% pullback off 4.50 high)   ->", trail.evaluate("510300", _row(4.25), pf))

    # ExitRule reads row[fast] / row[slow] -- arbitrary named columns, not
    # necessarily "close". It needs those columns present in the row.
    exit_rule = oxq.rules.ExitRule(fast="sma_fast", slow="sma_slow")
    pf = _held(entry_price=4.0)
    print("  ExitRule               fires  (sma_fast < sma_slow)         ->",
          exit_rule.evaluate("510300", _row(3.9, sma_fast=3.8, sma_slow=3.95), pf))

    # MaxDrawdownRisk/DailyLossLimitRisk are stateful: they remember a peak
    # (or day-start) portfolio value across calls via `portfolio.total_value(prices)`.
    # A single call can only ever set the peak, never breach it -- firing
    # requires a prior call at a higher value first.
    # cash="0" here on purpose: total_value() is cash + shares*price, so any
    # cash cushion dilutes the price move and masks the drawdown/loss ratio.
    mdd = oxq.rules.MaxDrawdownRisk(max_drawdown=0.15)
    pf = _held(entry_price=4.0, cash="0")
    mdd.evaluate("510300", _row(4.5), pf, prices={"510300": Decimal("4.5")})  # sets peak value
    print("  MaxDrawdownRisk        fires  (17.8% off the peak this call set) ->",
          mdd.evaluate("510300", _row(3.7), pf, prices={"510300": Decimal("3.7")}))

    dll = oxq.rules.DailyLossLimitRisk(max_daily_loss=0.03)
    pf = _held(entry_price=4.0, cash="0")
    dll.evaluate("510300", _row(4.0, date="2024-01-15"), pf, prices={"510300": Decimal("4.0")})  # day-start value
    print("  DailyLossLimitRisk     fires  (same day, -5% from day start) ->",
          dll.evaluate("510300", _row(3.8, date="2024-01-15"), pf, prices={"510300": Decimal("3.8")}))

    # MaxHoldingsRule only blocks a symbol NOT already held once the
    # portfolio is at capacity -- checking an already-held symbol always
    # passes (that's the "don't force-exit existing positions" behavior).
    mh = oxq.rules.MaxHoldingsRule(max_holdings=1)
    pf = _held(entry_price=4.0)  # already holding 510300 => at capacity
    print("  MaxHoldingsRule        blocks (new symbol, already at capacity 1) ->",
          mh.evaluate("510050", _row(2.0), pf))
    print("  MaxHoldingsRule        allows (already-held symbol, same portfolio) ->",
          mh.evaluate("510300", _row(4.0), pf))

    bl = oxq.rules.BlacklistRule(symbols={"510300"})
    print("  BlacklistRule          blocks (symbol in blacklist)         ->",
          bl.evaluate("510300", _row(4.0), _held(4.0)))

    # RebalanceFrequencyRule/CalendarRebalanceRule key off row.name (the bar's
    # timestamp) and hold internal counters -- they need consecutive calls
    # across dates to show the allow -> block -> allow-again cycle.
    reb = oxq.rules.RebalanceFrequencyRule(interval_days=2)
    pf = _held(entry_price=4.0)
    r1 = reb.evaluate("510300", _row(4.0, date="2024-01-01"), pf)   # first bar ever: always allowed
    r2 = reb.evaluate("510300", _row(4.0, date="2024-01-02"), pf)   # 1 day since rebalance < 2: blocked
    r3 = reb.evaluate("510300", _row(4.0, date="2024-01-03"), pf)   # 2 days since rebalance: allowed again
    print(f"  RebalanceFrequencyRule day1={r1.hold} day2(blocked)={r2.hold} day3(allowed)={r3.hold}")

    cal = oxq.rules.CalendarRebalanceRule(schedule="month_start")
    pf = _held(entry_price=4.0)
    c1 = cal.evaluate("510300", _row(4.0, date="2024-01-01"), pf)   # first bar of the month: allowed
    c2 = cal.evaluate("510300", _row(4.0, date="2024-01-15"), pf)   # mid-month: blocked
    print(f"  CalendarRebalanceRule  month-start={c1.hold} mid-month(blocked)={c2.hold}")

    covered = {
        "StopLossRule", "TakeProfitRule", "TrailingStopRule", "ExitRule",
        "MaxDrawdownRisk", "DailyLossLimitRisk", "MaxHoldingsRule",
        "BlacklistRule", "RebalanceFrequencyRule", "CalendarRebalanceRule",
    }
    registered = set(list_rules())
    assert covered == registered, f"scenario/registry mismatch: {covered ^ registered}"


# ===========================================================================
# Section 4: Portfolio Optimizers (6)
# ===========================================================================
#
# All six share one call shape: optimize(signals: dict[str, DataFrame],
# indicators: dict[str, DataFrame]) -> dict[str, float]. What differs is the
# constructor. Note the registry name ("Kelly") is not the class name
# ("KellyOptimizer") and Kelly's real params are column names, not the
# scalar win_rate/win_loss_ratio floats an earlier hand-written catalog
# guessed at -- this was the concrete motivation for this whole file.


def run_optimizers() -> None:
    print()
    print("=" * 70)
    print(f"SECTION 4: PORTFOLIO OPTIMIZERS ({len(list_portfolio_optimizers())})")
    print("=" * 70)

    signals = {
        "510300": pd.DataFrame({"entry_gate": [True]}),
        "510050": pd.DataFrame({"entry_gate": [True]}),
        "510500": pd.DataFrame({"entry_gate": [True]}),
    }
    indicators = {
        "510300": pd.DataFrame({"rps_60": [80.0], "vol_20": [0.010], "win_rate": [0.55], "avg_win": [0.02], "avg_loss": [0.01]}),
        "510050": pd.DataFrame({"rps_60": [60.0], "vol_20": [0.015], "win_rate": [0.50], "avg_win": [0.015], "avg_loss": [0.012]}),
        "510500": pd.DataFrame({"rps_60": [90.0], "vol_20": [0.020], "win_rate": [0.60], "avg_win": [0.025], "avg_loss": [0.015]}),
    }

    demos = {
        "EqualWeight": lambda: opt_mod.EqualWeightOptimizer().optimize(signals, indicators),
        "TopNRanking": lambda: opt_mod.TopNRankingOptimizer(
            score_col="rps_60", n=2, pre_filter_signal="entry_gate", weighting="score"
        ).optimize(signals, indicators),
        "PctEquity": lambda: opt_mod.PctEquityOptimizer(pct=0.2).optimize(signals, indicators),
        "RiskParity": lambda: opt_mod.RiskParityOptimizer(volatility_col="vol_20").optimize(signals, indicators),
        "Kelly": lambda: opt_mod.KellyOptimizer(
            win_rate_col="win_rate", avg_win_col="avg_win", avg_loss_col="avg_loss", fraction=0.5
        ).optimize(signals, indicators),
        "SignalToPosition": lambda: opt_mod.SignalToPositionOptimizer(
            signal="entry_gate", buy_weight=1.0, sell_weight=0.0
        ).optimize({"510300": pd.DataFrame({"entry_gate": ["BUY"]})}, indicators),
    }

    covered = set(demos)
    registered = set(list_portfolio_optimizers())
    assert covered == registered, f"scenario/registry mismatch: {covered ^ registered}"

    for name in sorted(demos):
        weights = demos[name]()
        print(f"  {name:<18} -> {weights}")


if __name__ == "__main__":
    run_indicators()
    run_signals()
    run_rules()
    run_optimizers()
    print()
    print("=" * 70)
    print("All 72 registered components ran without an uncaught exception.")
    print("=" * 70)
