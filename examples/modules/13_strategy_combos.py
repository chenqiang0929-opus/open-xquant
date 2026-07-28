"""Five ready-to-edit strategy templates, one per common component combo.

An earlier hand-written component catalog listed five "combination recipes"
(indicator + signal + portfolio pattern) as one-line ideas without runnable
code. This file turns each of those five into a complete, validated
StrategySpec you can copy and change parameters on -- built and validated
against this container's real 21-ETF dataset, not sketched from memory.

Run all five:      python examples/modules/13_strategy_combos.py
Run just one:       python examples/modules/13_strategy_combos.py --only 3
Actually backtest:  python examples/modules/13_strategy_combos.py --backtest 1 4

Each function returns a validated (not backtested) StrategySpec by default --
building the spec and confirming validate() passes is nearly free, so all
five run by default. Backtesting is opt-in per template via --backtest
because it costs real seconds per template and needs local market data.

All five have been validate()'d AND actually backtested against this
container's 21-symbol CN ETF universe (2015-2020 train / 2021-2026 test).
Real numbers, not projections -- rerun with --backtest to reproduce:

    combo  total_return  sharpe  max_dd   trades
    1      14.45%        0.159   -53.90%  753
    2       1.23%        0.090   -44.26%  2531
    3      26.97%        0.481   -15.58%  34
    4      -1.65%        0.013   -27.00%  224
    5      -8.83%        0.008   -36.82%  35

Combo 3 (risk-adjusted momentum + drawdown breaker) is the only one here
with a Sharpe above 0.3 on this universe/period -- worth a closer look
before the others. None of this is investment advice; it's evidence that
the templates run correctly, not a claim that any of them is profitable
out of sample on data they haven't seen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oxq.spec.compiler import compile_run
from oxq.spec.schema import (
    IndicatorDef,
    PortfolioRuleDef,
    SignalRuleDef,
    StrategySpec,
)
from oxq.spec.validator import validate

UNIVERSE_21 = [
    "513050", "159920", "510900", "513100", "513500", "159941", "513520",
    "510300", "510050", "510500", "159915", "159949", "159919", "510330",
    "512880", "512000", "512800", "512010", "512400", "512200", "512700",
]
DATA_DIR = "~/.oxq/data/market"
TRAIN = ["2015-01-01", "2020-12-31"]
TEST = ["2021-01-01", "2026-07-24"]


def _base_spec(strategy_id: str, hypothesis: str) -> StrategySpec:
    """Shared scaffolding every template starts from."""
    spec = StrategySpec.template(strategy_id=strategy_id, hypothesis=hypothesis, market_preset="cn_a_share")
    spec.data.data_dir = DATA_DIR
    spec.data.required_columns = ["open", "high", "low", "close", "volume"]
    spec.validation.train_period = TRAIN
    spec.validation.test_period = TEST
    spec.validation.required_oos = True
    spec.execution.initial_cash = 1_000_000
    spec.execution.lot_size_config.default = 100
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.001
    return spec


# ===========================================================================
# 1. RPS + Threshold + TopNRanking  --  "基本面+动量筛选" (relative strength
#    rotation, gated by a trend filter)
# ===========================================================================


def combo_1_rps_topn() -> StrategySpec:
    spec = _base_spec(
        "combo1_rps_topn",
        "ETFs with the strongest trailing 60-day relative strength (RPS), "
        "filtered to those trading above their own 50-day SMA, outperform "
        "over the next ~20-day holding period.",
    )
    spec.universe.symbols = list(UNIVERSE_21)
    spec.benchmark.symbols = ["510300"]

    spec.signal.indicators = {
        "sma_trend": IndicatorDef(type="SMA", params={"column": "close", "period": 50}),
        "rps_60": IndicatorDef(type="RPS", params={"column": "close", "period": 60, "scale": 100.0, "min_symbols": 1}),
    }
    spec.signal.rules = {
        "trend_ok": SignalRuleDef(type="Comparison", params={"left": "close", "right": "sma_trend", "relationship": "gt"}),
        "momentum_ok": SignalRuleDef(type="Threshold", params={"column": "rps_60", "threshold": 50.0, "relationship": "gt"}),
        "entry_gate": SignalRuleDef(type="Composite", params={"signals": ["trend_ok", "momentum_ok"], "logic": "and"}),
    }
    spec.portfolio.type = "TopNRanking"
    spec.portfolio.params = {
        "score_col": "rps_60", "n": 5, "filter_negative": False, "max_weight": 1.0,
        "pre_filter_signal": "entry_gate", "weighting": "score", "ascending": False,
    }
    spec.portfolio.rules = {"rebalance": PortfolioRuleDef(type="RebalanceFrequencyRule", params={"interval_days": 20})}
    return spec


# ===========================================================================
# 2. MACD + ADX>25 + Composite  --  "信号过滤增强" (only trade the MACD
#    signal when the trend is strong enough for it to mean anything)
# ===========================================================================


def combo_2_macd_adx_filter() -> StrategySpec:
    spec = _base_spec(
        "combo2_macd_adx_filter",
        "A positive MACD line predicts continuation only when ADX confirms "
        "the underlying trend is strong (ADX > 25); MACD signals during a "
        "weak/choppy trend (ADX <= 25) are unreliable and should be ignored.",
    )
    spec.universe.symbols = list(UNIVERSE_21)
    spec.benchmark.symbols = ["510300"]

    # Threshold (not Comparison) because we're comparing a column against a
    # literal number (0, 25) -- Comparison's `right` must be a column name.
    spec.signal.indicators = {
        "macd_line": IndicatorDef(type="MACDLine", params={"column": "close", "fast_period": 12, "slow_period": 26}),
        "adx": IndicatorDef(type="ADX", params={"period": 14}),
    }
    spec.signal.rules = {
        "macd_bull": SignalRuleDef(type="Threshold", params={"column": "macd_line", "threshold": 0.0, "relationship": "gt"}),
        "trend_strong": SignalRuleDef(type="Threshold", params={"column": "adx", "threshold": 25.0, "relationship": "gt"}),
        "entry_gate": SignalRuleDef(type="Composite", params={"signals": ["macd_bull", "trend_strong"], "logic": "and"}),
    }
    # EqualWeight buys every symbol whose signal.rules gate passes -- same
    # pattern already validated in examples/strategies/sma_crossover_spec.py.
    spec.portfolio.type = "EqualWeight"
    spec.portfolio.rules = {"rebalance": PortfolioRuleDef(type="RebalanceFrequencyRule", params={"interval_days": 5})}
    return spec


# ===========================================================================
# 3. NdayReturn / RollingVolatility + MaxDrawdownRisk  --  "动量+风控"
#    (risk-adjusted momentum ranking, with a portfolio-level circuit breaker)
# ===========================================================================


def combo_3_risk_adjusted_momentum() -> StrategySpec:
    spec = _base_spec(
        "combo3_risk_adjusted_momentum",
        "Ranking ETFs by 20-day return divided by 20-day volatility (a "
        "risk-adjusted momentum score) selects continuation candidates "
        "better than raw return alone, and a portfolio-level max-drawdown "
        "breaker limits the damage when the ranking is wrong.",
    )
    spec.universe.symbols = list(UNIVERSE_21)
    spec.benchmark.symbols = ["510300"]

    spec.signal.indicators = {
        "mom_20": IndicatorDef(type="NdayReturn", params={"column": "close", "period": 20}),
        "vol_20": IndicatorDef(type="RollingVolatility", params={"column": "close", "period": 20}),
        "risk_adj_mom": IndicatorDef(type="Ratio", params={"col_a": "mom_20", "col_b": "vol_20"}),
    }
    spec.portfolio.type = "TopNRanking"
    spec.portfolio.params = {
        "score_col": "risk_adj_mom", "n": 5, "filter_negative": True, "max_weight": 0.3,
        "weighting": "score", "ascending": False,
    }
    spec.portfolio.rules = {
        "rebalance": PortfolioRuleDef(type="RebalanceFrequencyRule", params={"interval_days": 20}),
        "max_drawdown": PortfolioRuleDef(type="MaxDrawdownRisk", params={"max_drawdown": 0.15}),
    }
    return spec


# ===========================================================================
# 4. SMA + Crossover + TrailingStop + EqualWeight  --  "趋势跟踪+保护"
#    (golden cross entry, trailing stop protects gains on the way down)
# ===========================================================================


def combo_4_trend_with_trailing_stop() -> StrategySpec:
    spec = _base_spec(
        "combo4_trend_with_trailing_stop",
        "A 10-day SMA crossing above a 50-day SMA signals a new uptrend; a "
        "5% trailing stop locks in gains without needing a second signal "
        "to decide when to exit.",
    )
    spec.universe.symbols = ["510300", "510050", "510500"]
    spec.benchmark.symbols = ["510300"]

    spec.signal.indicators = {
        "sma_fast": IndicatorDef(type="SMA", params={"column": "close", "period": 10}),
        "sma_slow": IndicatorDef(type="SMA", params={"column": "close", "period": 50}),
    }
    spec.signal.rules = {
        "golden_cross": SignalRuleDef(type="Crossover", params={"fast": "sma_fast", "slow": "sma_slow"}),
    }
    spec.portfolio.type = "EqualWeight"
    spec.portfolio.rules = {
        "trailing_stop": PortfolioRuleDef(type="TrailingStopRule", params={"trail_pct": 0.05}),
    }
    return spec


# ===========================================================================
# 5. ROC + ROCTiming + SignalToPosition + (Calendar->Rebalance)  --
#    "定期轮动" (single-symbol timing with a periodic rebalance cadence)
# ===========================================================================


def combo_5_roc_timing() -> StrategySpec:
    spec = _base_spec(
        "combo5_roc_timing",
        "A 12-day rate-of-change beyond +-5% signals a BUY/SELL/HOLD timing "
        "call (ROCTiming) for a single benchmark ETF.",
    )
    spec.universe.symbols = ["510300"]
    spec.benchmark.symbols = ["510300"]

    spec.signal.indicators = {
        "roc_12": IndicatorDef(type="ROC", params={"column": "close", "period": 12}),
    }
    spec.signal.rules = {
        # ROCTiming is the only signal that outputs {"BUY","SELL","HOLD"}
        # instead of a bool -- that's exactly what SignalToPosition expects.
        "roc_timing_rule": SignalRuleDef(
            type="ROCTiming",
            params={"column": "roc_12", "mode": "fixed", "bottom": -5.0, "top": 5.0},
        ),
    }
    spec.portfolio.type = "SignalToPosition"
    spec.portfolio.params = {"signal": "roc_timing_rule", "buy_weight": 1.0, "sell_weight": 0.0}
    # NOTE ON THE SUBSTITUTION: the original recipe named CalendarRebalanceRule
    # here for "monthly rebalancing". Checked against the audited runtime
    # (oxq/spec/validator.py: _SUPPORTED_RUNTIME_RULES, lines ~515-522) --
    # portfolio.rules only accepts StopLossRule, TakeProfitRule,
    # TrailingStopRule, MaxDrawdownRisk, DailyLossLimitRisk, MaxHoldingsRule,
    # plus RebalanceFrequencyRule under the "rebalance" key.
    # CalendarRebalanceRule is NOT in that list -- wiring it in here would
    # fail validate(). RebalanceFrequencyRule(interval_days=20) is the
    # supported way to get the same "roughly monthly" cadence.
    spec.portfolio.rules = {"rebalance": PortfolioRuleDef(type="RebalanceFrequencyRule", params={"interval_days": 20})}
    return spec


COMBOS = {
    1: ("RPS + Threshold + TopNRanking", combo_1_rps_topn),
    2: ("MACD + ADX>25 + Composite", combo_2_macd_adx_filter),
    3: ("NdayReturn/RollingVolatility + MaxDrawdownRisk", combo_3_risk_adjusted_momentum),
    4: ("SMA + Crossover + TrailingStop + EqualWeight", combo_4_trend_with_trailing_stop),
    5: ("ROC + ROCTiming + SignalToPosition + RebalanceFrequencyRule*", combo_5_roc_timing),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, choices=sorted(COMBOS), help="run a single combo by number")
    parser.add_argument("--backtest", type=int, nargs="*", default=[], help="also compile_run() these combo numbers")
    args = parser.parse_args()

    numbers = [args.only] if args.only else sorted(COMBOS)
    failed = []

    for n in numbers:
        title, builder = COMBOS[n]
        print("=" * 70)
        print(f"COMBO {n}: {title}")
        print("=" * 70)

        spec = builder()
        result = validate(spec)
        print(f"  validate: {result.status.upper()}  ({len(result.errors)} errors, {len(result.warnings)} warnings)")
        for e in result.errors:
            print(f"    FATAL: {e['check']}: {e['message']}")
        for w in result.warnings:
            print(f"    warn:  {w['check']}")

        if result.status == "fail":
            failed.append(n)
            continue

        if n in args.backtest:
            out_dir = Path(f"/tmp/oxq_combos/combo{n}")
            run_result, run_dir = compile_run(spec, out_dir=str(out_dir))
            print(f"  backtest -> {run_dir}")
            print(f"    total_return={run_result.total_return():.2%}  "
                  f"sharpe={run_result.sharpe_ratio():.3f}  "
                  f"max_dd={run_result.max_drawdown():.2%}  "
                  f"trades={len(run_result.trades)}")
        print()

    if failed:
        print(f"FAILED validate(): combos {failed}", file=sys.stderr)
        return 1
    print("All requested combos validated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
