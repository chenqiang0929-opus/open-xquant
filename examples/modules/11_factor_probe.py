"""Probe any built-in indicator as a cross-sectional factor.

This is the fast lane for the question "does this factor predict anything?".
It skips strategy_spec.yaml, the backtest engine, and the multi-Agent audit
workflow entirely, and answers in seconds using the SDK directly.

Examples
--------
List every indicator the registry knows about::

    python examples/research/factor_probe.py --list

Probe a momentum oscillator over a handful of symbols::

    python examples/research/factor_probe.py \\
        --indicator RSI --symbols 510300 510050 510500 \\
        --start 2022-01-01 --end 2024-12-31

Pass indicator-specific parameters as JSON::

    python examples/research/factor_probe.py \\
        --indicator MACDLine --params '{"fast_period": 12, "slow_period": 26}' \\
        --symbols SPY QQQ IWM

What this does NOT tell you: whether trading the factor makes money. Position
sizing, rebalance timing, and transaction costs all live in the backtest layer.
A factor with a promising IC can still lose money once costs are applied.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from typing import Any

import pandas as pd

import oxq
from oxq.core.registry import get_indicator_metadata, list_indicators
from oxq.data.market import LocalMarketDataProvider
from oxq.factor_eval.metrics import (
    compute_decay,
    compute_ic,
    compute_icir,
    compute_rank_ic,
    compute_turnover,
)

# Below this many symbols, cross-sectional IC is computed over so few
# observations per date that it carries no evidential weight.
MIN_SYMBOLS_FOR_IC = 10
CAUTION_SYMBOLS_FOR_IC = 30


def resolve_indicator(name: str):
    """Resolve a registry indicator name to its class, or exit with candidates."""
    indicator_cls = getattr(oxq.indicators, name, None)
    if indicator_cls is not None:
        return indicator_cls

    known = sorted(list_indicators())
    lowered = name.lower()
    close_matches = [n for n in known if lowered in n.lower()]
    print(f"Unknown indicator: {name!r}", file=sys.stderr)
    if close_matches:
        print(f"Did you mean: {', '.join(close_matches)}", file=sys.stderr)
    print(f"\n{len(known)} indicators available:", file=sys.stderr)
    print("  " + ", ".join(known), file=sys.stderr)
    raise SystemExit(2)


def filter_params(indicator_cls, params: dict[str, Any], method: str = "compute") -> dict[str, Any]:
    """Keep only params the indicator's compute signature accepts.

    Indicator signatures are not uniform: SMA takes (column, period), ATR takes
    only (period), Ratio takes (col_a, col_b). Passing an unsupported keyword
    would raise, so drop it here and say so rather than failing late.
    """
    signature = inspect.signature(getattr(indicator_cls, method))
    accepted = set(signature.parameters) - {"self", "mktdata"}
    kept = {k: v for k, v in params.items() if k in accepted}
    dropped = sorted(set(params) - set(kept))
    if dropped:
        print(f"  note: ignoring params not accepted by {indicator_cls.__name__}.{method}: {', '.join(dropped)}")
        print(f"        accepted here: {', '.join(sorted(accepted)) or '(none)'}")
    return kept


def load_bars(
    market: LocalMarketDataProvider, symbols: list[str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Load bars per symbol, reporting symbols that yielded no usable data."""
    bars: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in symbols:
        try:
            frame = market.get_bars(symbol, start, end)
        except Exception as exc:  # noqa: BLE001 - surface provider errors verbatim
            missing.append(f"{symbol} ({type(exc).__name__}: {exc})")
            continue
        if frame is None or frame.empty:
            missing.append(f"{symbol} (no rows in {start}..{end})")
            continue
        bars[symbol] = frame

    if missing:
        print(f"  skipped {len(missing)} symbol(s):")
        for entry in missing:
            print(f"    - {entry}")
    if not bars:
        print(
            "\nNo usable data. Check --data-dir points at a directory of "
            "<SYMBOL>.parquet files and that the date range overlaps them.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return bars


def build_factor_frame(
    indicator_cls, bars: dict[str, pd.DataFrame], params: dict[str, Any]
) -> pd.DataFrame:
    """Compute the factor for every symbol and assemble a dates x symbols frame.

    RPS is the only built-in indicator that ranks symbols against each other, so
    it needs the whole panel at once; every other indicator is per-symbol.
    """
    indicator = indicator_cls()

    if hasattr(indicator_cls, "compute_cross_section"):
        print(f"  {indicator_cls.__name__} is cross-sectional -> using compute_cross_section()")
        kwargs = filter_params(indicator_cls, params, method="compute_cross_section")
        series_by_symbol = indicator.compute_cross_section(bars, **kwargs)
        return pd.DataFrame(series_by_symbol)

    kwargs = filter_params(indicator_cls, params)
    return pd.DataFrame({symbol: indicator.compute(frame, **kwargs) for symbol, frame in bars.items()})


def align(factor: pd.DataFrame, forward_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to dates where the factor and forward return exist."""
    common = factor.dropna(how="all").index.intersection(forward_returns.dropna(how="all").index)
    return factor.loc[common], forward_returns.loc[common]


def report_horizon(factor: pd.DataFrame, prices: pd.DataFrame, horizon: int) -> None:
    """Print IC / Rank IC / ICIR for one forward-return horizon."""
    # shift(-h) puts the *future* return on today's row, so today's factor is
    # never scored against a return that already includes today.
    forward_returns = prices.pct_change(horizon).shift(-horizon)
    aligned_factor, aligned_returns = align(factor, forward_returns)

    if aligned_factor.empty:
        print(f"  horizon {horizon:>3}d: no overlapping dates, skipped")
        return

    ic = compute_ic(factor=aligned_factor, forward_returns=aligned_returns)
    rank_ic = compute_rank_ic(factor=aligned_factor, forward_returns=aligned_returns)
    icir = compute_icir(ic["mean"], ic["std"])

    if pd.isna(ic["mean"]):
        print(
            f"  horizon {horizon:>3}d: undefined -- a correlation needs at least "
            f"3 symbols per date, and this run has {aligned_factor.shape[1]}"
        )
        return

    print(
        f"  horizon {horizon:>3}d: "
        f"IC {ic['mean']:+.4f} (sd {ic['std']:.4f})  "
        f"RankIC {rank_ic['mean']:+.4f}  "
        f"ICIR {icir:+.4f}  "
        f"n={len(aligned_factor)} dates"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a built-in indicator as a factor (IC / Rank IC / decay / turnover).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="list available indicators and exit")
    parser.add_argument("--indicator", help="registry indicator name, e.g. RSI, SMA, MACDLine, RPS")
    parser.add_argument("--params", default="{}", help='indicator params as JSON, e.g. \'{"period": 20}\'')
    parser.add_argument("--symbols", nargs="+", default=[], help="symbols to probe (space separated)")
    parser.add_argument("--data-dir", default="~/.oxq/data/market", help="directory of <SYMBOL>.parquet files")
    parser.add_argument("--start", default="2020-01-01", help="start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="end date (YYYY-MM-DD)")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 5, 20],
        help="forward-return horizons in trading days",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        names = sorted(list_indicators())
        print(f"{len(names)} indicators available:\n")
        for name in names:
            meta = get_indicator_metadata(name) or {}
            category = meta.get("category", "")
            description = meta.get("description", "")
            print(f"  {name:<22} {category:<14} {description}")
        return 0

    if not args.indicator or not args.symbols:
        print("Both --indicator and --symbols are required (or use --list).", file=sys.stderr)
        return 2

    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as exc:
        print(f"--params is not valid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(params, dict):
        print("--params must be a JSON object, e.g. '{\"period\": 20}'", file=sys.stderr)
        return 2

    indicator_cls = resolve_indicator(args.indicator)
    meta = get_indicator_metadata(args.indicator) or {}

    print(f"Indicator: {args.indicator}  [{meta.get('category', 'uncategorized')}]")
    if meta.get("description"):
        print(f"  {meta['description']}")
    print(f"Signature: {inspect.signature(indicator_cls.compute)}")
    print(f"Window:    {args.start} .. {args.end}")
    print()

    print(f"Loading {len(args.symbols)} symbol(s) from {args.data_dir}")
    market = LocalMarketDataProvider(data_dir=args.data_dir)
    bars = load_bars(market, args.symbols, args.start, args.end)
    print(f"  loaded {len(bars)} symbol(s)")

    factor = build_factor_frame(indicator_cls, bars, params)
    prices = pd.DataFrame({symbol: frame["close"] for symbol, frame in bars.items()})
    factor = factor.reindex(columns=prices.columns)

    usable = factor.dropna(how="all")
    print(f"\nFactor values: {len(usable)} dated rows x {factor.shape[1]} symbols")
    if usable.empty:
        print(
            "  factor is entirely NaN -- the lookback window is probably longer "
            "than the data range. Widen --start/--end or shorten the period.",
            file=sys.stderr,
        )
        return 1
    print(usable.tail(3).to_string())

    print("\nPredictive power (factor at t vs return over t+1..t+h):")
    if len(bars) < MIN_SYMBOLS_FOR_IC:
        print(
            f"  WARNING: only {len(bars)} symbols. Cross-sectional IC over fewer "
            f"than {MIN_SYMBOLS_FOR_IC} names is noise, not evidence -- treat the "
            f"numbers below as a smoke test. Use a time-series evaluation "
            f"(oxq.factor_eval.tearsheet) for small universes."
        )
    elif len(bars) < CAUTION_SYMBOLS_FOR_IC:
        print(
            f"  NOTE: {len(bars)} symbols is on the thin side; IC is usable but "
            f"more defensible above {CAUTION_SYMBOLS_FOR_IC}."
        )

    for horizon in args.horizons:
        report_horizon(factor, prices, horizon)

    decay = compute_decay(factor=factor, prices=prices, horizons=args.horizons)
    pairs = ", ".join(
        f"{h}d {v:+.4f}" for h, v in zip(decay["horizons"], decay["ic_values"], strict=True)
    )
    print(f"\nIC decay:  {pairs}")
    print(f"Turnover:  {compute_turnover(factor):.4f}  (rank churn per rebalance; weigh against costs)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
