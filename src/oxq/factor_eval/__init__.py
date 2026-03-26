"""Factor evaluation — metrics and utilities for assessing indicator predictive power."""

from oxq.factor_eval.bias import detect_lookahead_bias
from oxq.factor_eval.bundle import AlignmentReport, FactorBundle, create_bundle
from oxq.factor_eval.cash_period import compute_cash_period_value
from oxq.factor_eval.comparison import compute_asset_comparison
from oxq.factor_eval.conditional import compute_conditional_analysis
from oxq.factor_eval.decay_curve import compute_decay_curve
from oxq.factor_eval.hit_rate import compute_hit_rate
from oxq.factor_eval.metrics import (
    compute_decay,
    compute_ic,
    compute_icir,
    compute_rank_ic,
    compute_ts_ic,
    compute_turnover,
)
from oxq.factor_eval.preprocessing import (
    apply_t1_offset,
    mark_limit_days,
    mark_suspension_days,
)
from oxq.factor_eval.profit_loss import compute_profit_loss_ratio
from oxq.factor_eval.returns import compute_forward_returns

__all__ = [
    # Data layer
    "AlignmentReport",
    "FactorBundle",
    "create_bundle",
    # Preprocessing
    "apply_t1_offset",
    "mark_limit_days",
    "mark_suspension_days",
    # Returns
    "compute_forward_returns",
    # Metrics (existing)
    "compute_decay",
    "compute_ic",
    "compute_icir",
    "compute_rank_ic",
    "compute_ts_ic",
    "compute_turnover",
    # P0 metrics
    "compute_decay_curve",
    "compute_hit_rate",
    "detect_lookahead_bias",
    # P1 metrics
    "compute_profit_loss_ratio",
    "compute_cash_period_value",
    "compute_conditional_analysis",
    "compute_asset_comparison",
    # Output (lazy import — requires matplotlib)
    "generate_tearsheet",
]


def __getattr__(name: str):
    if name == "generate_tearsheet":
        from oxq.factor_eval.tearsheet import generate_tearsheet

        return generate_tearsheet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
