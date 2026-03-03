"""Session store — holds mutable state across tool calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oxq.core.strategy import Strategy
    from oxq.portfolio.analytics import RunResult

_strategies: dict[str, Strategy] = {}
_run_results: dict[str, RunResult] = {}


def clear() -> None:
    """Reset session state (for testing)."""
    _strategies.clear()
    _run_results.clear()
