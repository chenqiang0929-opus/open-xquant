"""Strategy definition — declarative composition of pipeline components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from oxq.core.types import PortfolioOptimizer, Signal
from oxq.universe.base import UniverseProvider


@dataclass
class Strategy:
    """A complete strategy definition.

    Composes Universe, Signal, and PortfolioOptimizer into a declarative
    pipeline that produces target portfolios.

    Strategy is purely functional: given the same market data, it always
    produces the same target portfolio. State and side effects live in
    the Engine/Trade layer.
    """

    name: str
    universe: UniverseProvider
    signals: dict[str, tuple[Signal, dict[str, Any]]]
    portfolio: PortfolioOptimizer
    # Architecture metadata
    hypothesis: str = ""
    objectives: dict[str, dict[str, float]] = field(default_factory=dict)
    benchmarks: list[str] = field(default_factory=list)
