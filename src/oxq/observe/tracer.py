"""Tracer — component-level execution tracing for the engine pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TraceSpan:
    """A single component execution record.

    Captures inputs, output summary, timing, and status for one
    component (indicator, signal, or rule) during an engine run.
    """

    trace_id: str
    span_id: str
    parent_id: str | None
    component: str
    inputs: dict[str, Any]
    output_summary: dict[str, Any]
    started_at: str
    ended_at: str
    duration_ms: float
    status: str  # "ok" | "error" | "skipped"
    error: str | None
