"""Tests for Tracer — component-level execution tracing."""

from __future__ import annotations

import pytest


class TestTraceSpan:
    def test_frozen(self) -> None:
        from oxq.observe.tracer import TraceSpan

        span = TraceSpan(
            trace_id="t1",
            span_id="s1",
            parent_id=None,
            component="indicator:sma_fast",
            inputs={"column": "close", "period": 10},
            output_summary={"rows": 252, "non_null": 243},
            started_at="2026-01-01T00:00:00",
            ended_at="2026-01-01T00:00:01",
            duration_ms=1000.0,
            status="ok",
            error=None,
        )
        assert span.component == "indicator:sma_fast"
        assert span.status == "ok"
        with pytest.raises(AttributeError):
            span.status = "error"

    def test_error_span(self) -> None:
        from oxq.observe.tracer import TraceSpan

        span = TraceSpan(
            trace_id="t1",
            span_id="s2",
            parent_id="s1",
            component="indicator:broken",
            inputs={},
            output_summary={},
            started_at="2026-01-01T00:00:00",
            ended_at="2026-01-01T00:00:01",
            duration_ms=500.0,
            status="error",
            error="KeyError: 'missing_col'",
        )
        assert span.status == "error"
        assert "missing_col" in span.error
