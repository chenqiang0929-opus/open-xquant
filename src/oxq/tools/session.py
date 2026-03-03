"""Session store — holds mutable state across tool calls.

State is persisted to a temp file so it survives MCP server restarts
(each user message in agent_demo spawns a new MCP subprocess).
"""

from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oxq.core.strategy import Strategy
    from oxq.portfolio.analytics import RunResult

logger = logging.getLogger(__name__)

_SESSION_FILE = Path(tempfile.gettempdir()) / "oxq_mcp_session.pkl"

_strategies: dict[str, Strategy] = {}
_run_results: dict[str, RunResult] = {}


def _save() -> None:
    """Persist session state to disk."""
    try:
        with open(_SESSION_FILE, "wb") as f:
            pickle.dump(
                {"strategies": _strategies, "run_results": _run_results}, f,
            )
    except Exception:
        logger.warning("Failed to save session state", exc_info=True)


def _load() -> None:
    """Load session state from disk (if available)."""
    if not _SESSION_FILE.exists():
        return
    try:
        with open(_SESSION_FILE, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        _strategies.update(data.get("strategies", {}))
        _run_results.update(data.get("run_results", {}))
    except Exception:
        logger.warning("Failed to load session state", exc_info=True)


def clear() -> None:
    """Reset session state (for testing and Clear Chat)."""
    _strategies.clear()
    _run_results.clear()
    _SESSION_FILE.unlink(missing_ok=True)


# Auto-load persisted state when the MCP server process starts.
_load()
