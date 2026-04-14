"""Tests for universe_set(type='index') and universe_list_indexes() tools."""

from unittest.mock import patch

from oxq.tools.universe import universe_list_indexes, universe_set
from oxq.universe.index import INDEX_REGISTRY


def _patch_fetch(key: str, return_value: list[str]):
    """Patch the fetch_fn stored in INDEX_REGISTRY for a given key."""
    return patch.dict(
        INDEX_REGISTRY, {key: {**INDEX_REGISTRY[key], "fetch_fn": lambda code: return_value}}
    )


def test_universe_list_indexes_returns_builtin() -> None:
    """universe_list_indexes must return real indexes, not empty."""
    result = universe_list_indexes()
    assert len(result["indexes"]) >= 4
    keys = [idx["key"] for idx in result["indexes"]]
    assert "csi300" in keys
    assert "sse50" in keys
    # Phase 2 note should be gone
    assert "note" not in result or "Phase 2" not in result.get("note", "")


def test_universe_set_type_index() -> None:
    """universe_set(type='index') returns constituent symbols."""

    fake_symbols = ["600519", "000858", "000001"]

    with _patch_fetch("csi300", fake_symbols):
        result = universe_set(type="index", code="csi300")

    assert result["symbols"] == fake_symbols
    assert result["count"] == 3
    assert "csi300" in result["source"]


def test_universe_set_type_index_unknown_code() -> None:
    """universe_set(type='index') with unknown code returns error."""
    result = universe_set(type="index", code="nonexistent")
    assert "error" in result


def test_universe_set_type_index_uses_code_param() -> None:
    """universe_set uses the code parameter to select index."""

    with _patch_fetch("sse50", ["000001"]):
        result = universe_set(type="index", code="sse50")

    assert result["count"] == 1
    assert "sse50" in result["source"]
