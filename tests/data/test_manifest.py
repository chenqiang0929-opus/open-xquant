from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from oxq.data.manifest import ManifestVerification, read_manifest, write_manifest


@pytest.fixture()
def sample_parquet(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]},
        index=pd.DatetimeIndex(["2024-01-02"], name="date"),
    )
    path = tmp_path / "TEST.parquet"
    df.to_parquet(path)
    return path


class TestWriteAndReadManifest:
    def test_roundtrip(self, sample_parquet: Path) -> None:
        manifest_path = write_manifest(
            parquet_path=sample_parquet,
            symbol="TEST",
            provider="yfinance",
            start="2024-01-01",
            end="2024-12-31",
            rows=1,
            extra={"auto_adjust": True},
        )
        assert manifest_path == sample_parquet.parent / "TEST.manifest.json"
        data = read_manifest(manifest_path)
        assert data is not None
        assert data["symbol"] == "TEST"
        assert data["provider"] == "yfinance"
        assert data["start"] == "2024-01-01"
        assert data["end"] == "2024-12-31"
        assert data["rows"] == 1
        assert data["extra"] == {"auto_adjust": True}
        assert "sha256" in data
        assert "created_at" in data

    def test_sha256_correctness(self, sample_parquet: Path) -> None:
        write_manifest(
            parquet_path=sample_parquet,
            symbol="TEST",
            provider="yfinance",
            start="2024-01-01",
            end="2024-12-31",
            rows=1,
        )
        manifest_path = sample_parquet.parent / "TEST.manifest.json"
        data = read_manifest(manifest_path)
        expected_sha = hashlib.sha256(sample_parquet.read_bytes()).hexdigest()
        assert data["sha256"] == expected_sha

    def test_extra_none_omitted(self, sample_parquet: Path) -> None:
        write_manifest(
            parquet_path=sample_parquet,
            symbol="TEST",
            provider="yfinance",
            start="2024-01-01",
            end="2024-12-31",
            rows=1,
        )
        data = read_manifest(sample_parquet.parent / "TEST.manifest.json")
        assert data["extra"] is None

    def test_read_missing_returns_none(self, tmp_path: Path) -> None:
        assert read_manifest(tmp_path / "nope.manifest.json") is None

    def test_read_invalid_json_returns_none(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.manifest.json"
        bad.write_text("not json{{{")
        assert read_manifest(bad) is None
