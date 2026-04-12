"""Demo: download real stock data and verify manifests."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from oxq.data import YFinanceDownloader, verify_manifest, read_manifest
from oxq.tools.data import data_generate_mock


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def show_manifest(parquet_path: Path) -> None:
    manifest_path = parquet_path.parent / f"{parquet_path.stem}.manifest.json"
    data = read_manifest(manifest_path)
    if data:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print("  (no manifest)")


def main() -> None:
    with TemporaryDirectory() as tmp:
        dest = Path(tmp)

        # --- 1. Real data via yfinance ---
        section("1. Download real stocks via YFinance")
        dl = YFinanceDownloader(auto_adjust=True)
        symbols = ["AAPL", "MSFT", "GOOGL"]
        for sym in symbols:
            path = dl.download(sym, "2024-01-01", "2024-06-30", dest_dir=dest)
            print(f"\n[{sym}] parquet: {path.name}")
            show_manifest(path)

        # --- 2. Verify real data ---
        section("2. Verify real data manifests")
        for sym in symbols:
            result = verify_manifest(dest / f"{sym}.parquet")
            print(f"  {sym}: status={result.status}, provider={result.provider}, detail={result.detail}")

        # --- 3. Mock data ---
        section("3. Generate mock data")
        mock_dir = dest / "mock"
        mock_dir.mkdir()
        data_generate_mock(
            symbols=["FAKE1", "FAKE2"],
            start="2024-01-01",
            end="2024-06-30",
            seed=123,
            data_dir=str(mock_dir),
        )
        for sym in ["FAKE1", "FAKE2"]:
            result = verify_manifest(mock_dir / f"{sym}.parquet")
            print(f"  {sym}: status={result.status}, provider={result.provider}, detail={result.detail}")

        # --- 4. Missing manifest ---
        section("4. Missing manifest (no manifest file)")
        import pandas as pd
        bare = dest / "BARE.parquet"
        pd.DataFrame({"close": [1.0]}).to_parquet(bare)
        result = verify_manifest(bare)
        print(f"  BARE: status={result.status}, provider={result.provider}, detail={result.detail}")

        # --- 5. Corrupted data ---
        section("5. Corrupted data (tamper parquet after manifest)")
        aapl_path = dest / "AAPL.parquet"
        aapl_path.write_bytes(b"corrupted!")
        result = verify_manifest(aapl_path)
        print(f"  AAPL: status={result.status}, provider={result.provider}, detail={result.detail}")

    print(f"\n{'=' * 60}")
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
