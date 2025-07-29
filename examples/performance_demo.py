#!/usr/bin/env python3
"""Demonstration of GridGulp performance improvements."""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gridgulp.config import GridGulpConfig  # noqa: E402
from gridgulp.models import FileInfo, FileType  # noqa: E402
from gridgulp.readers import CalamineReader, ExcelReader, ReaderAdapter  # noqa: E402


async def benchmark_readers(file_path: Path):
    """Benchmark different Excel readers."""
    print(f"\nBenchmarking readers for: {file_path.name}")
    print("=" * 50)

    # Create file info
    file_info = FileInfo(
        path=file_path,
        type=FileType.XLSX,
        size_mb=file_path.stat().st_size / (1024 * 1024),
    )

    # Test 1: Calamine reader (fast)
    print("\n1. Calamine Reader (Rust-based):")
    start = time.time()
    calamine_reader = CalamineReader(file_path, file_info)
    file_data = await calamine_reader.read()
    calamine_time = time.time() - start
    print(f"   Time: {calamine_time:.2f}s")
    print(f"   Sheets: {len(file_data.sheets)}")
    print(f"   Total cells: {sum(len(sheet.get_non_empty_cells()) for sheet in file_data.sheets)}")

    # Test 2: Calamine to Polars
    print("\n2. Calamine Reader → Polars DataFrames:")
    start = time.time()
    dfs = calamine_reader.read_to_polars()
    polars_time = time.time() - start
    print(f"   Time: {polars_time:.2f}s")
    print(f"   DataFrames: {len(dfs)}")
    for i, df in enumerate(dfs):
        print(f"   Sheet {i}: {df.shape} (rows × cols)")

    # Test 3: Traditional openpyxl reader
    print("\n3. Openpyxl Reader (traditional):")
    start = time.time()
    excel_reader = ExcelReader(file_path, file_info)
    file_data = await excel_reader.read()
    openpyxl_time = time.time() - start
    print(f"   Time: {openpyxl_time:.2f}s")
    print(f"   Sheets: {len(file_data.sheets)}")

    # Show speedup
    print("\n📊 Performance Summary:")
    print(f"   Calamine is {openpyxl_time/calamine_time:.1f}x faster than openpyxl")
    print(f"   Calamine→Polars is {openpyxl_time/polars_time:.1f}x faster than openpyxl")


async def demonstrate_adapter():
    """Show how the adapter provides flexibility."""
    print("\n\nDemonstrating Reader Adapter")
    print("=" * 50)

    # Configuration 1: Use Calamine for speed
    config1 = GridGulpConfig(excel_reader="calamine", use_polars=True)
    ReaderAdapter(config1)
    print(f"Config 1: excel_reader={config1.excel_reader}, use_polars={config1.use_polars}")

    # Configuration 2: Use openpyxl for features
    config2 = GridGulpConfig(excel_reader="openpyxl", use_polars=False)
    ReaderAdapter(config2)
    print(f"Config 2: excel_reader={config2.excel_reader}, use_polars={config2.use_polars}")

    # Configuration 3: Auto-select
    config3 = GridGulpConfig(excel_reader="auto", use_polars=True)
    ReaderAdapter(config3)
    print(f"Config 3: excel_reader={config3.excel_reader}, use_polars={config3.use_polars}")


async def main():
    """Run all demonstrations."""
    print("GridGulp Performance Improvements Demo")
    print("=" * 50)

    # Find a sample Excel file
    sample_files = [
        Path("sample.xlsx"),
        Path("../sample_data/sample.xlsx"),
        Path("../tests/data/sample.xlsx"),
    ]

    excel_file = None
    for f in sample_files:
        if f.exists():
            excel_file = f
            break

    if excel_file:
        await benchmark_readers(excel_file)
    else:
        print("\n⚠️  No sample Excel file found for benchmarking")
        print("   Create a file named 'sample.xlsx' to test")

    # Always run these demos
    await demonstrate_adapter()

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
