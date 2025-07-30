#!/usr/bin/env python3
"""Demonstration of GridGulp performance."""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gridgulp.config import GridGulpConfig  # noqa: E402
from gridgulp.models import FileInfo, FileType  # noqa: E402
from gridgulp.readers import ExcelReader, ReaderAdapter  # noqa: E402


async def benchmark_readers(file_path: Path):
    """Benchmark Excel reader performance."""
    print(f"\nBenchmarking reader for: {file_path.name}")
    print("=" * 50)

    # Create file info
    file_info = FileInfo(
        path=file_path,
        type=FileType.XLSX,
        size_mb=file_path.stat().st_size / (1024 * 1024),
    )

    # Test: Openpyxl reader
    print("\nOpenpyxl Reader:")
    start = time.time()
    excel_reader = ExcelReader(file_path, file_info)
    file_data = await excel_reader.read()
    openpyxl_time = time.time() - start
    print(f"   Time: {openpyxl_time:.2f}s")
    print(f"   Sheets: {len(file_data.sheets)}")
    print(f"   Total cells: {sum(len(sheet.cells) for sheet in file_data.sheets)}")

    # Show results
    print("\n📊 Performance Summary:")
    print(f"   Processed {file_path.name} in {openpyxl_time:.2f}s")


async def demonstrate_adapter():
    """Show how the adapter provides flexibility."""
    print("\n\nDemonstrating Reader Adapter")
    print("=" * 50)

    # Configuration examples
    config = GridGulpConfig()
    ReaderAdapter(config)
    print("Default config loaded successfully")


async def main():
    """Run all demonstrations."""
    print("GridGulp Performance Demo")
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
