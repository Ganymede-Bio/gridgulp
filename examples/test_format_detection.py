#!/usr/bin/env python3
"""
Test script demonstrating enhanced file format detection.
This script tests GridGulp's ability to detect actual file formats
regardless of file extensions.
"""

import asyncio
import sys
from pathlib import Path

from gridgulp import GridGulp


async def test_format_detection():
    """Test format detection with files that have wrong extensions."""
    print("GridGulp Enhanced File Format Detection Test")
    print("=" * 50)

    # Initialize GridGulp
    porter = GridGulp()

    # Test files with wrong extensions
    test_files = [
        "examples/test_files/csv_with_xls_extension.xls",
        "examples/test_files/tsv_with_csv_extension.csv",
        "examples/test_files/csv_with_xlsx_extension.xlsx",
        # Also test correct files for comparison
        "examples/spreadsheets/simple/basic_table.csv",
        "examples/spreadsheets/financial/balance_sheet.csv",
    ]

    results = []

    for file_path in test_files:
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"\n❌ File not found: {file_path}")
            continue

        print(f"\n=== Testing {file_path.name} ===")

        try:
            result = await porter.detect_tables(file_path)
            file_info = result.file_info

            print(f"📁 File: {file_info.path.name}")
            print(f"📋 Extension suggests: {file_info.extension_format}")
            print(f"🔍 Detected format: {file_info.type}")
            print(f"🔧 Detection method: {file_info.detection_method}")
            print(f"📊 Confidence: {file_info.detection_confidence:.1%}")

            if file_info.detected_mime:
                print(f"🏷️  MIME type: {file_info.detected_mime}")

            if file_info.encoding:
                print(f"🔤 Encoding: {file_info.encoding}")

            if file_info.magic_bytes:
                print(f"🔮 Magic bytes: {file_info.magic_bytes[:16]}...")

            if file_info.format_mismatch:
                print("⚠️  WARNING: File extension doesn't match content!")
                print(
                    f"   → File appears to be {file_info.type} but has .{file_path.suffix[1:]} extension"
                )
            else:
                print("✅ Extension matches detected content")

            # Show processing results
            print(f"📈 Processing: {result.detection_time:.3f}s, {len(result.sheets)} sheets")

            results.append(
                {
                    "file": file_path.name,
                    "detected": file_info.type,
                    "extension": file_info.extension_format,
                    "mismatch": file_info.format_mismatch,
                    "confidence": file_info.detection_confidence,
                    "method": file_info.detection_method,
                }
            )

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    # Summary
    print(f"\n{'='*50}")
    print("DETECTION SUMMARY")
    print(f"{'='*50}")

    if results:
        print(
            f"{'File':<30} {'Detected':<8} {'Extension':<10} {'Match':<7} {'Confidence':<10} {'Method'}"
        )
        print("-" * 80)

        for r in results:
            match_status = "✅" if not r["mismatch"] else "⚠️ "
            print(
                f"{r['file']:<30} {r['detected']:<8} {r['extension']:<10} {match_status:<7} {r['confidence']:<10.1%} {r['method']}"
            )

        # Statistics
        mismatches = sum(1 for r in results if r["mismatch"])
        print(f"\nTotal files tested: {len(results)}")
        print(f"Format mismatches detected: {mismatches}")
        print(f"Average confidence: {sum(r['confidence'] for r in results) / len(results):.1%}")

        detection_methods = {}
        for r in results:
            detection_methods[r["method"]] = detection_methods.get(r["method"], 0) + 1

        print("\nDetection methods used:")
        for method, count in detection_methods.items():
            print(f"  - {method}: {count}")

    print(f"\n{'='*50}")
    print("✅ Format detection test completed!")

    if any(r["mismatch"] for r in results):
        print("\n💡 TIP: Files with format mismatches will still be processed correctly")
        print("   because GridGulp uses the detected format, not the file extension.")


async def test_single_file_detection():
    """Test detection on a single file with detailed output."""
    test_file = "examples/test_files/csv_with_xls_extension.xls"

    if not Path(test_file).exists():
        print(f"Test file not found: {test_file}")
        return

    print("\n" + "=" * 50)
    print("DETAILED SINGLE FILE ANALYSIS")
    print("=" * 50)

    from gridgulp.utils.file_magic import detect_file_info

    # Test with our enhanced detection
    result = detect_file_info(Path(test_file))

    print(f"File: {test_file}")
    print(f"Detected type: {result.detected_type}")
    print(f"Extension type: {result.extension_type}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Method: {result.method}")
    print(f"MIME type: {result.mime_type}")
    print(f"Encoding: {result.encoding}")
    print(f"Magic bytes: {result.magic_bytes}")
    print(f"Format mismatch: {result.format_mismatch}")


if __name__ == "__main__":
    print("Starting GridGulp format detection tests...\n")

    try:
        asyncio.run(test_format_detection())
        asyncio.run(test_single_file_detection())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        sys.exit(1)
