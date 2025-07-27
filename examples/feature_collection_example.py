"""Example of using GridPorter with feature collection enabled."""

import asyncio
import os
from pathlib import Path

from gridporter import GridPorter
from gridporter.telemetry import get_feature_collector


async def main():
    """Run GridPorter with feature collection enabled."""

    # Enable feature collection via environment variable
    os.environ["GRIDPORTER_ENABLE_FEATURE_COLLECTION"] = "true"

    # Initialize GridPorter
    gridporter = GridPorter()

    # Example: Process a file
    file_path = Path(__file__).parent.parent / "tests" / "data" / "test_complex.xlsx"

    if file_path.exists():
        print(f"Processing {file_path}...")

        # Detect tables
        result = await gridporter.detect_tables(file_path)

        print(f"\nDetected {len(result.sheets)} sheets")
        for sheet in result.sheets:
            print(f"\n  Sheet: {sheet.name}")
            print(f"  Tables found: {len(sheet.tables)}")
            for table in sheet.tables:
                print(f"    - Range: {table.range}, Confidence: {table.confidence:.2f}")

        # Check collected features
        feature_collector = get_feature_collector()
        if feature_collector.enabled:
            print("\n\nFeature Collection Statistics:")
            stats = feature_collector.get_summary_statistics()

            print(f"  Total detections recorded: {stats['total_records']}")
            print(f"  Average confidence: {stats['avg_confidence']:.3f}")
            print(f"  Success rate: {stats['success_rate']:.1%}")

            print("\n  Detection methods used:")
            for method, info in stats["by_method"].items():
                print(
                    f"    - {method}: {info['count']} detections, avg confidence {info['avg_confidence']:.3f}"
                )

            print("\n  Pattern types detected:")
            for pattern, info in stats["by_pattern"].items():
                print(
                    f"    - {pattern}: {info['count']} occurrences, avg confidence {info['avg_confidence']:.3f}"
                )

            # Export features for analysis
            export_path = Path("feature_export.csv")
            feature_collector.export_features(str(export_path))
            print(f"\n  Features exported to: {export_path}")

            # Query high-confidence detections
            high_conf = feature_collector._feature_store.query_features(min_confidence=0.8)
            print(f"\n  High confidence detections (>0.8): {len(high_conf)}")
    else:
        print(f"Test file not found: {file_path}")
        print("Creating a simple example...")

        # You could create a test file here or use a different path


if __name__ == "__main__":
    asyncio.run(main())
