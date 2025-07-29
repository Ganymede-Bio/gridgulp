#!/usr/bin/env python3
"""Run tests for new v0.3.0 features."""

import subprocess
import sys


def run_tests():
    """Run tests for new features."""
    print("Running tests for v0.3.0 features...")
    print("=" * 80)

    test_files = [
        "tests/extractors/test_dataframe_extractor.py",
        "tests/detectors/test_structured_text_detector.py",
        "tests/models/test_extraction_result.py",
    ]

    all_passed = True

    for test_file in test_files:
        print(f"\nRunning {test_file}...")
        print("-" * 40)

        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v"], capture_output=True, text=True
        )

        if result.returncode == 0:
            print(f"✓ {test_file} - All tests passed")
        else:
            print(f"✗ {test_file} - Some tests failed")
            print(result.stdout)
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All new feature tests passed!")
    else:
        print("✗ Some tests failed. See output above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
