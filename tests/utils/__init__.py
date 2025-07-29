"""Test utilities for GridGulp."""

# Import from scripts.testing where the actual implementation lives
import sys
from pathlib import Path

# Add scripts/testing to path if needed
scripts_testing_path = Path(__file__).parent.parent.parent / "scripts" / "testing"
if str(scripts_testing_path) not in sys.path:
    sys.path.insert(0, str(scripts_testing_path))

from output_capture import TestOutputCapture

__all__ = ["TestOutputCapture"]
