"""Pytest configuration for output capture."""

from pathlib import Path

import pytest

from tests.utils import TestOutputCapture


@pytest.fixture
def capture_outputs(request):
    """
    Fixture that provides output capture for tests.

    Usage:
        def test_something(capture_outputs):
            capture_outputs.capture_input("file.xlsx", {})
            result = process_file("file.xlsx")
            capture_outputs.capture_output(result)
    """
    test_name = request.node.name
    capture = TestOutputCapture(test_name)

    yield capture

    # Generate summary report after test
    capture.generate_summary_report()


@pytest.fixture
def golden_outputs():
    """Fixture providing access to golden outputs directory."""
    return Path(__file__).parent / "outputs" / "golden"


def pytest_addoption(parser):
    """Add command line options for output capture."""
    parser.addoption(
        "--capture-outputs",
        action="store_true",
        default=False,
        help="Enable output capture for all tests",
    )
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Update golden outputs with current results",
    )
    parser.addoption(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for test outputs (default: tests/outputs)",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on options."""
    if config.getoption("--capture-outputs"):
        # Add output capture marker to all tests
        for item in items:
            item.add_marker(pytest.mark.capture_outputs)


@pytest.mark.capture_outputs
def pytest_runtest_makereport(item, call):
    """Hook to capture test results."""
    if call.when == "call" and hasattr(item, "capture_outputs"):
        # Test has completed, finalize capture
        capture = item.capture_outputs

        # Compare with golden if available
        comparison = capture.compare_with_golden()

        # Update golden if requested
        if item.config.getoption("--update-golden"):
            capture.save_as_golden()


# Pytest plugin to automatically capture pipeline outputs
class OutputCapturePlugin:
    """Plugin to automatically capture GridGulp pipeline outputs."""

    def __init__(self, config):
        self.config = config
        self.enabled = config.getoption("--capture-outputs")
        self.output_dir = config.getoption("--output-dir")

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        """Wrap test execution to capture outputs."""
        if self.enabled and "capture_outputs" in item.fixturenames:
            # Test uses capture_outputs fixture
            capture = TestOutputCapture(item.name, outputs_dir=self.output_dir)
            item.capture_outputs = capture

        yield


def pytest_configure(config):
    """Configure pytest with output capture plugin."""
    # Register custom marker
    config.addinivalue_line("markers", "capture_outputs: mark test for output capture")
    config.pluginmanager.register(OutputCapturePlugin(config))
