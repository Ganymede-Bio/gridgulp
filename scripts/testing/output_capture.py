"""Test output capture system for GridGulp."""

import difflib
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class TestOutputCapture:
    """
    Captures pipeline inputs/outputs for analysis and refinement.

    This system enables:
    - Automatic capture of test inputs/outputs
    - JSON serialization of pipeline data
    - Comparison with golden outputs
    - HTML diff report generation
    """

    def __init__(self, test_name: str, outputs_dir: Path | None = None):
        """Initialize output capture for a test."""
        self.test_name = test_name
        self.timestamp = datetime.now().isoformat().replace(":", "-")

        # Set up directories
        if outputs_dir:
            self.outputs_dir = Path(outputs_dir)
        else:
            self.outputs_dir = Path(__file__).parent.parent / "outputs"

        self.captures_dir = self.outputs_dir / "captures"
        self.golden_dir = self.outputs_dir / "golden"
        self.diffs_dir = self.outputs_dir / "diffs"

        # Create directories
        for dir_path in [self.captures_dir, self.golden_dir, self.diffs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.captured_stages = {}

    def capture(self, stage: str, data: Any) -> None:
        """
        Capture data at any pipeline stage.

        Args:
            stage: Name of the pipeline stage
            data: Data to capture
        """
        # Store in memory
        self.captured_stages[stage] = data

        # Write to file
        filename = f"{self.timestamp}_{self.test_name}_{stage}.json"
        filepath = self.captures_dir / filename

        with open(filepath, "w") as f:
            json.dump(self._serialize(data), f, indent=2)

    def capture_input(self, file_path: str, options: dict[str, Any]) -> None:
        """Capture pipeline input."""
        self.capture(
            "input",
            {
                "file_path": file_path,
                "options": options,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def capture_detection(self, tables_detected: Any) -> None:
        """Capture detection results."""
        self.capture("detection", self._serialize_tables(tables_detected))

    def capture_vision(self, vision_request: Any, vision_response: Any) -> None:
        """Capture vision analysis."""
        self.capture(
            "vision",
            {
                "request": self._serialize(vision_request),
                "response": self._serialize(vision_response),
            },
        )

    def capture_extraction(self, extracted_tables: Any) -> None:
        """Capture extraction results."""
        self.capture("extraction", self._serialize_tables(extracted_tables))

    def capture_output(self, final_result: dict[str, Any]) -> None:
        """Capture final pipeline output."""
        self.capture("output", final_result)

    def capture_metrics(self, metrics: dict[str, Any]) -> None:
        """Capture performance metrics."""
        self.capture(
            "metrics",
            {
                "test_name": self.test_name,
                "timestamp": self.timestamp,
                "metrics": metrics,
            },
        )

    def compare_with_golden(self, stage: str = "output") -> dict[str, Any]:
        """
        Compare captured output with golden output.

        Args:
            stage: Stage to compare (default: "output")

        Returns:
            Comparison results
        """
        golden_path = self.golden_dir / f"{self.test_name}_{stage}.json"

        if not golden_path.exists():
            return {
                "status": "no_golden",
                "message": f"No golden output found at {golden_path}",
            }

        if stage not in self.captured_stages:
            return {
                "status": "no_capture",
                "message": f"No captured data for stage '{stage}'",
            }

        # Load golden output
        with open(golden_path) as f:
            golden_data = json.load(f)

        captured_data = self._serialize(self.captured_stages[stage])

        # Compare
        differences = self._compare_data(golden_data, captured_data)

        # Generate diff report
        if differences:
            self._generate_diff_report(stage, golden_data, captured_data, differences)

        return {
            "status": "match" if not differences else "mismatch",
            "differences": differences,
            "diff_report": self.diffs_dir / f"{self.timestamp}_{self.test_name}_{stage}_diff.html",
        }

    def save_as_golden(self, stage: str = "output") -> None:
        """Save current capture as golden output."""
        if stage not in self.captured_stages:
            raise ValueError(f"No captured data for stage '{stage}'")

        golden_path = self.golden_dir / f"{self.test_name}_{stage}.json"

        with open(golden_path, "w") as f:
            json.dump(self._serialize(self.captured_stages[stage]), f, indent=2)

    def generate_summary_report(self) -> Path:
        """Generate summary report of all captures."""
        report = {
            "test_name": self.test_name,
            "timestamp": self.timestamp,
            "stages_captured": list(self.captured_stages.keys()),
            "files_generated": [],
        }

        # List all generated files
        for stage in self.captured_stages:
            filename = f"{self.timestamp}_{self.test_name}_{stage}.json"
            report["files_generated"].append(str(self.captures_dir / filename))

        # Save summary
        summary_path = self.captures_dir / f"{self.timestamp}_{self.test_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(report, f, indent=2)

        return summary_path

    def _serialize(self, obj: Any) -> Any:
        """Serialize object to JSON-compatible format."""
        if hasattr(obj, "dict"):
            # Pydantic models
            return obj.dict()
        elif hasattr(obj, "__dict__"):
            # Regular objects
            return {
                "_type": obj.__class__.__name__,
                **{k: self._serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")},
            }
        elif isinstance(obj, list | tuple):
            return [self._serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        else:
            # Primitive types
            return obj

    def _serialize_tables(self, tables: Any) -> list:
        """Serialize table objects."""
        if not tables:
            return []

        serialized = []
        for table in tables:
            if hasattr(table, "dict"):
                serialized.append(table.dict())
            else:
                serialized.append(self._serialize(table))

        return serialized

    def _compare_data(self, expected: Any, actual: Any, path: str = "") -> list:
        """Compare two data structures and return differences."""
        differences = []

        if not isinstance(actual, type(expected)):
            differences.append(
                {
                    "path": path,
                    "type": "type_mismatch",
                    "expected_type": type(expected).__name__,
                    "actual_type": type(actual).__name__,
                }
            )
            return differences

        if isinstance(expected, dict):
            # Compare keys
            expected_keys = set(expected.keys())
            actual_keys = set(actual.keys())

            if expected_keys != actual_keys:
                differences.append(
                    {
                        "path": path,
                        "type": "keys_mismatch",
                        "missing_keys": list(expected_keys - actual_keys),
                        "extra_keys": list(actual_keys - expected_keys),
                    }
                )

            # Compare values
            for key in expected_keys & actual_keys:
                differences.extend(self._compare_data(expected[key], actual[key], f"{path}.{key}"))

        elif isinstance(expected, list):
            if len(expected) != len(actual):
                differences.append(
                    {
                        "path": path,
                        "type": "length_mismatch",
                        "expected_length": len(expected),
                        "actual_length": len(actual),
                    }
                )
            else:
                for i, (exp_item, act_item) in enumerate(zip(expected, actual, strict=False)):
                    differences.extend(self._compare_data(exp_item, act_item, f"{path}[{i}]"))

        else:
            # Primitive comparison
            if expected != actual:
                differences.append(
                    {
                        "path": path,
                        "type": "value_mismatch",
                        "expected": expected,
                        "actual": actual,
                    }
                )

        return differences

    def _generate_diff_report(
        self, stage: str, expected: Any, actual: Any, differences: list
    ) -> None:
        """Generate HTML diff report."""
        # Convert to pretty JSON strings
        expected_json = json.dumps(expected, indent=2)
        actual_json = json.dumps(actual, indent=2)

        # Generate diff
        diff = difflib.unified_diff(
            expected_json.splitlines(keepends=True),
            actual_json.splitlines(keepends=True),
            fromfile=f"golden/{self.test_name}_{stage}.json",
            tofile=f"captured/{self.timestamp}_{self.test_name}_{stage}.json",
        )

        # Create HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Diff Report: {self.test_name} - {stage}</title>
    <style>
        body {{ font-family: monospace; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
        .diff {{ background: #fff; border: 1px solid #ccc; padding: 10px; }}
        .added {{ background: #cfc; }}
        .removed {{ background: #fcc; }}
        .difference {{ margin: 10px 0; padding: 5px; background: #ffc; }}
    </style>
</head>
<body>
    <h1>Diff Report: {self.test_name} - {stage}</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p>Test: {self.test_name}</p>
        <p>Stage: {stage}</p>
        <p>Timestamp: {self.timestamp}</p>
        <p>Differences found: {len(differences)}</p>
    </div>

    <div class="differences">
        <h2>Differences</h2>
        {"".join(f'<div class="difference">{html.escape(str(d))}</div>' for d in differences)}
    </div>

    <div class="diff">
        <h2>Full Diff</h2>
        <pre>{"".join(html.escape(line) for line in diff)}</pre>
    </div>
</body>
</html>
"""

        # Save report
        report_path = self.diffs_dir / f"{self.timestamp}_{self.test_name}_{stage}_diff.html"
        with open(report_path, "w") as f:
            f.write(html_content)
