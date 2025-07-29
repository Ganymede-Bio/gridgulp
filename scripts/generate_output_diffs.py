#!/usr/bin/env python3
"""Generate diff reports for test outputs."""

import argparse
import difflib
import json
from datetime import datetime
from pathlib import Path


def generate_diffs(captures_dir: Path, golden_dir: Path, diffs_dir: Path):
    """Generate diff reports for all captured outputs."""
    diffs_dir.mkdir(parents=True, exist_ok=True)

    # Find all capture files
    capture_files = list(captures_dir.glob("*_output.json"))

    reports = []

    for capture_file in capture_files:
        # Extract test name from filename
        parts = capture_file.stem.split("_")
        # Format: timestamp_testname_output
        test_name = "_".join(parts[1:-1])

        # Find corresponding golden file
        golden_file = golden_dir / f"{test_name}_output.json"

        if not golden_file.exists():
            print(f"No golden output for {test_name}")
            continue

        # Load files
        with open(capture_file) as f:
            captured = json.load(f)

        with open(golden_file) as f:
            golden = json.load(f)

        # Compare
        if captured != golden:
            # Generate diff
            diff_html = generate_diff_html(test_name, golden, captured)

            # Save diff
            diff_file = diffs_dir / f"{capture_file.stem}_diff.html"
            with open(diff_file, "w") as f:
                f.write(diff_html)

            reports.append(
                {
                    "test": test_name,
                    "capture_file": str(capture_file),
                    "golden_file": str(golden_file),
                    "diff_file": str(diff_file),
                    "status": "differences_found",
                }
            )
        else:
            reports.append(
                {
                    "test": test_name,
                    "capture_file": str(capture_file),
                    "golden_file": str(golden_file),
                    "status": "match",
                }
            )

    # Generate summary report
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_tests": len(reports),
        "matches": sum(1 for r in reports if r["status"] == "match"),
        "differences": sum(1 for r in reports if r["status"] == "differences_found"),
        "reports": reports,
    }

    summary_file = diffs_dir / "diff_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Generated {len(reports)} diff reports")
    print(f"Summary saved to {summary_file}")


def generate_diff_html(test_name: str, golden: dict, captured: dict) -> str:
    """Generate HTML diff report."""
    # Convert to pretty JSON
    golden_json = json.dumps(golden, indent=2, sort_keys=True)
    captured_json = json.dumps(captured, indent=2, sort_keys=True)

    # Generate line-by-line diff
    diff = list(
        difflib.unified_diff(
            golden_json.splitlines(keepends=False),
            captured_json.splitlines(keepends=False),
            fromfile="golden",
            tofile="captured",
            lineterm="",
        )
    )

    # Create HTML
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>Diff: {test_name}</title>",
        "<style>",
        "body { font-family: monospace; margin: 20px; }",
        "h1 { color: #333; }",
        ".diff { background: #f5f5f5; padding: 10px; }",
        ".added { background: #cfc; }",
        ".removed { background: #fcc; }",
        ".linenum { color: #666; margin-right: 10px; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Diff Report: {test_name}</h1>",
        '<div class="diff">',
        "<pre>",
    ]

    for line in diff:
        if line.startswith("+"):
            html_lines.append(f'<span class="added">{escape_html(line)}</span>')
        elif line.startswith("-"):
            html_lines.append(f'<span class="removed">{escape_html(line)}</span>')
        else:
            html_lines.append(escape_html(line))

    html_lines.extend(["</pre>", "</div>", "</body>", "</html>"])

    return "\n".join(html_lines)


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate diff reports for test outputs")
    parser.add_argument(
        "--captures-dir",
        type=Path,
        default=Path("tests/outputs/captures"),
        help="Directory containing captured outputs",
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=Path("tests/outputs/golden"),
        help="Directory containing golden outputs",
    )
    parser.add_argument(
        "--diffs-dir",
        type=Path,
        default=Path("tests/outputs/diffs"),
        help="Directory to save diff reports",
    )

    args = parser.parse_args()

    generate_diffs(args.captures_dir, args.golden_dir, args.diffs_dir)


if __name__ == "__main__":
    main()
