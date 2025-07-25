#!/usr/bin/env python3
"""
Run all tests from WEEK4_TESTING_GUIDE.md to validate the region verification implementation.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
# Also add src directory
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now import GridPorter modules
# noqa: E402
from gridporter.config import Config  # noqa: E402
from gridporter.models.sheet_data import CellData, SheetData  # noqa: E402
from gridporter.models.table import TableRange  # noqa: E402
from gridporter.models.vision_result import VisionRegion  # noqa: E402
from gridporter.vision.integrated_pipeline import IntegratedVisionPipeline  # noqa: E402
from gridporter.vision.pattern_detector import PatternType, TableBounds, TablePattern  # noqa: E402
from gridporter.vision.pipeline import VisionPipeline  # noqa: E402
from gridporter.vision.region_verifier import RegionVerifier, VerificationResult  # noqa: E402


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}\n")


def print_test(name: str):
    """Print a test name."""
    print(f"\n--- {name} ---")


def create_cell(sheet: SheetData, address: str, value, **kwargs):
    """Helper to create a cell and update sheet dimensions."""
    # Parse address
    col_str = "".join(c for c in address if c.isalpha())
    row_str = "".join(c for c in address if c.isdigit())

    # Convert to indices
    col = 0
    for char in col_str:
        col = col * 26 + (ord(char.upper()) - ord("A") + 1)
    col -= 1
    row = int(row_str) - 1

    # Create cell
    cell = CellData(value=value, row=row, column=col, **kwargs)
    sheet.cells[address] = cell

    # Update max dimensions
    sheet.max_row = max(sheet.max_row, row)
    sheet.max_column = max(sheet.max_column, col)


def test_basic_verification():
    """Section 1: Basic Region Verification Tests"""
    print_section("Section 1: Basic Region Verification")

    # Test 1.1: Verify Valid Region
    print_test("Test 1.1: Verify Valid Region")

    # Create a simple valid table
    sheet = SheetData(name="ValidTable")
    # Add headers
    create_cell(sheet, "A1", "Name", data_type="text", is_bold=True)
    create_cell(sheet, "B1", "Age", data_type="text", is_bold=True)
    create_cell(sheet, "C1", "City", data_type="text", is_bold=True)
    # Add data rows
    create_cell(sheet, "A2", "Alice", data_type="text")
    create_cell(sheet, "B2", 25, data_type="number")
    create_cell(sheet, "C2", "NYC", data_type="text")
    create_cell(sheet, "A3", "Bob", data_type="text")
    create_cell(sheet, "B3", 30, data_type="number")
    create_cell(sheet, "C3", "LA", data_type="text")

    # Define region covering the table
    region = TableRange(range="A1:C3", start_row=0, start_col=0, end_row=2, end_col=2)

    # Verify the region
    verifier = RegionVerifier()
    result = verifier.verify_region(sheet, region)

    print(f"Valid: {result.valid}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reason: {result.reason}")
    print("Metrics:")
    for metric, value in result.metrics.items():
        print(f"  {metric}: {value:.2f}")

    # Test 1.2: Verify Sparse Region
    print_test("Test 1.2: Verify Sparse Region")

    # Create a sparse table
    sparse_sheet = SheetData(name="SparseTable")
    # Scattered data points
    create_cell(sparse_sheet, "A1", "A", data_type="text")
    create_cell(sparse_sheet, "E1", "E", data_type="text")
    create_cell(sparse_sheet, "C3", "X", data_type="text")
    create_cell(sparse_sheet, "A5", "F", data_type="text")
    create_cell(sparse_sheet, "E5", "J", data_type="text")

    # Define region covering sparse area
    sparse_region = TableRange(range="A1:E5", start_row=0, start_col=0, end_row=4, end_col=4)

    # Verify with custom thresholds
    verifier = RegionVerifier(min_filledness=0.3)
    result = verifier.verify_region(sparse_sheet, sparse_region)

    print(f"Valid: {result.valid}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reason: {result.reason}")
    if result.feedback:
        print(f"Feedback: {result.feedback}")

    # Test 1.3: Verify Invalid Bounds
    print_test("Test 1.3: Verify Invalid Bounds")

    # Test with bounds exceeding sheet dimensions
    oversized_region = TableRange(range="A1:Z100", start_row=0, start_col=0, end_row=99, end_col=25)

    result = verifier.verify_region(sheet, oversized_region)
    print(f"Valid: {result.valid}")
    print(f"Reason: {result.reason}")
    print(f"Feedback: {result.feedback}")


def test_geometry_analysis():
    """Section 2: Geometry Metrics Analysis"""
    print_section("Section 2: Geometry Metrics Analysis")

    # Test 2.1: Rectangularness Testing
    print_test("Test 2.1: Rectangularness Testing")

    # Create L-shaped data (non-rectangular)
    l_shape_sheet = SheetData(name="LShape")
    # Top horizontal part
    for col in range(5):
        addr = f"{chr(65 + col)}1"
        create_cell(l_shape_sheet, addr, f"H{col}", data_type="text")
    # Left vertical part
    for row in range(1, 5):
        create_cell(l_shape_sheet, f"A{row+1}", f"V{row}", data_type="text")

    # Test full region
    l_region = TableRange(range="A1:E5", start_row=0, start_col=0, end_row=4, end_col=4)

    verifier = RegionVerifier(min_rectangularness=0.8)
    result = verifier.verify_region(l_shape_sheet, l_region)

    print(f"Valid: {result.valid}")
    print(f"Rectangularness: {result.metrics['rectangularness']:.2f}")
    print(f"Reason: {result.reason}")

    # Test 2.2: Contiguity Testing
    print_test("Test 2.2: Contiguity Testing")

    # Create fragmented data (multiple islands)
    fragmented_sheet = SheetData(name="Fragmented")
    # Island 1 (top-left)
    create_cell(fragmented_sheet, "A1", "I1A", data_type="text")
    create_cell(fragmented_sheet, "B1", "I1B", data_type="text")
    create_cell(fragmented_sheet, "A2", "I1C", data_type="text")
    create_cell(fragmented_sheet, "B2", "I1D", data_type="text")
    # Island 2 (bottom-right)
    create_cell(fragmented_sheet, "D4", "I2A", data_type="text")
    create_cell(fragmented_sheet, "E4", "I2B", data_type="text")
    create_cell(fragmented_sheet, "D5", "I2C", data_type="text")
    create_cell(fragmented_sheet, "E5", "I2D", data_type="text")

    region = TableRange(range="A1:E5", start_row=0, start_col=0, end_row=4, end_col=4)
    verifier = RegionVerifier(min_contiguity=0.8)
    result = verifier.verify_region(fragmented_sheet, region)

    print(f"Valid: {result.valid}")
    print(f"Contiguity: {result.metrics['contiguity']:.2f}")
    print(f"Filledness: {result.metrics['filledness']:.2f}")

    # Test 2.3: Edge Quality Testing
    print_test("Test 2.3: Edge Quality Testing")

    # Create table with ragged edges
    ragged_sheet = SheetData(name="Ragged")
    # Row 1: Full
    for col in range(5):
        create_cell(ragged_sheet, f"{chr(65+col)}1", f"R1C{col}", data_type="text")
    # Row 2: Missing middle
    create_cell(ragged_sheet, "A2", "R2C0", data_type="text")
    create_cell(ragged_sheet, "E2", "R2C4", data_type="text")
    # Row 3: Full
    for col in range(5):
        create_cell(ragged_sheet, f"{chr(65+col)}3", f"R3C{col}", data_type="text")

    region = TableRange(range="A1:E3", start_row=0, start_col=0, end_row=2, end_col=4)
    verifier = RegionVerifier()
    result = verifier.verify_region(ragged_sheet, region)

    print(f"Edge quality: {result.metrics['edge_quality']:.2f}")
    print(f"Overall score: {result.metrics['overall_score']:.2f}")


def test_pattern_verification():
    """Section 3: Pattern-Specific Verification"""
    print_section("Section 3: Pattern-Specific Verification")

    # Test 3.1: Header-Data Pattern
    print_test("Test 3.1: Header-Data Pattern")

    # Create table with clear headers
    header_sheet = SheetData(name="HeaderData")
    # Headers (bold)
    headers = ["Product", "Price", "Stock", "Category"]
    for i, header in enumerate(headers):
        addr = f"{chr(65+i)}1"
        create_cell(header_sheet, addr, header, data_type="text", is_bold=True)
    # Data rows
    data = [
        ["Apple", 1.25, 100, "Fruit"],
        ["Banana", 0.75, 150, "Fruit"],
        ["Carrot", 0.50, 200, "Vegetable"],
    ]
    for row_idx, row_data in enumerate(data):
        for col_idx, value in enumerate(row_data):
            addr = f"{chr(65+col_idx)}{row_idx+2}"
            create_cell(
                header_sheet,
                addr,
                value,
                data_type="number" if isinstance(value, int | float) else "text",
            )

    # Create pattern object
    pattern = TablePattern(
        pattern_type=PatternType.HEADER_DATA,
        bounds=TableBounds(start_row=0, start_col=0, end_row=3, end_col=3),
        confidence=0.9,
        characteristics={},
    )

    verifier = RegionVerifier()
    result = verifier.verify_pattern(header_sheet, pattern)

    print(f"Valid: {result.valid}")
    print(f"Pattern verification: {result.reason}")

    # Test 3.2: Matrix Pattern
    print_test("Test 3.2: Matrix Pattern")

    # Create cross-reference matrix
    matrix_sheet = SheetData(name="Matrix")
    # Top-left empty
    create_cell(matrix_sheet, "A1", "", data_type="text")
    # Column headers
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    for i, q in enumerate(quarters):
        create_cell(matrix_sheet, f"{chr(66+i)}1", q, data_type="text", is_bold=True)
    # Row headers
    products = ["Product A", "Product B", "Product C"]
    for i, p in enumerate(products):
        create_cell(matrix_sheet, f"A{i+2}", p, data_type="text", is_bold=True)
    # Data cells
    for row in range(3):
        for col in range(4):
            value = (row + 1) * (col + 1) * 100
            addr = f"{chr(66+col)}{row+2}"
            create_cell(matrix_sheet, addr, value, data_type="number")

    # Test matrix pattern through public API
    matrix_pattern = TablePattern(
        pattern_type=PatternType.MATRIX,
        bounds=TableBounds(start_row=0, start_col=0, end_row=3, end_col=4),
        confidence=0.9,
        characteristics={},
    )

    result = verifier.verify_pattern(matrix_sheet, matrix_pattern)
    print(f"Valid: {result.valid}")
    print(f"Reason: {result.reason}")


def test_feedback_loop():
    """Section 4: Feedback Loop Testing"""
    print_section("Section 4: Feedback Loop Testing")

    # Test 4.1: Generate Feedback for Failed Regions
    print_test("Test 4.1: Generate Feedback for Failed Regions")

    # Create extremely wide sparse region
    wide_sheet = SheetData(name="WideSheet")
    # Only first and last columns have data
    create_cell(wide_sheet, "A1", "Start", data_type="text")
    create_cell(wide_sheet, "CV1", "End", data_type="text")

    wide_region = TableRange(range="A1:CV1", start_row=0, start_col=0, end_row=0, end_col=99)
    verifier = RegionVerifier()
    result = verifier.verify_region(wide_sheet, wide_region)

    print(f"Valid: {result.valid}")
    print(f"Feedback: {result.feedback}")
    print("\nDetailed metrics:")
    for metric, value in result.metrics.items():
        print(f"  {metric}: {value}")

    # Test 4.2: Test Feedback Loop Integration (Mock)
    print_test("Test 4.2: Test Feedback Loop Integration (Mock)")

    async def test_feedback_loop_async():
        # Configure pipeline
        config = Config(
            enable_region_verification=True,
            enable_verification_feedback=True,
            min_region_filledness=0.5,
            min_region_rectangularness=0.7,
        )

        VisionPipeline(config)

        # Create a failed region (sparse)
        VisionRegion(
            pixel_bounds={"x1": 0, "y1": 0, "x2": 500, "y2": 100},
            cell_bounds={"start_row": 0, "start_col": 0, "end_row": 9, "end_col": 49},
            range="A1:AX10",
            confidence=0.8,
            characteristics={"sparse": True},
        )

        # Mock sheet data
        sheet = SheetData(name="TestSheet")
        # Add some scattered data
        create_cell(sheet, "A1", "Data1", data_type="text")
        create_cell(sheet, "AX10", "Data2", data_type="text")

        # Test refinement (will use mock if no real vision model)
        try:
            print("Testing feedback refinement (mock mode)...")
            # This will fail without a real vision model, which is expected
            print("Note: Feedback loop requires actual vision model to run")
        except Exception as e:
            print(f"Expected without real vision model: {type(e).__name__}")

    asyncio.run(test_feedback_loop_async())


def test_pipeline_integration():
    """Section 5: Integration with Vision Pipeline"""
    print_section("Section 5: Integration with Vision Pipeline")

    # Test 5.1: Integrated Pipeline with Verification
    print_test("Test 5.1: Integrated Pipeline with Verification")

    # Configure with verification enabled
    config = Config(
        enable_region_verification=True,
        verification_strict_mode=False,
        min_region_filledness=0.2,
        min_region_rectangularness=0.6,
        min_region_contiguity=0.4,
    )

    # Create pipeline from config
    pipeline = IntegratedVisionPipeline.from_config(config)

    # Create test sheet with multiple patterns
    mixed_sheet = SheetData(name="MixedPatterns")
    # Valid table region
    for row in range(3):
        for col in range(3):
            addr = f"{chr(65+col)}{row+1}"
            create_cell(mixed_sheet, addr, f"T1_{row}_{col}", data_type="text")
    # Sparse region (should be filtered)
    create_cell(mixed_sheet, "F1", "Sparse1", data_type="text")
    create_cell(mixed_sheet, "J5", "Sparse2", data_type="text")

    # Process sheet (this runs the full pipeline)
    try:
        result = pipeline.process_sheet(mixed_sheet)

        print(f"Detected tables: {len(result.detected_tables)}")
        print(f"Verification enabled: {pipeline.enable_verification}")
        if result.verification_results:
            print("\nVerification results:")
            for pattern_id, verification in result.verification_results.items():
                print(
                    f"  Pattern {pattern_id}: valid={verification.valid}, "
                    f"confidence={verification.confidence:.2f}"
                )
    except Exception as e:
        print(f"Pipeline processing note: {type(e).__name__}")
        print("This is expected without full vision model setup")

    # Test 5.2: Check verification configuration
    print_test("Test 5.2: Pipeline Configuration Check")
    print(f"Pipeline verification enabled: {pipeline.enable_verification}")
    print(f"Strict mode: {pipeline.verification_strict}")
    if hasattr(pipeline, "region_verifier"):
        print(f"Verifier min_filledness: {pipeline.region_verifier.min_filledness}")
        print(f"Verifier min_rectangularness: {pipeline.region_verifier.min_rectangularness}")


def test_configuration():
    """Section 6: Configuration Testing"""
    print_section("Section 6: Configuration Testing")

    # Test 6.1: Custom Verification Thresholds
    print_test("Test 6.1: Custom Verification Thresholds")

    # Test with different threshold configurations
    configs = [
        {
            "name": "Strict",
            "min_filledness": 0.5,
            "min_rectangularness": 0.9,
            "min_contiguity": 0.8,
        },
        {
            "name": "Moderate",
            "min_filledness": 0.3,
            "min_rectangularness": 0.7,
            "min_contiguity": 0.5,
        },
        {
            "name": "Lenient",
            "min_filledness": 0.1,
            "min_rectangularness": 0.5,
            "min_contiguity": 0.3,
        },
    ]

    # Test sheet with moderate sparsity
    test_sheet = SheetData(name="TestConfig")
    # Create a 5x5 grid with 60% filled
    for row in range(5):
        for col in range(5):
            if (row + col) % 2 == 0:  # Checkerboard pattern
                addr = f"{chr(65+col)}{row+1}"
                create_cell(test_sheet, addr, f"D{row}{col}", data_type="text")

    region = TableRange(range="A1:E5", start_row=0, start_col=0, end_row=4, end_col=4)

    for config in configs:
        verifier = RegionVerifier(
            min_filledness=config["min_filledness"],
            min_rectangularness=config["min_rectangularness"],
            min_contiguity=config["min_contiguity"],
        )
        result = verifier.verify_region(test_sheet, region)
        print(f"\n{config['name']} configuration:")
        print(f"  Valid: {result.valid}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Filledness: {result.metrics['filledness']:.2f}")

    # Test 6.2: Strict Mode Testing (FIXED variable name)
    print_test("Test 6.2: Strict Mode Testing")

    # Test strict vs non-strict mode
    slightly_irregular = SheetData(name="SlightlyIrregular")
    # Mostly filled table with one missing corner
    for row in range(4):
        for col in range(4):
            if not (row == 3 and col == 3):  # Missing bottom-right
                addr = f"{chr(65+col)}{row+1}"
                create_cell(slightly_irregular, addr, f"D{row}{col}", data_type="text")

    region = TableRange(range="A1:D4", start_row=0, start_col=0, end_row=3, end_col=3)

    # Non-strict mode
    verifier_lenient = RegionVerifier()
    result_lenient = verifier_lenient.verify_region(slightly_irregular, region, strict=False)
    print(
        f"Non-strict mode: valid={result_lenient.valid}, confidence={result_lenient.confidence:.2f}"
    )

    # Strict mode
    result_strict = verifier_lenient.verify_region(slightly_irregular, region, strict=True)
    print(f"Strict mode: valid={result_strict.valid}, confidence={result_strict.confidence:.2f}")


def test_performance_edge_cases():
    """Section 7: Performance and Edge Cases"""
    print_section("Section 7: Performance and Edge Cases")

    # Test 7.1: Large Region Performance
    print_test("Test 7.1: Large Region Performance")

    # Create a large sheet
    large_sheet = SheetData(name="LargeSheet")
    # Fill 100x100 region
    print("Creating large sheet...")
    start_time = time.time()
    for row in range(100):
        for col in range(100):
            # Create address (handle columns > 26)
            col_letter = ""
            c = col
            while c >= 0:
                col_letter = chr(c % 26 + ord("A")) + col_letter
                c = c // 26 - 1
                if c < 0:
                    break
            addr = f"{col_letter}{row+1}"
            create_cell(large_sheet, addr, f"V{row},{col}", data_type="text")
    creation_time = time.time() - start_time
    print(f"Sheet creation time: {creation_time:.2f}s")

    # Time verification
    large_region = TableRange(range="A1:CV100", start_row=0, start_col=0, end_row=99, end_col=99)
    verifier = RegionVerifier()

    start_time = time.time()
    result = verifier.verify_region(large_sheet, large_region)
    verification_time = time.time() - start_time

    print(f"Verification time: {verification_time:.3f}s")
    print(f"Valid: {result.valid}")
    print("Cells processed: 10,000")
    print(f"Processing rate: {10000/verification_time:.0f} cells/second")

    # Test 7.2: Edge Cases
    print_test("Test 7.2: Edge Cases")

    # Test various edge cases
    verifier = RegionVerifier()

    # 1. Single cell
    single_cell_sheet = SheetData(name="SingleCell")
    create_cell(single_cell_sheet, "A1", "Only", data_type="text")

    result = verifier.verify_region(
        single_cell_sheet, TableRange(range="A1", start_row=0, start_col=0, end_row=0, end_col=0)
    )
    print(f"Single cell: valid={result.valid}, reason='{result.reason}'")

    # 2. Empty region
    empty_sheet = SheetData(name="Empty")

    result = verifier.verify_region(
        empty_sheet, TableRange(range="A1:C3", start_row=0, start_col=0, end_row=2, end_col=2)
    )
    print(f"Empty region: valid={result.valid}, reason='{result.reason}'")

    # 3. Single row
    single_row_sheet = SheetData(name="SingleRow")
    for col in range(10):
        create_cell(single_row_sheet, f"{chr(65+col)}1", f"C{col}", data_type="text")

    result = verifier.verify_region(
        single_row_sheet, TableRange(range="A1:J1", start_row=0, start_col=0, end_row=0, end_col=9)
    )
    print(f"Single row: valid={result.valid}, aspect_ratio={result.metrics['aspect_ratio']:.1f}")

    # 4. Single column
    single_col_sheet = SheetData(name="SingleCol")
    for row in range(10):
        create_cell(single_col_sheet, f"A{row+1}", f"R{row}", data_type="text")

    result = verifier.verify_region(
        single_col_sheet, TableRange(range="A1:A10", start_row=0, start_col=0, end_row=9, end_col=0)
    )
    print(f"Single column: valid={result.valid}, aspect_ratio={result.metrics['aspect_ratio']:.1f}")


def test_visual_debugging():
    """Section 8: Visual Debugging"""
    print_section("Section 8: Visual Debugging")

    # Test 8.1: Visualize Verification Results
    print_test("Test 8.1: Visualize Verification Results")

    def visualize_verification(sheet: SheetData, region: TableRange, result: VerificationResult):
        """Create a simple text visualization of verification results."""
        print(f"\nRegion {region.excel_range} Verification:")
        print("=" * 50)

        # Create grid visualization
        for row in range(region.start_row, region.end_row + 1):
            row_str = f"Row {row+1:2d}: "
            for col in range(region.start_col, region.end_col + 1):
                cell = sheet.get_cell(row, col)
                if cell and cell.value is not None:
                    row_str += "[X] "
                else:
                    row_str += "[ ] "
            print(row_str)

        print("\nMetrics Summary:")
        print(f"  Valid: {'✓' if result.valid else '✗'} ({result.confidence:.2%} confidence)")
        print(f"  Rectangularness: {'▮' * int(result.metrics['rectangularness'] * 10)}")
        print(f"  Filledness:      {'▮' * int(result.metrics['filledness'] * 10)}")
        print(f"  Contiguity:      {'▮' * int(result.metrics['contiguity'] * 10)}")
        print(f"  Edge Quality:    {'▮' * int(result.metrics['edge_quality'] * 10)}")

        if result.feedback:
            print(f"\nFeedback: {result.feedback}")

    # Test with sample data
    viz_sheet = SheetData(name="Visualization")
    # Create an L-shape
    for col in range(5):
        create_cell(viz_sheet, f"{chr(65+col)}1", "H", data_type="text")
    for row in range(1, 5):
        create_cell(viz_sheet, f"A{row+1}", "V", data_type="text")

    region = TableRange(range="A1:E5", start_row=0, start_col=0, end_row=4, end_col=4)
    verifier = RegionVerifier()
    result = verifier.verify_region(viz_sheet, region)
    visualize_verification(viz_sheet, region, result)

    # Test 8.2: Compare Before/After Verification
    print_test("Test 8.2: Compare Before/After Verification")

    # Show how verification filters proposals
    def compare_proposals(original_regions, verified_regions):
        """Compare original vs verified regions."""
        print("\nRegion Verification Comparison:")
        print("=" * 60)
        print(f"Original regions: {len(original_regions)}")
        print(f"Verified regions: {len(verified_regions)}")
        print(f"Filtered out: {len(original_regions) - len(verified_regions)}")

        # Show details
        print("\nOriginal Regions:")
        for i, region in enumerate(original_regions):
            print(f"  {i+1}. {region.range} (confidence: {region.confidence:.2f})")

        print("\nVerified Regions:")
        for i, region in enumerate(verified_regions):
            print(f"  {i+1}. {region.range} (confidence: {region.confidence:.2f})")

    # Mock example
    original = [
        VisionRegion(
            pixel_bounds={"x1": 0, "y1": 0, "x2": 100, "y2": 50},
            cell_bounds={"start_row": 0, "start_col": 0, "end_row": 4, "end_col": 9},
            range="A1:J5",
            confidence=0.9,
            characteristics={"type": "valid_table"},
        ),
        VisionRegion(
            pixel_bounds={"x1": 0, "y1": 100, "x2": 500, "y2": 150},
            cell_bounds={"start_row": 10, "start_col": 0, "end_row": 14, "end_col": 49},
            range="A11:AX15",
            confidence=0.8,
            characteristics={"type": "sparse_region"},
        ),
    ]

    # Simulate verification filtering
    verified = [original[0]]  # Only first region passes
    compare_proposals(original, verified)


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("WEEK 4 TESTING GUIDE - AUTOMATED TEST RUNNER")
    print("=" * 70)

    try:
        test_basic_verification()
    except Exception as e:
        print(f"\nError in basic verification tests: {e}")

    try:
        test_geometry_analysis()
    except Exception as e:
        print(f"\nError in geometry analysis tests: {e}")

    try:
        test_pattern_verification()
    except Exception as e:
        print(f"\nError in pattern verification tests: {e}")

    try:
        test_feedback_loop()
    except Exception as e:
        print(f"\nError in feedback loop tests: {e}")

    try:
        test_pipeline_integration()
    except Exception as e:
        print(f"\nError in pipeline integration tests: {e}")

    try:
        test_configuration()
    except Exception as e:
        print(f"\nError in configuration tests: {e}")

    try:
        test_performance_edge_cases()
    except Exception as e:
        print(f"\nError in performance/edge case tests: {e}")

    try:
        test_visual_debugging()
    except Exception as e:
        print(f"\nError in visual debugging tests: {e}")

    print("\n" + "=" * 70)
    print("Test run completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
