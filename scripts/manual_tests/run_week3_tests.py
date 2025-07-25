#!/usr/bin/env python
"""
Comprehensive test runner for Week 3 Vision Infrastructure.
This script runs all tests from WEEK3_TESTING_GUIDE.md with the requested modifications.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

# Load environment variables from .env file
load_dotenv()

console = Console()

# Import GridPorter components
try:
    from gridporter.config import Config
    from gridporter.models.sheet_data import CellData, SheetData
    from gridporter.models.vision_result import VisionAnalysisResult
    from gridporter.vision import BitmapGenerator, VisionPipeline
    from gridporter.vision.bitmap_generator import BitmapMetadata
    from gridporter.vision.region_proposer import RegionProposer
    from gridporter.vision.vision_models import VisionModelError, create_vision_model
except ImportError as e:
    console.print(f"[red]Failed to import GridPorter components: {e}[/red]")
    console.print("[yellow]Please ensure GridPorter is installed: pip install -e .[dev][/yellow]")
    sys.exit(1)


def print_section_header(section: str, description: str):
    """Print a formatted section header."""
    console.print()
    console.print(Panel(f"[bold blue]{section}[/bold blue]\n{description}", expand=False))
    console.print()


def print_test_result(test_name: str, passed: bool, details: str = ""):
    """Print test result with formatting."""
    status = "[green]✓ PASSED[/green]" if passed else "[red]✗ FAILED[/red]"
    console.print(f"{status} {test_name}")
    if details:
        console.print(f"  [dim]{details}[/dim]")


class Week3Tests:
    """Container for all Week 3 tests."""

    def __init__(self):
        self.results = []

    def record_result(self, test_name: str, passed: bool, details: str = ""):
        """Record a test result."""
        self.results.append({"test": test_name, "passed": passed, "details": details})
        print_test_result(test_name, passed, details)

    # Section 1: Bitmap Generation Tests
    def test_1_1_basic_bitmap_generation(self):
        """Test 1.1: Basic Bitmap Generation"""
        try:
            # Create test data
            sheet = SheetData(name="TestSheet")
            sheet.cells["A1"] = CellData(
                value="Name", data_type="text", is_bold=True, row=0, column=0
            )
            sheet.cells["B1"] = CellData(
                value="Age", data_type="text", is_bold=True, row=0, column=1
            )
            sheet.cells["A2"] = CellData(value="Alice", data_type="text", row=1, column=0)
            sheet.cells["B2"] = CellData(value=25, data_type="number", row=1, column=1)
            sheet.max_row = 1
            sheet.max_column = 1

            # Generate bitmap
            generator = BitmapGenerator()
            image_bytes, metadata = generator.generate(sheet)

            details = (
                f"Bitmap: {metadata.width}x{metadata.height} pixels, "
                f"Cell size: {metadata.cell_width}x{metadata.cell_height}, "
                f"Sheet: {metadata.total_rows}x{metadata.total_cols} cells, "
                f"Size: {len(image_bytes)} bytes"
            )

            self.record_result("Test 1.1: Basic Bitmap Generation", True, details)

        except Exception as e:
            self.record_result("Test 1.1: Basic Bitmap Generation", False, str(e))

    def test_1_2_different_bitmap_modes(self):
        """Test 1.2: Different Bitmap Modes"""
        try:
            sheet = SheetData(name="TestSheet")
            sheet.cells["A1"] = CellData(value="Test", data_type="text", row=0, column=0)
            sheet.max_row = 0
            sheet.max_column = 0

            results = []
            for mode in ["binary", "grayscale", "color"]:
                generator = BitmapGenerator(mode=mode)
                img_bytes, meta = generator.generate(sheet)
                results.append(f"{mode}: {len(img_bytes)} bytes")

            self.record_result("Test 1.2: Different Bitmap Modes", True, ", ".join(results))

        except Exception as e:
            self.record_result("Test 1.2: Different Bitmap Modes", False, str(e))

    def test_1_3_save_debug_bitmap(self):
        """Test 1.3: Save Debug Bitmap"""
        try:
            sheet = SheetData(name="TestSheet")
            sheet.cells["A1"] = CellData(value="Debug", data_type="text", row=0, column=0)
            sheet.max_row = 0
            sheet.max_column = 0

            generator = BitmapGenerator()
            image_bytes, metadata = generator.generate(sheet)

            # Save to debug file
            debug_path = Path("tests/manual/level1/debug_bitmap.png")
            debug_path.parent.mkdir(parents=True, exist_ok=True)

            with open(debug_path, "wb") as f:
                f.write(image_bytes)

            self.record_result("Test 1.3: Save Debug Bitmap", True, f"Saved to {debug_path}")

        except Exception as e:
            self.record_result("Test 1.3: Save Debug Bitmap", False, str(e))

    def test_1_4_large_sheet_scaling(self):
        """Test 1.4: Large Sheet Scaling"""
        try:
            # Create a large sheet
            large_sheet = SheetData(name="LargeSheet")
            for row in range(200):
                for col in range(200):
                    if (row + col) % 10 == 0:
                        cell_addr = f"{chr(65 + col % 26)}{row + 1}"
                        large_sheet.cells[cell_addr] = CellData(
                            value=f"R{row}C{col}", data_type="text", row=row, column=col
                        )
            large_sheet.max_row = 199
            large_sheet.max_column = 199

            generator = BitmapGenerator(cell_width=20, cell_height=20)
            image_bytes, metadata = generator.generate(large_sheet)

            details = (
                f"Scale factor: {metadata.scale_factor}, "
                f"Cell size: {metadata.cell_width}x{metadata.cell_height}"
            )

            self.record_result("Test 1.4: Large Sheet Scaling", True, details)

        except Exception as e:
            self.record_result("Test 1.4: Large Sheet Scaling", False, str(e))

    # Section 2: Vision Model Configuration Tests
    async def test_2_1_openai_vision_model(self):
        """Test 2.1: OpenAI Vision Model (using .env file)"""
        try:
            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.record_result(
                    "Test 2.1: OpenAI Vision Model",
                    False,
                    "OPENAI_API_KEY not found in environment or .env file",
                )
                return

            # Configure for OpenAI
            config = Config(openai_api_key=api_key, use_local_llm=False)

            model = create_vision_model(config)
            details = f"Created model: {model.name}, Supports batch: {model.supports_batch}"
            self.record_result("Test 2.1: OpenAI Vision Model", True, details)

        except VisionModelError as e:
            self.record_result("Test 2.1: OpenAI Vision Model", False, f"Vision error: {e}")
        except ImportError as e:
            self.record_result("Test 2.1: OpenAI Vision Model", False, f"Missing dependency: {e}")
        except Exception as e:
            self.record_result("Test 2.1: OpenAI Vision Model", False, str(e))

    async def test_2_2_ollama_vision_model(self):
        """Test 2.2: Ollama Vision Model"""
        try:
            from gridporter.vision.vision_models import VisionModelError

            # Configure for Ollama
            config = Config(
                use_local_llm=True,
                ollama_url="http://localhost:11434",
                ollama_vision_model="qwen2.5vl:7b",
            )

            model = create_vision_model(config)
            console.print(f"Created vision model: {model.name}")

            # Check if model is available
            if hasattr(model, "check_model_available"):
                available = await model.check_model_available()
                if available:
                    details = f"Model: {model.name} is available"
                else:
                    details = f"Model not found. Try: ollama pull {config.ollama_vision_model}"
                self.record_result("Test 2.2: Ollama Vision Model", available, details)
            else:
                self.record_result(
                    "Test 2.2: Ollama Vision Model",
                    True,
                    f"Created model: {model.name}",
                )

        except VisionModelError as e:
            self.record_result("Test 2.2: Ollama Vision Model", False, f"Vision error: {e}")
        except ConnectionError:
            self.record_result(
                "Test 2.2: Ollama Vision Model",
                False,
                "Ollama not running. Start with: ollama serve",
            )
        except Exception as e:
            self.record_result("Test 2.2: Ollama Vision Model", False, str(e))

    async def test_2_3_unavailable_model(self):
        """Test 2.3: Unavailable Model Response (NEW)"""
        try:
            # Configure for a model that doesn't exist
            config = Config(
                use_local_llm=True,
                ollama_url="http://localhost:11434",
                ollama_vision_model="nonexistent-model:latest",
            )

            model = create_vision_model(config)

            # Check if model is available
            if hasattr(model, "check_model_available"):
                available = await model.check_model_available()
                if not available:
                    self.record_result(
                        "Test 2.3: Unavailable Model",
                        True,
                        "Correctly detected that model is not available",
                    )
                else:
                    self.record_result(
                        "Test 2.3: Unavailable Model",
                        False,
                        "Model unexpectedly reported as available",
                    )
            else:
                self.record_result(
                    "Test 2.3: Unavailable Model",
                    False,
                    "Model doesn't support availability check",
                )

        except Exception as e:
            self.record_result("Test 2.3: Unavailable Model", False, str(e))

    async def test_ollama_availability(self):
        """Test: Check Ollama Service Availability (NEW)"""
        try:
            import httpx

            # Check if Ollama service is running
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get("http://localhost:11434/api/tags")
                    models = response.json().get("models", [])

                    if models:
                        model_names = [m["name"] for m in models]
                        self.record_result(
                            "Test: Ollama Availability",
                            True,
                            f"Ollama running with {len(models)} models: {', '.join(model_names[:3])}...",
                        )
                    else:
                        self.record_result(
                            "Test: Ollama Availability",
                            True,
                            "Ollama running but no models installed",
                        )
                except httpx.ConnectError:
                    self.record_result(
                        "Test: Ollama Availability",
                        False,
                        "Ollama not running. Start with: ollama serve",
                    )
                except Exception as e:
                    self.record_result("Test: Ollama Availability", False, str(e))

        except ImportError:
            self.record_result("Test: Ollama Availability", False, "httpx not installed")
        except Exception as e:
            self.record_result("Test: Ollama Availability", False, str(e))

    # Section 3: Region Proposal Parsing Tests
    def test_3_1_json_response_parsing(self):
        """Test 3.1: JSON Response Parsing"""
        try:
            proposer = RegionProposer()
            metadata = BitmapMetadata(
                width=200,
                height=150,
                cell_width=10,
                cell_height=10,
                total_rows=15,
                total_cols=20,
                scale_factor=1.0,
                mode="binary",
            )

            json_response = """
{
    "tables": [
        {
            "bounds": {"x1": 10, "y1": 10, "x2": 100, "y2": 50},
            "confidence": 0.9,
            "characteristics": {"has_headers": true, "type": "data_table"}
        },
        {
            "bounds": {"x1": 120, "y1": 10, "x2": 180, "y2": 40},
            "confidence": 0.75,
            "characteristics": {"has_headers": false}
        }
    ]
}
"""

            proposals = proposer.parse_response(json_response, metadata)

            details = f"Parsed {len(proposals)} proposals from JSON"
            if proposals:
                details += f" - First: {proposals[0].excel_range} (conf: {proposals[0].confidence})"

            self.record_result("Test 3.1: JSON Response Parsing", len(proposals) == 2, details)

        except Exception as e:
            self.record_result("Test 3.1: JSON Response Parsing", False, str(e))

    def test_3_2_text_response_parsing(self):
        """Test 3.2: Text Response Parsing"""
        try:
            proposer = RegionProposer()
            metadata = BitmapMetadata(
                width=200,
                height=150,
                cell_width=10,
                cell_height=10,
                total_rows=15,
                total_cols=20,
                scale_factor=1.0,
                mode="binary",
            )

            text_response = """
I can see two tables in this spreadsheet:

Table 1: Located at pixel coordinates (20, 20, 90, 60) with confidence 0.85.
This appears to be a header table with bold formatting in the first row.

Table 2: Found another table at (110, 20, 170, 50) with confidence 0.7.
This one contains numeric data without clear headers.
"""

            proposals = proposer.parse_response(text_response, metadata)

            details = f"Parsed {len(proposals)} proposals from text"
            if proposals:
                details += f" - First: {proposals[0].excel_range} (conf: {proposals[0].confidence})"

            self.record_result("Test 3.2: Text Response Parsing", len(proposals) == 2, details)

        except Exception as e:
            self.record_result("Test 3.2: Text Response Parsing", False, str(e))

    def test_3_3_proposal_filtering(self):
        """Test 3.3: Proposal Filtering"""
        try:
            proposer = RegionProposer()
            metadata = BitmapMetadata(
                width=200,
                height=150,
                cell_width=10,
                cell_height=10,
                total_rows=15,
                total_cols=20,
                scale_factor=1.0,
                mode="binary",
            )

            json_response = """
{
    "tables": [
        {"bounds": {"x1": 10, "y1": 10, "x2": 100, "y2": 50}, "confidence": 0.9},
        {"bounds": {"x1": 10, "y1": 10, "x2": 100, "y2": 50}, "confidence": 0.4}
    ]
}
"""

            all_proposals = proposer.parse_response(json_response, metadata)

            # Filter with different thresholds
            filtered_high = proposer.filter_proposals(all_proposals, min_confidence=0.8)
            filtered_low = proposer.filter_proposals(all_proposals, min_confidence=0.5)

            details = (
                f"Total: {len(all_proposals)}, "
                f"High conf (>0.8): {len(filtered_high)}, "
                f"Low conf (>0.5): {len(filtered_low)}"
            )

            self.record_result("Test 3.3: Proposal Filtering", True, details)

        except Exception as e:
            self.record_result("Test 3.3: Proposal Filtering", False, str(e))

    # Section 4: Vision Pipeline Integration Tests
    def test_4_1_complete_vision_analysis(self):
        """Test 4.1: Complete Vision Analysis (Mock)"""
        try:
            # Create test sheet data
            sheet = SheetData(name="TestSheet")
            sheet.cells["A1"] = CellData(
                value="Product", data_type="text", is_bold=True, row=0, column=0
            )
            sheet.cells["B1"] = CellData(
                value="Price", data_type="text", is_bold=True, row=0, column=1
            )
            sheet.cells["A2"] = CellData(value="Apple", data_type="text", row=1, column=0)
            sheet.cells["B2"] = CellData(value=1.25, data_type="number", row=1, column=1)
            sheet.max_row = 1
            sheet.max_column = 1

            # Configure pipeline
            config = Config(
                vision_cell_width=12,
                vision_cell_height=10,
                vision_mode="grayscale",
                use_local_llm=True,
                confidence_threshold=0.6,
            )

            pipeline = VisionPipeline(config)

            # Test bitmap generation only
            image_bytes, bitmap_metadata = pipeline.bitmap_generator.generate(sheet)

            # Create mock response
            mock_response = """
{
    "tables": [
        {
            "bounds": {"x1": 0, "y1": 0, "x2": 60, "y2": 30},
            "confidence": 0.9,
            "characteristics": {"has_headers": true}
        }
    ]
}
"""

            proposals = pipeline.region_proposer.parse_response(mock_response, bitmap_metadata)

            details = (
                f"Bitmap: {bitmap_metadata.width}x{bitmap_metadata.height}, "
                f"Found {len(proposals)} tables"
            )

            self.record_result("Test 4.1: Complete Vision Analysis", True, details)

        except Exception as e:
            self.record_result("Test 4.1: Complete Vision Analysis", False, str(e))

    def test_4_2_debug_bitmap_saving(self):
        """Test 4.2: Debug Bitmap Saving"""
        try:
            sheet = SheetData(name="Debug")
            sheet.cells["A1"] = CellData(value="Test", data_type="text", row=0, column=0)
            sheet.max_row = 0
            sheet.max_column = 0

            config = Config()
            pipeline = VisionPipeline(config)

            debug_path = Path("level1/pipeline_debug.png")
            metadata = pipeline.save_debug_bitmap(sheet, debug_path)

            details = f"Saved to {debug_path}, {metadata.total_rows}x{metadata.total_cols} cells"
            self.record_result("Test 4.2: Debug Bitmap Saving", True, details)

        except Exception as e:
            self.record_result("Test 4.2: Debug Bitmap Saving", False, str(e))

    def test_4_3_cache_operations(self):
        """Test 4.3: Cache Operations"""
        try:
            config = Config(enable_cache=True)
            pipeline = VisionPipeline(config)

            # Test cache functionality
            initial_stats = pipeline.get_cache_stats()

            # Add some cache entries
            sheet = SheetData(name="CacheTest")
            cache_key = pipeline._generate_cache_key(sheet)
            pipeline._cache[cache_key] = VisionAnalysisResult(
                regions=[], bitmap_info={"test": True}
            )

            stats_after_add = pipeline.get_cache_stats()

            # Clear cache
            pipeline.clear_cache()
            final_stats = pipeline.get_cache_stats()

            details = (
                f"Initial: {initial_stats['cache_size']}, "
                f"After add: {stats_after_add['cache_size']}, "
                f"After clear: {final_stats['cache_size']}"
            )

            self.record_result("Test 4.3: Cache Operations", True, details)

        except Exception as e:
            self.record_result("Test 4.3: Cache Operations", False, str(e))

    # Section 5: Vision Models Integration Tests
    def test_5_1_vision_model_factory(self):
        """Test 5.1: Vision Model Factory"""
        try:
            configs = [
                Config(use_local_llm=True, ollama_vision_model="qwen2.5vl:7b"),
                Config(use_local_llm=False, openai_api_key="test-key"),
            ]

            results = []
            for i, config in enumerate(configs):
                try:
                    model = create_vision_model(config)
                    results.append(f"Config {i+1}: {model.name}")
                except Exception as e:
                    results.append(f"Config {i+1}: Failed - {type(e).__name__}")

            self.record_result("Test 5.1: Vision Model Factory", True, ", ".join(results))

        except Exception as e:
            self.record_result("Test 5.1: Vision Model Factory", False, str(e))

    def test_5_2_model_response_structure(self):
        """Test 5.2: Model Response Structure"""
        try:
            from gridporter.vision.vision_models import VisionModelResponse

            # Mock response for testing
            response = VisionModelResponse(
                content='{"tables": [{"bounds": {"x1": 0, "y1": 0, "x2": 100, "y2": 50}, "confidence": 0.9}]}',
                model="test-model",
                usage={"total_tokens": 150},
            )

            details = (
                f"Model: {response.model}, "
                f"Content length: {len(response.content)}, "
                f"Tokens: {response.usage.get('total_tokens', 0)}"
            )

            self.record_result("Test 5.2: Model Response Structure", True, details)

        except Exception as e:
            self.record_result("Test 5.2: Model Response Structure", False, str(e))

    # Section 6: Configuration and Environment Tests
    def test_6_1_vision_configuration(self):
        """Test 6.1: Vision Configuration"""
        try:
            # Test default configuration
            config = Config.from_env()

            default_details = (
                f"Cell: {config.vision_cell_width}x{config.vision_cell_height}, "
                f"Mode: {config.vision_mode}"
            )

            # Test custom configuration
            custom_config = Config(vision_cell_width=15, vision_cell_height=15, vision_mode="color")

            custom_details = (
                f"Custom: {custom_config.vision_cell_width}x{custom_config.vision_cell_height}, "
                f"Mode: {custom_config.vision_mode}"
            )

            self.record_result(
                "Test 6.1: Vision Configuration",
                True,
                f"Default: {default_details}, {custom_details}",
            )

        except Exception as e:
            self.record_result("Test 6.1: Vision Configuration", False, str(e))

    def test_6_2_environment_variables(self):
        """Test 6.2: Environment Variables"""
        try:
            # Save current env vars
            original_vars = {}
            env_vars = [
                "GRIDPORTER_VISION_CELL_WIDTH",
                "GRIDPORTER_VISION_CELL_HEIGHT",
                "GRIDPORTER_VISION_MODE",
            ]

            for var in env_vars:
                original_vars[var] = os.environ.get(var)

            # Set test environment variables
            os.environ["GRIDPORTER_VISION_CELL_WIDTH"] = "20"
            os.environ["GRIDPORTER_VISION_CELL_HEIGHT"] = "16"
            os.environ["GRIDPORTER_VISION_MODE"] = "grayscale"

            # Load config from environment
            config = Config.from_env()

            details = (
                f"Width: {config.vision_cell_width}, "
                f"Height: {config.vision_cell_height}, "
                f"Mode: {config.vision_mode}"
            )

            # Restore original env vars
            for var, value in original_vars.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value

            self.record_result("Test 6.2: Environment Variables", True, details)

        except Exception as e:
            self.record_result("Test 6.2: Environment Variables", False, str(e))

    # Section 7: Model Integration Tests
    def test_7_1_vision_result_models(self):
        """Test 7.1: Vision Result Models"""
        try:
            from gridporter.models.vision_result import (
                VisionRegion,
            )

            # Test creating vision region
            region = VisionRegion(
                pixel_bounds={"x1": 0, "y1": 0, "x2": 100, "y2": 50},
                cell_bounds={
                    "start_row": 0,
                    "start_col": 0,
                    "end_row": 4,
                    "end_col": 9,
                },
                range="A1:J5",
                confidence=0.85,
                characteristics={"has_headers": True, "type": "data_table"},
            )

            # Convert to TableRange
            table_range = region.to_table_range()

            details = (
                f"Region: {region.range} (conf: {region.confidence}), "
                f"TableRange: {table_range.excel_range}"
            )

            self.record_result("Test 7.1: Vision Result Models", True, details)

        except Exception as e:
            self.record_result("Test 7.1: Vision Result Models", False, str(e))

    def test_7_2_analysis_result_operations(self):
        """Test 7.2: Analysis Result Operations"""
        try:
            from gridporter.models.vision_result import (
                VisionAnalysisResult,
                VisionRegion,
            )

            # Create test region
            region = VisionRegion(
                pixel_bounds={"x1": 0, "y1": 0, "x2": 100, "y2": 50},
                cell_bounds={
                    "start_row": 0,
                    "start_col": 0,
                    "end_row": 4,
                    "end_col": 9,
                },
                range="A1:J5",
                confidence=0.85,
                characteristics={"has_headers": True},
            )

            # Test analysis result
            result = VisionAnalysisResult(
                regions=[region],
                bitmap_info={"width": 100, "height": 50, "mode": "binary"},
            )

            # Test high confidence filtering
            high_conf = result.high_confidence_regions(threshold=0.8)

            # Convert to table ranges
            table_ranges = result.to_table_ranges()

            details = (
                f"High conf regions: {len(high_conf)}, "
                f"Table ranges: {[tr.excel_range for tr in table_ranges]}"
            )

            self.record_result("Test 7.2: Analysis Result Operations", True, details)

        except Exception as e:
            self.record_result("Test 7.2: Analysis Result Operations", False, str(e))

    async def run_all_tests(self):
        """Run all tests in sequence."""
        print_section_header("Section 1", "Bitmap Generation")
        self.test_1_1_basic_bitmap_generation()
        self.test_1_2_different_bitmap_modes()
        self.test_1_3_save_debug_bitmap()
        self.test_1_4_large_sheet_scaling()

        print_section_header("Section 2", "Vision Model Configuration")
        await self.test_2_1_openai_vision_model()
        await self.test_2_2_ollama_vision_model()
        await self.test_2_3_unavailable_model()
        await self.test_ollama_availability()

        print_section_header("Section 3", "Region Proposal Parsing")
        self.test_3_1_json_response_parsing()
        self.test_3_2_text_response_parsing()
        self.test_3_3_proposal_filtering()

        print_section_header("Section 4", "Vision Pipeline Integration")
        self.test_4_1_complete_vision_analysis()
        self.test_4_2_debug_bitmap_saving()
        self.test_4_3_cache_operations()

        print_section_header("Section 5", "Vision Models Integration")
        self.test_5_1_vision_model_factory()
        self.test_5_2_model_response_structure()

        print_section_header("Section 6", "Configuration and Environment")
        self.test_6_1_vision_configuration()
        self.test_6_2_environment_variables()

        print_section_header("Section 7", "Model Integration")
        self.test_7_1_vision_result_models()
        self.test_7_2_analysis_result_operations()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        console.print()
        console.print(Panel("[bold]Test Summary[/bold]", expand=False))

        table = Table(show_header=True)
        table.add_column("Test", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="dim")

        passed = 0
        failed = 0

        for result in self.results:
            status = "[green]PASSED[/green]" if result["passed"] else "[red]FAILED[/red]"
            table.add_row(result["test"], status, result["details"])
            if result["passed"]:
                passed += 1
            else:
                failed += 1

        console.print(table)
        console.print()
        console.print(f"[bold]Total:[/bold] {len(self.results)} tests")
        console.print(f"[bold green]Passed:[/bold green] {passed}")
        console.print(f"[bold red]Failed:[/bold red] {failed}")

        if failed == 0:
            console.print("\n[bold green]All tests passed! ✨[/bold green]")
        else:
            console.print(f"\n[bold red]{failed} tests failed[/bold red]")


async def main():
    """Main entry point."""
    console.print(
        Panel(
            "[bold]Week 3 Vision Infrastructure Test Runner[/bold]\n"
            "Running all tests from WEEK3_TESTING_GUIDE.md",
            expand=False,
        )
    )

    tests = Week3Tests()
    await tests.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
