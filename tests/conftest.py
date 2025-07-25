"""Pytest configuration and shared fixtures."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gridporter.config import Config
from gridporter.models.sheet_data import CellData, SheetData
from gridporter.vision.vision_models import VisionModelResponse


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_excel_file(fixtures_dir: Path) -> Path:
    """Path to sample Excel file for testing."""
    # This would be a real Excel file in a full implementation
    return fixtures_dir / "sample_data.xlsx"


@pytest.fixture
def sample_csv_file(fixtures_dir: Path) -> Path:
    """Path to sample CSV file for testing."""
    return fixtures_dir / "sample_data.csv"


# Vision-specific fixtures


@pytest.fixture
def sample_sheet_data() -> SheetData:
    """Create sample sheet data for vision testing."""
    sheet = SheetData(name="SampleSheet")
    sheet.cells["A1"] = CellData(value="Name", data_type="text", is_bold=True, row=0, column=0)
    sheet.cells["B1"] = CellData(value="Age", data_type="text", is_bold=True, row=0, column=1)
    sheet.cells["C1"] = CellData(value="City", data_type="text", is_bold=True, row=0, column=2)
    sheet.cells["A2"] = CellData(value="Alice", data_type="text", row=1, column=0)
    sheet.cells["B2"] = CellData(value=25, data_type="number", row=1, column=1)
    sheet.cells["C2"] = CellData(value="New York", data_type="text", row=1, column=2)
    sheet.cells["A3"] = CellData(value="Bob", data_type="text", row=2, column=0)
    sheet.cells["B3"] = CellData(value=30, data_type="number", row=2, column=1)
    sheet.cells["C3"] = CellData(value="London", data_type="text", row=2, column=2)
    sheet.max_row = 2
    sheet.max_column = 2
    return sheet


@pytest.fixture
def large_sheet_data() -> SheetData:
    """Create large sheet data for scaling tests."""
    sheet = SheetData(name="LargeSheet")

    # Create a 100x50 sheet with sparse data
    for row in range(100):
        for col in range(50):
            if (row + col) % 5 == 0:  # Sparse pattern
                col_letter = chr(65 + col % 26)
                if col >= 26:
                    col_letter = chr(65 + col // 26 - 1) + col_letter
                addr = f"{col_letter}{row + 1}"
                sheet.cells[addr] = CellData(
                    value=f"Cell_{row}_{col}", data_type="text", row=row, column=col
                )

    sheet.max_row = 99
    sheet.max_column = 49
    return sheet


@pytest.fixture
def vision_config() -> Config:
    """Create vision-specific configuration for testing."""
    return Config(
        vision_cell_width=12,
        vision_cell_height=10,
        vision_mode="binary",
        confidence_threshold=0.7,
        min_table_size=(2, 2),
        enable_cache=True,
    )


@pytest.fixture
def mock_openai_response() -> dict:
    """Mock OpenAI vision model response."""
    return {
        "choices": [
            {
                "message": {
                    "content": '{"tables": [{"bounds": {"x1": 0, "y1": 0, "x2": 100, "y2": 50}, "confidence": 0.9, "characteristics": {"has_headers": true}}]}'
                }
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
    }


@pytest.fixture
def mock_ollama_response() -> dict:
    """Mock Ollama vision model response."""
    return {
        "response": '{"tables": [{"bounds": {"x1": 10, "y1": 10, "x2": 90, "y2": 40}, "confidence": 0.85}]}',
        "eval_duration": 1500000000,  # 1.5 seconds in nanoseconds
        "total_duration": 2000000000,  # 2 seconds in nanoseconds
    }


@pytest.fixture
def mock_vision_model_response() -> VisionModelResponse:
    """Create a mock VisionModelResponse for testing."""
    return VisionModelResponse(
        content='{"tables": [{"bounds": {"x1": 0, "y1": 0, "x2": 60, "y2": 30}, "confidence": 0.8, "characteristics": {"has_headers": true}}]}',
        model="test-vision-model",
        usage={"total_tokens": 100},
    )


@pytest.fixture
def mock_async_vision_model(mock_vision_model_response: VisionModelResponse) -> AsyncMock:
    """Create a mock async vision model for testing."""
    mock_model = AsyncMock()
    mock_model.name = "mock-vision-model"
    mock_model.supports_batch = False
    mock_model.analyze_image.return_value = mock_vision_model_response
    return mock_model


@pytest.fixture
def mock_openai_client(mock_openai_response: dict) -> MagicMock:
    """Create a mock OpenAI client for testing."""
    mock_client = AsyncMock()

    # Mock the response structure
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = mock_openai_response["choices"][0]["message"][
        "content"
    ]
    mock_response.usage.prompt_tokens = mock_openai_response["usage"]["prompt_tokens"]
    mock_response.usage.completion_tokens = mock_openai_response["usage"]["completion_tokens"]
    mock_response.usage.total_tokens = mock_openai_response["usage"]["total_tokens"]

    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_httpx_response(mock_ollama_response: dict) -> MagicMock:
    """Create a mock httpx response for Ollama testing."""
    mock_response = MagicMock()
    mock_response.json.return_value = mock_ollama_response
    mock_response.raise_for_status.return_value = None
    return mock_response


@pytest.fixture
def complex_sheet_data() -> SheetData:
    """Create complex sheet data with various cell types for comprehensive testing."""
    sheet = SheetData(name="ComplexSheet")

    # Header row
    sheet.cells["A1"] = CellData(value="Product", data_type="text", is_bold=True, row=0, column=0)
    sheet.cells["B1"] = CellData(value="Price", data_type="text", is_bold=True, row=0, column=1)
    sheet.cells["C1"] = CellData(value="Quantity", data_type="text", is_bold=True, row=0, column=2)
    sheet.cells["D1"] = CellData(value="Total", data_type="text", is_bold=True, row=0, column=3)
    sheet.cells["E1"] = CellData(value="Date", data_type="text", is_bold=True, row=0, column=4)

    # Data rows with various types
    sheet.cells["A2"] = CellData(value="Widget A", data_type="text", row=1, column=0)
    sheet.cells["B2"] = CellData(value=19.99, data_type="number", row=1, column=1)
    sheet.cells["C2"] = CellData(value=5, data_type="number", row=1, column=2)
    sheet.cells["D2"] = CellData(
        value="=B2*C2", data_type="formula", has_formula=True, row=1, column=3
    )
    sheet.cells["E2"] = CellData(value="2024-01-15", data_type="date", row=1, column=4)

    sheet.cells["A3"] = CellData(value="Widget B", data_type="text", row=2, column=0)
    sheet.cells["B3"] = CellData(value=29.99, data_type="number", row=2, column=1)
    sheet.cells["C3"] = CellData(value=3, data_type="number", row=2, column=2)
    sheet.cells["D3"] = CellData(
        value="=B3*C3", data_type="formula", has_formula=True, row=2, column=3
    )
    sheet.cells["E3"] = CellData(value="2024-01-16", data_type="date", row=2, column=4)

    # Merged cell
    sheet.cells["A5"] = CellData(
        value="Summary", data_type="text", is_merged=True, is_bold=True, row=4, column=0
    )

    sheet.max_row = 5
    sheet.max_column = 4
    return sheet
