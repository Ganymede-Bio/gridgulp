"""Tests for the main GridGulp class."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import tempfile

import pytest

from gridgulp import GridGulp
from gridgulp.config import Config, DetectionStrategy
from gridgulp.models import (
    DetectionResult,
    EntitySchema,
    ExtractionOrientation,
    FileInfo,
    FileType,
    FieldSchema,
    RecordGroup,
    SheetResult,
    SheetSchemaResult,
    TableInfo,
    TableRange,
    WorkbookSchemaResult,
)
from gridgulp.models.sheet_data import SheetData, FileData
from gridgulp.models.file_info import UnsupportedFormatError


@pytest.fixture
def heuristic_config():
    """Create a heuristic-only config for tests that do not need Gemma."""
    return Config(detection_strategy=DetectionStrategy.HEURISTIC_ONLY)


@pytest.fixture
def mock_file_info():
    """Create mock FileInfo."""
    return FileInfo(
        path=Path("test.xlsx"),
        type=FileType.XLSX,
        size=1024,
        encoding="utf-8",
    )


@pytest.fixture
def mock_sheet_data():
    """Create mock SheetData."""
    sheet = SheetData(name="Sheet1")
    # Add some mock cells
    from gridgulp.models.sheet_data import CellData

    sheet.cells[(0, 0)] = CellData(row=0, column=0, value="Name", data_type="s")
    sheet.cells[(0, 1)] = CellData(row=0, column=1, value="Age", data_type="s")
    sheet.cells[(1, 0)] = CellData(row=1, column=0, value="John", data_type="s")
    sheet.cells[(1, 1)] = CellData(row=1, column=1, value=25, data_type="n")
    return sheet


@pytest.fixture
def mock_detection_result(mock_file_info):
    """Create mock DetectionResult."""
    return DetectionResult(
        file_info=mock_file_info,
        sheets=[
            SheetResult(
                name="Sheet1",
                tables=[
                    TableInfo(
                        id="table_1",
                        range=TableRange(start_row=0, start_col=0, end_row=9, end_col=3),
                        confidence=0.95,
                        detection_method="simple_case",
                        suggested_name="Table1",
                        headers=["Name", "Age", "City", "Country"],
                    )
                ],
                processing_time=0.1,
            )
        ],
        detection_time=0.5,
        total_tables=1,
        methods_used=["simple_case"],
    )


@pytest.fixture
def mock_schema_result(mock_file_info):
    """Create mock WorkbookSchemaResult."""
    return WorkbookSchemaResult(
        file_info=mock_file_info,
        workbook_metadata={"file_format": "xlsx"},
        sheet_results=[SheetSchemaResult(sheet_name="Sheet1", entity_ids=["table_1"])],
        entities=[
            EntitySchema(
                id="table_1",
                suggested_name="Table 1",
                source_regions=["Sheet1::table_1"],
                sheet_names=["Sheet1"],
                fields=[
                    FieldSchema(
                        id="name",
                        display_name="Name",
                        semantic_type="name",
                        physical_type="string",
                        confidence=0.9,
                    )
                ],
                extraction_orientation=ExtractionOrientation.ROW_RECORDS,
                record_groups=[
                    RecordGroup(
                        id="table_1::records",
                        entity_id="table_1",
                        orientation=ExtractionOrientation.ROW_RECORDS,
                        record_count=1,
                        records=[{"Name": "Alice"}],
                        confidence=0.9,
                    )
                ],
                record_count_estimate=1,
                confidence=0.9,
            )
        ],
        relationships=[],
        warnings=[],
        provenance_summary={},
        schematization_time=0.5,
        methods_used=["workbook_schematizer"],
    )


class TestGridGulpInit:
    """Test GridGulp initialization."""

    def test_init_default_config_requires_gemma(self, monkeypatch: pytest.MonkeyPatch):
        """Default initialization should require Gemma config."""
        monkeypatch.delenv("GRIDGULP_DETECTION_STRATEGY", raising=False)
        with pytest.raises(ValueError, match="Gemma base_url is required"):
            GridGulp()

    def test_init_custom_config(self, heuristic_config):
        """Test initialization with custom config."""
        config = heuristic_config.with_overrides(confidence_threshold=0.9, max_file_size_mb=50)
        gg = GridGulp(config=config)
        assert gg.config.confidence_threshold == 0.9
        assert gg.config.max_file_size_mb == 50

    def test_init_with_overrides(self, heuristic_config):
        """Test initialization with parameter overrides."""
        gg = GridGulp(config=heuristic_config, confidence_threshold=0.85, max_file_size_mb=100)
        assert gg.config.confidence_threshold == 0.85
        assert gg.config.max_file_size_mb == 100

    def test_init_with_config_and_overrides(self, heuristic_config):
        """Test that overrides take precedence over config."""
        config = heuristic_config.with_overrides(confidence_threshold=0.7)
        gg = GridGulp(config=config, confidence_threshold=0.9)
        assert gg.config.confidence_threshold == 0.9


class TestDetectTables:
    """Test the main detect_tables method."""

    @pytest.mark.asyncio
    async def test_detect_tables_success(self, mock_detection_result, heuristic_config):
        """Test successful table detection."""
        gg = GridGulp(config=heuristic_config)

        # Patch file validation
        with patch.object(gg, "_validate_file"):
            with patch.object(gg, "_analyze_file", new_callable=AsyncMock) as mock_analyze:
                mock_analyze.return_value = mock_detection_result.file_info

                with patch("gridgulp.gridgulp.create_reader") as mock_create_reader:
                    from gridgulp.readers.base_reader import SyncBaseReader

                    mock_reader = Mock(spec=SyncBaseReader)
                    # Create proper SheetData
                    sheet_data = SheetData(name="Sheet1")
                    mock_reader.read_sync.return_value = FileData(
                        sheets=[sheet_data], metadata={}, file_format="xlsx"
                    )
                    mock_create_reader.return_value = mock_reader

                    # Patch TableDetectionAgent at the module level where it's imported
                    with patch("gridgulp.gridgulp.TableDetectionAgent") as mock_agent_class:
                        mock_agent_instance = Mock()
                        mock_agent_instance.detect_tables = AsyncMock(
                            return_value=Mock(
                                tables=[mock_detection_result.sheets[0].tables[0]],
                                processing_metadata={"methods_used": ["simple_case"]},
                            )
                        )
                        mock_agent_class.return_value = mock_agent_instance

                        result = await gg.detect_tables("test.xlsx")

                        assert result.total_tables == 1
                        assert len(result.sheets) == 1
                        assert result.sheets[0].name == "Sheet1"

    @pytest.mark.asyncio
    async def test_detect_tables_file_not_found(self, heuristic_config):
        """Test detection with non-existent file."""
        gg = GridGulp(config=heuristic_config)

        with pytest.raises(FileNotFoundError):
            await gg.detect_tables("nonexistent.xlsx")

    @pytest.mark.asyncio
    async def test_detect_tables_file_too_large(self, heuristic_config):
        """Test detection with file exceeding size limit."""
        gg = GridGulp(config=heuristic_config, max_file_size_mb=0.1)  # 100KB limit

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            # Write more than 100KB
            f.write(b"x" * 200000)
            f.flush()
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="File too large"):
                await gg.detect_tables(temp_path)
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_detect_tables_reader_error(self, mock_file_info, heuristic_config):
        """Test handling of reader errors."""
        gg = GridGulp(config=heuristic_config)

        # Patch _validate_file to skip file existence check
        with patch.object(gg, "_validate_file"):
            # Patch _analyze_file to return mock file info
            with patch.object(gg, "_analyze_file", new_callable=AsyncMock) as mock_analyze:
                mock_analyze.return_value = mock_file_info

                with patch("gridgulp.gridgulp.create_reader") as mock_create_reader:
                    from gridgulp.readers.base_reader import SyncBaseReader, ReaderError

                    mock_reader = Mock(spec=SyncBaseReader)
                    mock_reader.read_sync.side_effect = ReaderError("Read error")
                    mock_create_reader.return_value = mock_reader

                    with pytest.raises(ValueError, match="Could not read file"):
                        await gg.detect_tables("test.xlsx")


class TestSchematize:
    """Test the public workbook schematization API."""

    @pytest.mark.asyncio
    async def test_schematize_success(self, mock_file_info, mock_schema_result, heuristic_config):
        """Schematize should reuse loaded file data and delegate to WorkbookSchematizer."""
        gg = GridGulp(config=heuristic_config)
        sheet_data = SheetData(name="Sheet1")

        with patch.object(gg, "_load_file_data", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = (
                mock_file_info,
                FileData(sheets=[sheet_data], metadata={}, file_format="xlsx"),
            )
            with patch.object(gg, "_detect_loaded_file", new_callable=AsyncMock) as mock_detect:
                mock_detect.return_value = DetectionResult(
                    file_info=mock_file_info,
                    sheets=[],
                    detection_time=0.1,
                    methods_used=["heuristic_table"],
                    metadata={"detection_strategy": "heuristic_only"},
                )
                with patch("gridgulp.gridgulp.WorkbookSchematizer") as mock_schematizer_class:
                    mock_schematizer = Mock()
                    mock_schematizer.schematize = AsyncMock(return_value=mock_schema_result)
                    mock_schematizer_class.return_value = mock_schematizer

                    result = await gg.schematize("test.xlsx")

        assert result == mock_schema_result
        mock_load.assert_called_once()
        mock_detect.assert_called_once()
        mock_schematizer.schematize.assert_called_once()


class TestDetectTablesInDirectory:
    """Test directory processing methods."""

    @pytest.mark.asyncio
    async def test_detect_tables_in_directory_success(
        self, mock_detection_result, heuristic_config
    ):
        """Test successful directory processing."""
        gg = GridGulp(config=heuristic_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "test1.xlsx").write_text("test")
            (temp_path / "test2.csv").write_text("a,b,c\n1,2,3")
            (temp_path / "subdir").mkdir()
            (temp_path / "subdir" / "test3.xlsx").write_text("test")

            # Mock detect_tables to return success for each file
            gg.detect_tables = AsyncMock(return_value=mock_detection_result)

            results = await gg.detect_tables_in_directory(temp_path)

            assert len(results) == 3  # 3 files found
            assert all(isinstance(r, DetectionResult) for r in results.values())
            assert gg.detect_tables.call_count == 3

    @pytest.mark.asyncio
    async def test_detect_tables_in_directory_patterns(
        self, mock_detection_result, heuristic_config
    ):
        """Test directory processing with file patterns."""
        gg = GridGulp(config=heuristic_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "test1.xlsx").write_text("test")
            (temp_path / "test2.csv").write_text("a,b,c")
            (temp_path / "test3.txt").write_text("text")

            gg.detect_tables = AsyncMock(return_value=mock_detection_result)

            # Only process Excel files
            results = await gg.detect_tables_in_directory(temp_path, patterns=["*.xlsx"])

            assert len(results) == 1
            assert any("test1.xlsx" in str(p) for p in results.keys())

    @pytest.mark.asyncio
    async def test_detect_tables_in_directory_no_recursive(
        self, mock_detection_result, heuristic_config
    ):
        """Test non-recursive directory processing."""
        gg = GridGulp(config=heuristic_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files in root and subdirectory
            (temp_path / "test1.xlsx").write_text("test")
            (temp_path / "subdir").mkdir()
            (temp_path / "subdir" / "test2.xlsx").write_text("test")

            gg.detect_tables = AsyncMock(return_value=mock_detection_result)

            results = await gg.detect_tables_in_directory(temp_path, recursive=False)

            assert len(results) == 1  # Only root file
            assert any("test1.xlsx" in str(p) for p in results.keys())

    @pytest.mark.asyncio
    async def test_detect_tables_in_directory_progress_callback(self, heuristic_config):
        """Test progress callback functionality."""
        gg = GridGulp(config=heuristic_config)
        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test1.xlsx").write_text("test")
            (temp_path / "test2.xlsx").write_text("test")

            gg.detect_tables = AsyncMock(return_value=Mock(spec=DetectionResult))

            await gg.detect_tables_in_directory(temp_path, progress_callback=progress_callback)

            assert len(progress_calls) == 2
            assert progress_calls[0] == (1, 2)
            assert progress_calls[1] == (2, 2)

    @pytest.mark.asyncio
    async def test_detect_tables_in_directory_error_handling(self, heuristic_config):
        """Test that directory processing continues despite individual file errors."""
        gg = GridGulp(config=heuristic_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "good.xlsx").write_text("test")
            (temp_path / "bad.xlsx").write_text("test")

            # Mock detect_tables to fail on one file
            async def mock_detect(file_path):
                if "bad" in str(file_path):
                    raise Exception("File corrupted")
                return Mock(spec=DetectionResult, total_tables=1)

            gg.detect_tables = mock_detect

            results = await gg.detect_tables_in_directory(temp_path)

            assert len(results) == 2
            # Check that error is captured in metadata
            bad_result = next(r for p, r in results.items() if "bad" in str(p))
            assert "error" in bad_result.metadata
            assert bad_result.total_tables == 0

    @pytest.mark.asyncio
    async def test_detect_tables_in_directory_invalid_path(self, heuristic_config):
        """Test error when directory doesn't exist."""
        gg = GridGulp(config=heuristic_config)

        with pytest.raises(ValueError, match="Directory not found"):
            await gg.detect_tables_in_directory("/nonexistent/directory")

    @pytest.mark.asyncio
    async def test_detect_tables_in_directory_file_instead_of_dir(self, heuristic_config):
        """Test error when path is a file, not directory."""
        gg = GridGulp(config=heuristic_config)

        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(ValueError, match="Not a directory"):
                await gg.detect_tables_in_directory(f.name)


class TestSyncMethods:
    """Test synchronous wrapper methods."""

    def test_detect_tables_sync(self, mock_detection_result, heuristic_config):
        """Test synchronous table detection."""
        gg = GridGulp(config=heuristic_config)
        gg.detect_tables = AsyncMock(return_value=mock_detection_result)

        result = gg.detect_tables_sync("test.xlsx")

        assert result == mock_detection_result
        gg.detect_tables.assert_called_once_with("test.xlsx")

    def test_detect_tables_in_directory_sync(self, heuristic_config):
        """Test synchronous directory processing."""
        gg = GridGulp(config=heuristic_config)
        mock_results = {Path("test.xlsx"): Mock(spec=DetectionResult)}
        gg.detect_tables_in_directory = AsyncMock(return_value=mock_results)

        results = gg.detect_tables_in_directory_sync("~/data", patterns=["*.xlsx"], recursive=True)

        assert results == mock_results
        gg.detect_tables_in_directory.assert_called_once_with("~/data", ["*.xlsx"], True, None)


class TestPathHandling:
    """Test various path input formats."""

    @pytest.mark.asyncio
    async def test_string_path(self, mock_detection_result, heuristic_config):
        """Test with string path."""
        gg = GridGulp(config=heuristic_config)

        with patch.object(gg, "_validate_file"):
            with patch.object(gg, "_analyze_file", new_callable=AsyncMock) as mock_analyze:
                mock_analyze.return_value = mock_detection_result.file_info

                with patch("gridgulp.gridgulp.create_reader") as mock_create_reader:
                    from gridgulp.readers.base_reader import SyncBaseReader

                    mock_reader = Mock(spec=SyncBaseReader)
                    mock_reader.read_sync.return_value = FileData(
                        sheets=[], metadata={}, file_format="xlsx"
                    )
                    mock_create_reader.return_value = mock_reader

                    with patch("gridgulp.gridgulp.TableDetectionAgent"):
                        await gg.detect_tables("test.xlsx")

    @pytest.mark.asyncio
    async def test_path_object(self, mock_detection_result, heuristic_config):
        """Test with Path object."""
        gg = GridGulp(config=heuristic_config)

        with patch.object(gg, "_validate_file"):
            with patch.object(gg, "_analyze_file", new_callable=AsyncMock) as mock_analyze:
                mock_analyze.return_value = mock_detection_result.file_info

                with patch("gridgulp.gridgulp.create_reader") as mock_create_reader:
                    from gridgulp.readers.base_reader import SyncBaseReader

                    mock_reader = Mock(spec=SyncBaseReader)
                    mock_reader.read_sync.return_value = FileData(
                        sheets=[], metadata={}, file_format="xlsx"
                    )
                    mock_create_reader.return_value = mock_reader

                    with patch("gridgulp.gridgulp.TableDetectionAgent"):
                        await gg.detect_tables(Path("test.xlsx"))

    @pytest.mark.asyncio
    async def test_expanduser_in_directory(self, heuristic_config):
        """Test that ~ is expanded in directory paths."""
        gg = GridGulp(config=heuristic_config)
        gg.detect_tables = AsyncMock(return_value=Mock(spec=DetectionResult))

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                with patch("pathlib.Path.rglob", return_value=[]):
                    results = await gg.detect_tables_in_directory("~/data")

                    # Verify path was expanded
                    assert results == {}  # No files found is OK for this test
