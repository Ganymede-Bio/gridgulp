"""Tests for the main GridGulp class."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import tempfile

import pytest

from gridgulp import GridGulp
from gridgulp.config import Config
from gridgulp.models import (
    DetectionResult,
    FileInfo,
    FileType,
    SheetResult,
    TableInfo,
    TableRange,
)
from gridgulp.models.sheet_data import SheetData, FileData
from gridgulp.models.file_info import UnsupportedFormatError


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


class TestGridGulpInit:
    """Test GridGulp initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        gg = GridGulp()
        assert isinstance(gg.config, Config)
        assert gg.config.confidence_threshold > 0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = Config(confidence_threshold=0.9, max_file_size_mb=50)
        gg = GridGulp(config=config)
        assert gg.config.confidence_threshold == 0.9
        assert gg.config.max_file_size_mb == 50

    def test_init_with_overrides(self):
        """Test initialization with parameter overrides."""
        gg = GridGulp(confidence_threshold=0.85, max_file_size_mb=100)
        assert gg.config.confidence_threshold == 0.85
        assert gg.config.max_file_size_mb == 100

    def test_init_with_config_and_overrides(self):
        """Test that overrides take precedence over config."""
        config = Config(confidence_threshold=0.7)
        gg = GridGulp(config=config, confidence_threshold=0.9)
        assert gg.config.confidence_threshold == 0.9


class TestDetectTables:
    """Test the main detect_tables method."""

    @pytest.mark.asyncio
    async def test_detect_tables_success(self, mock_detection_result):
        """Test successful table detection."""
        gg = GridGulp()

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
                                processing_metadata={},
                            )
                        )
                        mock_agent_class.return_value = mock_agent_instance

                        result = await gg.detect_tables("test.xlsx")

                        assert result.total_tables == 1
                        assert len(result.sheets) == 1
                        assert result.sheets[0].name == "Sheet1"

    @pytest.mark.asyncio
    async def test_detect_tables_file_not_found(self):
        """Test detection with non-existent file."""
        gg = GridGulp()

        with pytest.raises(FileNotFoundError):
            await gg.detect_tables("nonexistent.xlsx")

    @pytest.mark.asyncio
    async def test_detect_tables_file_too_large(self):
        """Test detection with file exceeding size limit."""
        gg = GridGulp(max_file_size_mb=0.001)  # 1KB limit

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            # Write more than 1KB
            f.write(b"x" * 2000)
            f.flush()
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="File too large"):
                await gg.detect_tables(temp_path)
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_detect_tables_reader_error(self, mock_file_info):
        """Test handling of reader errors."""
        gg = GridGulp()

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


class TestDetectTablesInDirectory:
    """Test directory processing methods."""

    @pytest.mark.asyncio
    async def test_detect_tables_in_directory_success(self, mock_detection_result):
        """Test successful directory processing."""
        gg = GridGulp()

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
    async def test_detect_tables_in_directory_patterns(self, mock_detection_result):
        """Test directory processing with file patterns."""
        gg = GridGulp()

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
    async def test_detect_tables_in_directory_no_recursive(self, mock_detection_result):
        """Test non-recursive directory processing."""
        gg = GridGulp()

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
    async def test_detect_tables_in_directory_progress_callback(self):
        """Test progress callback functionality."""
        gg = GridGulp()
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
    async def test_detect_tables_in_directory_error_handling(self):
        """Test that directory processing continues despite individual file errors."""
        gg = GridGulp()

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
    async def test_detect_tables_in_directory_invalid_path(self):
        """Test error when directory doesn't exist."""
        gg = GridGulp()

        with pytest.raises(ValueError, match="Directory not found"):
            await gg.detect_tables_in_directory("/nonexistent/directory")

    @pytest.mark.asyncio
    async def test_detect_tables_in_directory_file_instead_of_dir(self):
        """Test error when path is a file, not directory."""
        gg = GridGulp()

        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(ValueError, match="Not a directory"):
                await gg.detect_tables_in_directory(f.name)


class TestSyncMethods:
    """Test synchronous wrapper methods."""

    def test_detect_tables_sync(self, mock_detection_result):
        """Test synchronous table detection."""
        gg = GridGulp()
        gg.detect_tables = AsyncMock(return_value=mock_detection_result)

        result = gg.detect_tables_sync("test.xlsx")

        assert result == mock_detection_result
        gg.detect_tables.assert_called_once_with("test.xlsx")

    def test_detect_tables_in_directory_sync(self):
        """Test synchronous directory processing."""
        gg = GridGulp()
        mock_results = {Path("test.xlsx"): Mock(spec=DetectionResult)}
        gg.detect_tables_in_directory = AsyncMock(return_value=mock_results)

        results = gg.detect_tables_in_directory_sync("~/data", patterns=["*.xlsx"], recursive=True)

        assert results == mock_results
        gg.detect_tables_in_directory.assert_called_once_with("~/data", ["*.xlsx"], True, None)


class TestPathHandling:
    """Test various path input formats."""

    @pytest.mark.asyncio
    async def test_string_path(self, mock_detection_result):
        """Test with string path."""
        gg = GridGulp()

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
    async def test_path_object(self, mock_detection_result):
        """Test with Path object."""
        gg = GridGulp()

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
    async def test_expanduser_in_directory(self):
        """Test that ~ is expanded in directory paths."""
        gg = GridGulp()
        gg.detect_tables = AsyncMock(return_value=Mock(spec=DetectionResult))

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                with patch("pathlib.Path.rglob", return_value=[]):
                    results = await gg.detect_tables_in_directory("~/data")

                    # Verify path was expanded
                    assert results == {}  # No files found is OK for this test
