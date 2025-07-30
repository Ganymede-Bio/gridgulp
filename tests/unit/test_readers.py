"""Unit tests for file readers."""

from pathlib import Path
from unittest.mock import patch

import pytest

from gridgulp.models import FileInfo, FileType, SheetData
from gridgulp.readers import (
    BaseReader,
    CSVReader,
    ExcelReader,
    ReaderFactory,
    UnsupportedFileError,
    create_reader,
    get_factory,
)


class MockReader(BaseReader):
    """Mock reader for testing."""

    def __init__(self, file_path: Path, file_info: FileInfo):
        super().__init__(file_path, file_info)
        self.read_called = False

    async def read(self):
        self.read_called = True
        return self._create_mock_file_data()

    def can_read(self):
        return True

    def get_supported_formats(self):
        return ["mock"]

    def _create_mock_file_data(self):
        from gridgulp.models import FileData

        sheet = self._create_empty_sheet("MockSheet")
        return FileData(sheets=[sheet], metadata={}, file_format="mock")


class TestBaseReader:
    """Test BaseReader functionality."""

    def test_init(self):
        """Test reader initialization."""
        file_path = Path("test.xlsx")
        file_info = FileInfo(path=file_path, type=FileType.XLSX, size=1000)

        reader = MockReader(file_path, file_info)
        assert reader.file_path == file_path
        assert reader.file_info == file_info

    def test_validate_file_not_exists(self):
        """Test validation with non-existent file."""
        file_path = Path("nonexistent.xlsx")
        file_info = FileInfo(path=file_path, type=FileType.XLSX, size=1000)
        reader = MockReader(file_path, file_info)

        with pytest.raises(FileNotFoundError):
            reader.validate_file()

    def test_detect_encoding_with_chardet(self):
        """Test encoding detection with chardet available."""
        file_path = Path("test.csv")
        file_info = FileInfo(path=file_path, type=FileType.CSV, size=1000)
        reader = MockReader(file_path, file_info)

        # Mock chardet
        with patch("chardet.detect") as mock_detect:
            mock_detect.return_value = {"encoding": "utf-8", "confidence": 0.9}

            encoding = reader._detect_encoding(b"test data")
            assert encoding == "utf-8"

    def test_detect_encoding_without_chardet(self):
        """Test encoding detection without chardet."""
        file_path = Path("test.csv")
        file_info = FileInfo(path=file_path, type=FileType.CSV, size=1000)
        reader = MockReader(file_path, file_info)

        # Mock ImportError for chardet
        with patch("chardet.detect", side_effect=ImportError):
            encoding = reader._detect_encoding(b"test data")
            assert encoding == "utf-8"

    def test_create_empty_sheet(self):
        """Test empty sheet creation."""
        file_path = Path("test.xlsx")
        file_info = FileInfo(path=file_path, type=FileType.XLSX, size=1000)
        reader = MockReader(file_path, file_info)

        sheet = reader._create_empty_sheet("TestSheet")
        assert isinstance(sheet, SheetData)
        assert sheet.name == "TestSheet"
        assert len(sheet.cells) == 0


class TestReaderFactory:
    """Test ReaderFactory functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.factory = ReaderFactory()

    def test_default_readers_registered(self):
        """Test that default readers are registered."""
        supported_types = self.factory.get_supported_types()

        assert "xlsx" in supported_types
        assert "xls" in supported_types
        assert "csv" in supported_types
        assert "tsv" in supported_types

    def test_register_custom_reader(self):
        """Test registering a custom reader."""
        self.factory.register_reader(FileType.UNKNOWN, MockReader)

        assert FileType.UNKNOWN in self.factory._readers
        assert self.factory._readers[FileType.UNKNOWN] == MockReader

    def test_unregister_reader(self):
        """Test unregistering a reader."""
        self.factory.unregister_reader(FileType.CSV)
        assert FileType.CSV not in self.factory._readers

    def test_get_reader_excel(self):
        """Test getting Excel reader."""
        file_path = Path("test.xlsx")
        file_info = FileInfo(path=file_path, type=FileType.XLSX, size=1000)

        reader = self.factory.get_reader(file_path, file_info)
        assert isinstance(reader, ExcelReader)

    def test_get_reader_csv(self):
        """Test getting CSV reader."""
        file_path = Path("test.csv")
        file_info = FileInfo(path=file_path, type=FileType.CSV, size=1000)

        reader = self.factory.get_reader(file_path, file_info)
        assert isinstance(reader, CSVReader)

    def test_get_reader_unsupported(self):
        """Test getting reader for unsupported format."""
        file_path = Path("test.unknown")
        file_info = FileInfo(path=file_path, type=FileType.UNKNOWN, size=1000)

        with pytest.raises(UnsupportedFileError):
            self.factory.get_reader(file_path, file_info)

    def test_can_read(self):
        """Test can_read method."""
        xlsx_info = FileInfo(path=Path("test.xlsx"), type=FileType.XLSX, size=1000)
        unknown_info = FileInfo(path=Path("test.unknown"), type=FileType.UNKNOWN, size=1000)

        assert self.factory.can_read(xlsx_info) is True
        assert self.factory.can_read(unknown_info) is False

    def test_get_reader_info(self):
        """Test getting reader information."""
        info = self.factory.get_reader_info()

        assert "ExcelReader" in info
        assert "CSVReader" in info
        assert isinstance(info["ExcelReader"], list)
        assert isinstance(info["CSVReader"], list)


class TestCSVReader:
    """Test CSVReader functionality."""

    def test_can_read_csv(self):
        """Test CSV reader can read CSV files."""
        file_path = Path("test.csv")
        file_info = FileInfo(path=file_path, type=FileType.CSV, size=1000)
        reader = CSVReader(file_path, file_info)

        assert reader.can_read() is True

    def test_can_read_tsv(self):
        """Test CSV reader can read TSV files."""
        file_path = Path("test.tsv")
        file_info = FileInfo(path=file_path, type=FileType.TSV, size=1000)
        reader = CSVReader(file_path, file_info)

        assert reader.can_read() is True

    def test_cannot_read_excel(self):
        """Test CSV reader cannot read Excel files."""
        file_path = Path("test.xlsx")
        file_info = FileInfo(path=file_path, type=FileType.XLSX, size=1000)
        reader = CSVReader(file_path, file_info)

        assert reader.can_read() is False

    def test_get_supported_formats(self):
        """Test getting supported formats."""
        file_path = Path("test.csv")
        file_info = FileInfo(path=file_path, type=FileType.CSV, size=1000)
        reader = CSVReader(file_path, file_info)

        formats = reader.get_supported_formats()
        assert "csv" in formats
        assert "tsv" in formats
        assert "txt" in formats

    def test_detect_delimiter_manual(self):
        """Test manual delimiter detection."""
        file_path = Path("test.csv")
        file_info = FileInfo(path=file_path, type=FileType.CSV, size=1000)
        reader = CSVReader(file_path, file_info)

        # Test comma detection
        sample = "a,b,c\nd,e,f\ng,h,i"
        delimiter = reader._detect_delimiter_manual(sample)
        assert delimiter == ","

        # Test semicolon detection
        sample = "a;b;c\nd;e;f\ng;h;i"
        delimiter = reader._detect_delimiter_manual(sample)
        assert delimiter == ";"

    def test_infer_type_string(self):
        """Test type inference for strings."""
        file_path = Path("test.csv")
        file_info = FileInfo(path=file_path, type=FileType.CSV, size=1000)
        reader = CSVReader(file_path, file_info)

        value, data_type = reader._infer_type("hello world")
        assert value == "hello world"
        assert data_type == "string"

    def test_infer_type_number(self):
        """Test type inference for numbers."""
        file_path = Path("test.csv")
        file_info = FileInfo(path=file_path, type=FileType.CSV, size=1000)
        reader = CSVReader(file_path, file_info)

        # Integer
        value, data_type = reader._infer_type("123")
        assert value == 123
        assert data_type == "number"

        # Float
        value, data_type = reader._infer_type("123.45")
        assert value == 123.45
        assert data_type == "number"

    def test_infer_type_boolean(self):
        """Test type inference for booleans."""
        file_path = Path("test.csv")
        file_info = FileInfo(path=file_path, type=FileType.CSV, size=1000)
        reader = CSVReader(file_path, file_info)

        value, data_type = reader._infer_type("true")
        assert value is True
        assert data_type == "boolean"

        value, data_type = reader._infer_type("false")
        assert value is False
        assert data_type == "boolean"

    def test_looks_like_date(self):
        """Test date detection."""
        file_path = Path("test.csv")
        file_info = FileInfo(path=file_path, type=FileType.CSV, size=1000)
        reader = CSVReader(file_path, file_info)

        assert reader._looks_like_date("2023-12-25") is True
        assert reader._looks_like_date("12/25/2023") is True
        assert reader._looks_like_date("25 Dec 2023") is True
        assert reader._looks_like_date("hello world") is False


class TestExcelReader:
    """Test ExcelReader functionality."""

    def test_can_read_xlsx(self):
        """Test Excel reader can read XLSX files."""
        file_path = Path("test.xlsx")
        file_info = FileInfo(path=file_path, type=FileType.XLSX, size=1000)
        reader = ExcelReader(file_path, file_info)

        assert reader.can_read() is True

    def test_can_read_xls(self):
        """Test Excel reader can read XLS files."""
        file_path = Path("test.xls")
        file_info = FileInfo(path=file_path, type=FileType.XLS, size=1000)
        reader = ExcelReader(file_path, file_info)

        assert reader.can_read() is True

    def test_cannot_read_csv(self):
        """Test Excel reader cannot read CSV files."""
        file_path = Path("test.csv")
        file_info = FileInfo(path=file_path, type=FileType.CSV, size=1000)
        reader = ExcelReader(file_path, file_info)

        assert reader.can_read() is False

    def test_get_supported_formats(self):
        """Test getting supported formats."""
        file_path = Path("test.xlsx")
        file_info = FileInfo(path=file_path, type=FileType.XLSX, size=1000)
        reader = ExcelReader(file_path, file_info)

        formats = reader.get_supported_formats()
        assert "xlsx" in formats
        assert "xls" in formats
        assert "xlsm" in formats

    def test_get_data_type(self):
        """Test data type detection."""
        file_path = Path("test.xlsx")
        file_info = FileInfo(path=file_path, type=FileType.XLSX, size=1000)
        reader = ExcelReader(file_path, file_info)

        assert reader._get_data_type(None) == "empty"
        assert reader._get_data_type(True) == "boolean"
        assert reader._get_data_type(123) == "number"
        assert reader._get_data_type(123.45) == "number"
        assert reader._get_data_type("hello") == "string"


class TestFactoryFunctions:
    """Test factory convenience functions."""

    def test_get_factory(self):
        """Test getting global factory."""
        factory = get_factory()
        assert isinstance(factory, ReaderFactory)

        # Should return same instance
        factory2 = get_factory()
        assert factory is factory2

    def test_create_reader(self):
        """Test create_reader convenience function."""
        file_path = Path("test.csv")
        file_info = FileInfo(path=file_path, type=FileType.CSV, size=1000)

        reader = create_reader(file_path, file_info)
        assert isinstance(reader, CSVReader)

    @pytest.mark.asyncio
    async def test_mock_reader_integration(self):
        """Test mock reader integration."""
        file_path = Path("test.mock")
        file_info = FileInfo(path=file_path, type=FileType.UNKNOWN, size=1000)
        reader = MockReader(file_path, file_info)

        assert reader.read_called is False
        result = await reader.read()
        assert reader.read_called is True
        assert result.sheet_count == 1
