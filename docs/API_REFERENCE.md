# GridPorter API Reference

## Core Classes

### GridPorter

The main entry point for table detection.

```python
class GridPorter:
    def __init__(self, config: Config | None = None):
        """Initialize GridPorter with optional configuration.

        Args:
            config: Configuration object. If None, uses defaults.
        """

    async def detect_tables(self, file_path: str | Path) -> DetectionResult:
        """Detect tables in a spreadsheet file.

        Args:
            file_path: Path to the file to process

        Returns:
            DetectionResult containing all detected tables

        Raises:
            FileNotFoundError: If file doesn't exist
            ReaderError: If file cannot be read
            DetectionError: If detection fails
        """
```

### Config

Configuration options for GridPorter.

```python
class Config(BaseModel):
    # Detection Configuration
    confidence_threshold: float = 0.7  # Min confidence (0.0-1.0)
    max_tables_per_sheet: int = 50    # Max tables per sheet
    min_table_size: tuple[int, int] = (2, 2)  # Min (rows, cols)
    detect_merged_cells: bool = True   # Handle merged cells

    # File Detection
    enable_magika: bool = True         # AI file type detection
    strict_format_checking: bool = False  # Strict format validation
    file_detection_buffer_size: int = 8192  # Detection buffer size

    # Processing Limits
    max_file_size_mb: float = 2000.0   # Max file size
    timeout_seconds: int = 300         # Processing timeout
    max_sheets: int = 10               # Max sheets to process

    # Performance
    excel_reader: str = "calamine"     # "calamine" or "openpyxl"
    max_memory_mb: int = 1000          # Max memory usage
    chunk_size: int = 10000            # Streaming chunk size

    # Detection Methods
    enable_simple_case_detection: bool = True  # Fast path
    enable_island_detection: bool = True       # Multi-table
    use_excel_metadata: bool = True           # Excel metadata

    # Detection Thresholds
    island_min_cells: int = 20         # Min cells for island
    island_density_threshold: float = 0.8  # Required density

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
```

## Data Models

### DetectionResult

The main result object returned by table detection.

```python
class DetectionResult(BaseModel):
    file_info: FileInfo                # File metadata
    sheets: list[SheetResult]          # Results per sheet
    metadata: dict[str, Any]           # Additional metadata

    @property
    def total_tables(self) -> int:
        """Total number of tables detected."""

    @property
    def detection_time(self) -> float:
        """Total detection time in seconds."""

    def to_dict(self) -> dict:
        """Convert to dictionary format."""

    def to_summary(self) -> dict:
        """Create summary statistics."""
```

### SheetResult

Detection results for a single sheet.

```python
class SheetResult(BaseModel):
    name: str                          # Sheet name
    tables: list[TableInfo]            # Detected tables
    metadata: dict[str, Any] = {}      # Sheet metadata

    @property
    def has_tables(self) -> bool:
        """Check if sheet has any tables."""
```

### TableInfo

Information about a detected table.

```python
class TableInfo(BaseModel):
    range: TableRange                  # Table boundaries
    suggested_name: str | None = None  # Optional name
    confidence: float                  # Detection confidence (0-1)
    detection_method: str              # Method used
    headers: list[str] | None = None   # Column headers
    data_preview: list[dict] | None = None  # Sample data

    @property
    def shape(self) -> tuple[int, int]:
        """Table dimensions (rows, cols)."""

    @property
    def id(self) -> str:
        """Unique table identifier."""
```

### TableRange

Table boundary definition.

```python
class TableRange(BaseModel):
    start_row: int                     # 0-based start row
    start_col: int                     # 0-based start column
    end_row: int                       # 0-based end row (inclusive)
    end_col: int                       # 0-based end column (inclusive)

    @property
    def excel_range(self) -> str:
        """Excel-style range (e.g., 'A1:D10')."""

    @property
    def rows(self) -> int:
        """Number of rows in range."""

    @property
    def cols(self) -> int:
        """Number of columns in range."""

    def contains(self, row: int, col: int) -> bool:
        """Check if cell is within range."""
```

### FileInfo

File metadata and type information.

```python
class FileInfo(BaseModel):
    path: Path                         # File path
    type: FileType                     # Detected file type
    size_mb: float                     # File size in MB
    detected_type: str | None = None   # MIME type

    @property
    def extension(self) -> str:
        """File extension (lowercase)."""
```

### FileType

Enumeration of supported file types.

```python
class FileType(str, Enum):
    XLSX = "xlsx"      # Modern Excel
    XLS = "xls"        # Legacy Excel
    XLSM = "xlsm"      # Excel with macros
    XLSB = "xlsb"      # Excel binary
    CSV = "csv"        # Comma-separated
    TSV = "tsv"        # Tab-separated
    TXT = "txt"        # Text file
    UNKNOWN = "unknown"
```

## Detectors

### SimpleCaseDetector

Fast detection for single tables.

```python
class SimpleCaseDetector:
    def detect_simple_table(self, sheet_data: SheetData) -> SimpleCaseResult:
        """Detect a single table starting near A1.

        Args:
            sheet_data: Sheet data to analyze

        Returns:
            SimpleCaseResult with detection info
        """

    def is_simple_case(self, sheet_data: SheetData) -> bool:
        """Check if sheet is a simple single-table case."""
```

### IslandDetector

Multi-table detection using connected components.

```python
class IslandDetector:
    def detect_islands(self, sheet_data: SheetData) -> list[Island]:
        """Detect disconnected data regions.

        Args:
            sheet_data: Sheet data to analyze

        Returns:
            List of Island objects
        """

    def calculate_density(self, island: Island) -> float:
        """Calculate cell density for an island."""
```

### ExcelMetadataExtractor

Extract tables from Excel metadata.

```python
class ExcelMetadataExtractor:
    def extract_list_objects(self, workbook: Workbook) -> list[TableInfo]:
        """Extract Excel ListObjects (defined tables).

        Args:
            workbook: openpyxl Workbook object

        Returns:
            List of TableInfo for defined tables
        """

    def extract_named_ranges(self, workbook: Workbook) -> list[NamedRange]:
        """Extract named ranges from workbook."""
```

## Readers

### BaseReader

Abstract base class for file readers.

```python
class BaseReader(ABC):
    @abstractmethod
    async def read(self) -> FileData:
        """Read file and return structured data."""

    @abstractmethod
    def can_read(self) -> bool:
        """Check if reader can handle the file."""

    def validate_file(self) -> None:
        """Validate file exists and is readable."""
```

### ExcelReader

Reader for Excel files.

```python
class ExcelReader(SyncBaseReader):
    def read_sync(self) -> FileData:
        """Read Excel file synchronously."""

    def get_sheet_names(self) -> list[str]:
        """Get list of sheet names."""
```

### CSVReader

Reader for CSV/TSV files.

```python
class CSVReader(SyncBaseReader):
    def read_sync(self) -> FileData:
        """Read CSV file with auto-detection."""

    def _detect_dialect(self, sample: str) -> csv.Dialect:
        """Detect CSV dialect (delimiter, etc)."""
```

### TextReader

Reader for text files with tabular data.

```python
class TextReader(BaseReader):
    def can_read(self) -> bool:
        """Check if text file contains tabular data."""

    def _detect_encoding_sophisticated(self,
                                     header_bytes: bytes) -> EncodingResult:
        """Sophisticated encoding detection."""
```

## Sheet Data

### SheetData

Container for sheet content.

```python
class SheetData(BaseModel):
    name: str                          # Sheet name
    cells: dict[str, CellData]         # Cells by address
    max_row: int = 0                   # Max row index
    max_column: int = 0                # Max column index

    def has_data(self) -> bool:
        """Check if sheet has any data."""

    def get_cell(self, row: int, column: int) -> CellData | None:
        """Get cell at position."""

    def set_cell(self, row: int, column: int, cell_data: CellData) -> None:
        """Set cell at position."""

    def get_range_data(self, start_row: int, start_col: int,
                      end_row: int, end_col: int) -> list[list[CellData | None]]:
        """Get cells in range."""
```

### CellData

Individual cell information.

```python
class CellData(BaseModel):
    value: Any                         # Cell value
    formatted_value: str | None = None # Display value
    data_type: str = "string"          # Data type

    # Formatting
    is_bold: bool = False
    is_italic: bool = False
    font_size: float | None = None

    # Position
    row: int = 0
    column: int = 0

    @property
    def is_empty(self) -> bool:
        """Check if cell is empty."""

    @property
    def excel_address(self) -> str:
        """Get Excel address (e.g., 'A1')."""
```

## Utilities

### FileFormatDetector

Advanced file type detection.

```python
class FileFormatDetector:
    def detect_file_type(self, file_path: Path) -> FileInfo:
        """Detect file type using multiple methods.

        Uses Magika, python-magic, and content analysis.
        """

    def _detect_encoding_sophisticated(self,
                                     raw_data: bytes) -> EncodingResult:
        """Multi-layer encoding detection."""
```

### ReaderFactory

Factory for creating appropriate readers.

```python
class ReaderFactory:
    @staticmethod
    def create_reader(file_path: Path,
                     file_info: FileInfo) -> BaseReader:
        """Create reader for file type.

        Args:
            file_path: Path to file
            file_info: Detected file information

        Returns:
            Appropriate reader instance
        """
```

## Exceptions

### ReaderError

Base exception for reader errors.

```python
class ReaderError(Exception):
    """Base exception for file reading errors."""
```

### DetectionError

Base exception for detection errors.

```python
class DetectionError(Exception):
    """Base exception for table detection errors."""
```

### Specific Exceptions

```python
class UnsupportedFileError(ReaderError):
    """File format not supported."""

class CorruptedFileError(ReaderError):
    """File is corrupted or unreadable."""

class NoTablesFoundError(DetectionError):
    """No tables detected in file."""

class DetectionTimeoutError(DetectionError):
    """Detection exceeded timeout."""
```

## Constants

Key constants used throughout the system:

```python
# From core.constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
MAX_TABLES_PER_SHEET = 50
MIN_TABLE_SIZE = (2, 2)

# Detection methods
DETECTION_SIMPLE_CASE = "simple_case_fast"
DETECTION_ISLAND = "island_detection_fast"
DETECTION_METADATA = "excel_metadata"

# File size limits
MAX_FILE_SIZE_MB = 2000.0
DEFAULT_TIMEOUT_SECONDS = 300
```

## Type Definitions

Common type aliases:

```python
from typing import TypeAlias

# Cell position
CellPos: TypeAlias = tuple[int, int]  # (row, col)

# Table bounds
TableBounds: TypeAlias = tuple[int, int, int, int]  # (r1, c1, r2, c2)

# Detection confidence
Confidence: TypeAlias = float  # 0.0 to 1.0
```
