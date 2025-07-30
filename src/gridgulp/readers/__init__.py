"""File readers for various spreadsheet formats."""

from .base_reader import (
    BaseReader,
    CorruptedFileError,
    PasswordProtectedError,
    ReaderError,
    UnsupportedFileError,
)
from .convenience import get_async_reader, get_reader, is_supported
from .csv_reader import CSVReader
from .excel_reader import ExcelReader
from .factory import ReaderFactory, create_reader, get_factory, register_custom_reader
from .reader_adapter import ReaderAdapter

__all__ = [
    "BaseReader",
    "ExcelReader",
    "CSVReader",
    "ReaderFactory",
    "ReaderAdapter",
    "create_reader",
    "get_factory",
    "register_custom_reader",
    "get_reader",
    "get_async_reader",
    "is_supported",
    "ReaderError",
    "UnsupportedFileError",
    "CorruptedFileError",
    "PasswordProtectedError",
]
