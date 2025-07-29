"""Shared type definitions for GridGulp."""

from typing import TypeAlias, TypeVar

# Type aliases for common types
CellValue: TypeAlias = str | int | float | bool | None
RowIndex: TypeAlias = int
ColumnIndex: TypeAlias = int
CellAddress: TypeAlias = tuple[RowIndex, ColumnIndex]

# Generic type variables
T = TypeVar("T")
SheetT = TypeVar("SheetT")  # Type variable for sheet-like objects
