"""File data models for GridGulp."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FileData(BaseModel):
    """Represents a loaded spreadsheet file."""

    model_config = ConfigDict(strict=True)

    path: str = Field(..., description="File path")
    format: str = Field(..., description="File format (xlsx, csv, etc.)")
    filename: str = Field(default="", description="File name without path")
    sheets: dict[str, Any] = Field(default_factory=dict, description="Sheet data by name")

    @property
    def sheet_names(self) -> list[str]:
        """Get list of sheet names."""
        return list(self.sheets.keys())

    def get_sheet(self, name: str) -> Any:
        """Get sheet data by name."""
        return self.sheets.get(name)

    def add_sheet(self, name: str, sheet_data: Any) -> None:
        """Add sheet data."""
        self.sheets[name] = sheet_data
