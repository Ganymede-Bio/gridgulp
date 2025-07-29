"""Detection result models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .file_info import FileInfo
from .table import TableInfo


class SheetResult(BaseModel):
    """Results for a single sheet."""

    model_config = ConfigDict(strict=True)

    name: str = Field(..., description="Sheet name")
    tables: list[TableInfo] = Field(
        default_factory=list, description="Detected tables in this sheet"
    )
    processing_time: float = Field(
        ..., ge=0.0, description="Time taken to process this sheet (seconds)"
    )
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")


class DetectionResult(BaseModel):
    """Complete detection results for a file."""

    model_config = ConfigDict(strict=True)

    file_info: FileInfo = Field(..., description="Information about the source file")
    sheets: list[SheetResult] = Field(default_factory=list, description="Results for each sheet")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    total_tables: int = Field(0, ge=0, description="Total number of tables detected")
    detection_time: float = Field(..., ge=0.0, description="Total detection time (seconds)")
    methods_used: list[str] = Field(
        default_factory=list, description="Detection methods that were used"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When detection was performed"
    )

    def __init__(self, **data: Any) -> None:
        """Initialize and calculate total tables."""
        super().__init__(**data)
        # Update total tables count
        self.total_tables = sum(len(sheet.tables) for sheet in self.sheets)

    @property
    def success_rate(self) -> float:
        """Percentage of sheets processed without errors."""
        if not self.sheets:
            return 0.0
        sheets_with_errors = sum(1 for sheet in self.sheets if sheet.errors)
        return (len(self.sheets) - sheets_with_errors) / len(self.sheets)

    def to_summary(self) -> dict[str, Any]:
        """Generate a summary of the detection results."""
        return {
            "file": str(self.file_info.path),
            "file_type": self.file_info.type.value,
            "total_sheets": len(self.sheets),
            "total_tables": self.total_tables,
            "detection_time": f"{self.detection_time:.2f}s",
            "success_rate": f"{self.success_rate * 100:.1f}%",
            "tables_by_sheet": {sheet.name: len(sheet.tables) for sheet in self.sheets},
        }
