"""Canonical workbook schematization result models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .file_info import FileInfo
from .table import TableRange


class RegionRole(StrEnum):
    """Supported semantic region roles inside a worksheet."""

    DATA_TABLE = "data_table"
    TITLE_BLOCK = "title_block"
    METADATA_BLOCK = "metadata_block"
    NOTES_BLOCK = "notes_block"
    LEGEND_BLOCK = "legend_block"
    FORM_BLOCK = "form_block"
    MATRIX_BLOCK = "matrix_block"
    UNKNOWN = "unknown"


class ExtractionOrientation(StrEnum):
    """Canonical record extraction orientation."""

    ROW_RECORDS = "row_records"
    COLUMN_RECORDS = "column_records"
    MATRIX = "matrix"


class RelationshipType(StrEnum):
    """Supported inter-entity relationship types."""

    LOOKUP_REFERENCE = "lookup_reference"
    SHARED_FIELD_SET = "shared_field_set"


class ProvenanceRef(BaseModel):
    """Source-trace information for any schema decision."""

    model_config = ConfigDict(strict=True)

    sheet_name: str = Field(..., description="Sheet containing the source evidence")
    cell_range: TableRange | None = Field(
        None,
        description="Structured range containing the source evidence",
    )
    header_cells: list[str] = Field(
        default_factory=list,
        description="Header cells used to derive the semantic meaning",
    )
    source_columns: list[str] = Field(
        default_factory=list,
        description="Excel column letters tied to the evidence",
    )
    source_cells: list[str] = Field(
        default_factory=list,
        description="Specific source cells tied to the evidence",
    )
    rationale: str | None = Field(None, description="Short explanation for the linkage")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in this reference")


class FieldSchema(BaseModel):
    """Canonical workbook field definition."""

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Stable field id")
    display_name: str = Field(..., description="Original or normalized field display name")
    semantic_type: str = Field(..., description="Semantic type such as identifier or date")
    physical_type: str = Field(..., description="Physical storage type such as string or number")
    unit: str | None = Field(None, description="Detected unit if present")
    currency: str | None = Field(None, description="Detected currency if present")
    nullable: bool | None = Field(None, description="Whether the field appears nullable")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Field normalization confidence")
    source_columns: list[str] = Field(
        default_factory=list,
        description="Excel column letters contributing to this field",
    )
    source_cells: list[str] = Field(
        default_factory=list,
        description="Header or representative cells contributing to this field",
    )
    provenance: list[ProvenanceRef] = Field(
        default_factory=list,
        description="Detailed provenance for the field definition",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional field metadata")


class RecordGroup(BaseModel):
    """Extracted records for one entity occurrence."""

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Stable record-group id")
    entity_id: str = Field(..., description="Owning entity id")
    orientation: ExtractionOrientation = Field(..., description="Record orientation")
    record_count: int = Field(..., ge=0, description="Number of records in this group")
    records: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Normalized records extracted from the source region",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the extraction")
    provenance: list[ProvenanceRef] = Field(
        default_factory=list,
        description="Source evidence for this record group",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional record metadata")


class RegionSchema(BaseModel):
    """Schema understanding for a contiguous worksheet region."""

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Stable region id")
    sheet_name: str = Field(..., description="Sheet containing the region")
    range: TableRange = Field(..., description="Region bounds")
    role: RegionRole = Field(..., description="Semantic role for the region")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the role")
    suggested_name: str | None = Field(None, description="Suggested semantic label")
    source_table_id: str | None = Field(
        None,
        description="Originating table id when the region came from table detection",
    )
    entity_id: str | None = Field(
        None,
        description="Canonical entity id produced from this region when applicable",
    )
    headers: list[str] = Field(default_factory=list, description="Headers observed for the region")
    provenance: list[ProvenanceRef] = Field(
        default_factory=list,
        description="Evidence used to classify the region",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional region metadata")


class EntitySchema(BaseModel):
    """Canonical entity inferred from one or more regions."""

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Stable entity id")
    suggested_name: str = Field(..., description="Suggested entity name")
    source_regions: list[str] = Field(
        default_factory=list,
        description="Region ids that contributed to this entity",
    )
    sheet_names: list[str] = Field(
        default_factory=list,
        description="Sheets where this entity appears",
    )
    fields: list[FieldSchema] = Field(default_factory=list, description="Normalized entity fields")
    extraction_orientation: ExtractionOrientation = Field(..., description="Entity record layout")
    record_groups: list[RecordGroup] = Field(
        default_factory=list,
        description="Extracted record groups contributing to the entity",
    )
    record_count_estimate: int = Field(..., ge=0, description="Estimated total record count")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Entity confidence")
    provenance: list[ProvenanceRef] = Field(
        default_factory=list,
        description="Provenance references across all source regions",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional entity metadata")


class RelationshipEdge(BaseModel):
    """Inferred relationship between two entities."""

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Stable relationship id")
    from_entity_id: str = Field(..., description="Source entity id")
    to_entity_id: str = Field(..., description="Target entity id")
    relationship_type: RelationshipType = Field(..., description="Relationship type")
    shared_field_ids: list[str] = Field(
        default_factory=list,
        description="Normalized field ids supporting the link",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Relationship confidence")
    rationale: str = Field(..., description="Short explanation for the inferred link")
    provenance: list[ProvenanceRef] = Field(
        default_factory=list,
        description="Provenance backing the inferred relationship",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SheetSchemaResult(BaseModel):
    """Schema understanding for one sheet."""

    model_config = ConfigDict(strict=True)

    sheet_name: str = Field(..., description="Sheet name")
    regions: list[RegionSchema] = Field(default_factory=list, description="All classified regions")
    entity_ids: list[str] = Field(
        default_factory=list,
        description="Entity ids that appear on this sheet",
    )
    warnings: list[str] = Field(default_factory=list, description="Sheet-level warnings")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional sheet metadata")


class WorkbookSchemaResult(BaseModel):
    """Canonical JSON-first workbook schema result."""

    model_config = ConfigDict(strict=True)

    file_info: FileInfo = Field(..., description="Source workbook file information")
    workbook_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Workbook metadata surfaced by the reader",
    )
    sheet_results: list[SheetSchemaResult] = Field(
        default_factory=list,
        description="Per-sheet schema understanding",
    )
    entities: list[EntitySchema] = Field(default_factory=list, description="Canonical entities")
    relationships: list[RelationshipEdge] = Field(
        default_factory=list,
        description="Relationships inferred across entities",
    )
    warnings: list[str] = Field(default_factory=list, description="Workbook-level warnings")
    provenance_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary provenance statistics",
    )
    schematization_time: float = Field(..., ge=0.0, description="Total runtime in seconds")
    methods_used: list[str] = Field(
        default_factory=list,
        description="Methods contributing to the schema result",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When schematization completed",
    )

    @field_validator("methods_used")
    @classmethod
    def _dedupe_methods(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @property
    def total_regions(self) -> int:
        """Total regions across all sheets."""
        return sum(len(sheet.regions) for sheet in self.sheet_results)

    @property
    def total_entities(self) -> int:
        """Total canonical entities."""
        return len(self.entities)
