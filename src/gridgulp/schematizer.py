"""Workbook schematization built on top of detection and extraction."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from gridgulp.config import Config
from gridgulp.extractors import DataFrameExtractor, HeaderDetectionResult
from gridgulp.models import DetectionResult
from gridgulp.models.file_info import FileInfo
from gridgulp.models.schema import (
    EntitySchema,
    ExtractionOrientation,
    FieldSchema,
    ProvenanceRef,
    RecordGroup,
    RegionRole,
    RegionSchema,
    RelationshipEdge,
    RelationshipType,
    SheetSchemaResult,
    WorkbookSchemaResult,
)
from gridgulp.models.sheet_data import CellData, FileData, SheetData
from gridgulp.models.table import TableInfo, TableRange

_MONTH_TOKEN_PATTERN = re.compile(
    r"\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|"
    r"sep|sept|september|oct|october|nov|november|dec|december|q[1-4]|20\d{2})\b",
    re.IGNORECASE,
)
_NON_ID_CHARS_PATTERN = re.compile(r"[^a-z0-9]+")
_CURRENCY_PATTERN = re.compile(r"\b(usd|eur|gbp|cad|aud|jpy|chf|sek|nok|dkk)\b", re.IGNORECASE)
_UNIT_PATTERN = re.compile(
    r"\(([^)]+)\)|\b(kg|kgs|lb|lbs|hours|hrs|items|units|%)\b", re.IGNORECASE
)


@dataclass(slots=True)
class _EntityCandidate:
    """Internal candidate used before cross-sheet entity merging."""

    sheet_name: str
    region_id: str
    region_role: RegionRole
    entity: EntitySchema


class WorkbookSchematizer:
    """Turn detected tables and loose worksheet regions into canonical workbook schema."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.extractor = DataFrameExtractor()

    async def schematize(
        self,
        *,
        file_path: str | Path,
        file_info: FileInfo,
        file_data: FileData,
        detection_result: DetectionResult,
    ) -> WorkbookSchemaResult:
        """Schematize a workbook from loaded reader and detection results."""
        start_time = time.time()

        sheet_payloads: list[tuple[SheetSchemaResult, list[_EntityCandidate]]] = []
        workbook_warnings: list[str] = []

        detection_by_sheet = {sheet.name: sheet for sheet in detection_result.sheets}

        for sheet_data in file_data.sheets:
            sheet_detection = detection_by_sheet.get(sheet_data.name)
            sheet_result, candidates = self._schematize_sheet(sheet_data, sheet_detection)
            sheet_payloads.append((sheet_result, candidates))
            workbook_warnings.extend(
                f"{sheet_data.name}: {warning}" for warning in sheet_result.warnings
            )

        merged_entities, entity_id_map = self._merge_entity_candidates(
            [candidate for _, candidates in sheet_payloads for candidate in candidates]
        )

        sheet_results = []
        for sheet_result, candidates in sheet_payloads:
            entity_ids = sorted(
                {
                    entity_id_map[candidate.entity.id]
                    for candidate in candidates
                    if candidate.entity.id in entity_id_map
                }
            )
            remapped_regions = []
            for region in sheet_result.regions:
                final_entity_id = (
                    entity_id_map.get(region.entity_id) if region.entity_id is not None else None
                )
                remapped_regions.append(region.model_copy(update={"entity_id": final_entity_id}))
            sheet_results.append(
                sheet_result.model_copy(
                    update={
                        "entity_ids": entity_ids,
                        "regions": remapped_regions,
                    }
                )
            )

        relationships = self._infer_relationships(merged_entities)

        methods_used = [
            *detection_result.methods_used,
            "workbook_schematizer",
            "region_classification",
            "field_normalization",
            "record_extraction",
            "relationship_inference",
        ]

        return WorkbookSchemaResult(
            file_info=file_info,
            workbook_metadata={
                **file_data.metadata,
                "file_format": file_data.file_format,
                "sheet_count": file_data.sheet_count,
                "source_path": str(Path(file_path)),
            },
            sheet_results=sheet_results,
            entities=merged_entities,
            relationships=relationships,
            warnings=workbook_warnings,
            provenance_summary=self._build_provenance_summary(sheet_results, merged_entities),
            schematization_time=time.time() - start_time,
            methods_used=methods_used,
            metadata={
                "detection_strategy": detection_result.metadata.get("detection_strategy"),
                "detection_total_tables": detection_result.total_tables,
                "detection_methods": detection_result.methods_used,
            },
        )

    def _schematize_sheet(
        self,
        sheet_data: SheetData,
        detection_sheet: Any | None,
    ) -> tuple[SheetSchemaResult, list[_EntityCandidate]]:
        warnings: list[str] = []
        regions: list[RegionSchema] = []
        candidates: list[_EntityCandidate] = []
        detected_tables = detection_sheet.tables if detection_sheet is not None else []

        for table in detected_tables:
            region, candidate, table_warnings = self._schematize_table_region(sheet_data, table)
            regions.append(region)
            warnings.extend(table_warnings)
            if candidate is not None:
                candidates.append(candidate)

        uncovered_regions = self._discover_uncovered_regions(sheet_data, detected_tables)
        regions.extend(uncovered_regions)

        return (
            SheetSchemaResult(
                sheet_name=sheet_data.name,
                regions=regions,
                warnings=warnings,
                metadata={
                    "detected_table_count": len(detected_tables),
                    "uncovered_region_count": len(uncovered_regions),
                },
            ),
            candidates,
        )

    def _schematize_table_region(
        self,
        sheet_data: SheetData,
        table: TableInfo,
    ) -> tuple[RegionSchema, _EntityCandidate | None, list[str]]:
        df, header_info, quality = self.extractor.extract_dataframe(sheet_data, table.range)
        warnings: list[str] = []
        role = self._classify_table_role(table, header_info, df)
        orientation = self._determine_orientation(role, header_info)
        region_id = f"{sheet_data.name}::{table.id}"
        region_provenance = [
            ProvenanceRef(
                sheet_name=sheet_data.name,
                cell_range=table.range,
                rationale=f"Detected by {table.detection_method}",
                confidence=table.confidence,
            )
        ]

        entity_candidate: _EntityCandidate | None = None
        entity_id: str | None = None
        headers = self._resolve_headers(table, header_info, df)

        if role in {RegionRole.DATA_TABLE, RegionRole.MATRIX_BLOCK}:
            entity_candidate = self._build_entity_candidate(
                sheet_data=sheet_data,
                table=table,
                region_id=region_id,
                role=role,
                header_info=header_info,
                df=df,
                quality=quality,
                headers=headers,
                orientation=orientation,
            )
            entity_id = entity_candidate.entity.id

            if not entity_candidate.entity.record_groups:
                warnings.append(
                    f"Table {table.range.excel_range} could not be converted into normalized records."
                )
            if entity_candidate.entity.confidence < 0.75:
                warnings.append(
                    f"Low-confidence schema inference for table {table.range.excel_range}."
                )
            if role == RegionRole.MATRIX_BLOCK:
                warnings.append(
                    f"Table {table.range.excel_range} looks matrix-like; records may require review."
                )

        region = RegionSchema(
            id=region_id,
            sheet_name=sheet_data.name,
            range=table.range,
            role=role,
            confidence=self._combine_confidence(table.confidence, quality),
            suggested_name=table.suggested_name,
            source_table_id=table.id,
            entity_id=entity_id,
            headers=headers,
            provenance=region_provenance,
            metadata={
                "detection_method": table.detection_method,
                "table_confidence": table.confidence,
                "extraction_quality": quality,
                "orientation": orientation.value,
                "header_rows": header_info.header_rows if header_info else 0,
                "table_metadata": table.metadata,
            },
        )
        return region, entity_candidate, warnings

    def _build_entity_candidate(
        self,
        *,
        sheet_data: SheetData,
        table: TableInfo,
        region_id: str,
        role: RegionRole,
        header_info: HeaderDetectionResult | None,
        df: pd.DataFrame | None,
        quality: float,
        headers: list[str],
        orientation: ExtractionOrientation,
    ) -> _EntityCandidate:
        entity_name = self._suggest_entity_name(sheet_data, table, headers)
        temp_entity_id = f"{sheet_data.name}::{self._slugify(entity_name or table.id)}"
        fields = self._build_fields(
            sheet_data=sheet_data,
            table=table,
            header_info=header_info,
            df=df,
            headers=headers,
            orientation=orientation,
            entity_id=temp_entity_id,
        )
        record_group = self._build_record_group(
            sheet_data=sheet_data,
            table=table,
            entity_id=temp_entity_id,
            df=df,
            orientation=orientation,
            confidence=self._combine_confidence(table.confidence, quality),
        )

        entity = EntitySchema(
            id=temp_entity_id,
            suggested_name=entity_name,
            source_regions=[region_id],
            sheet_names=[sheet_data.name],
            fields=fields,
            extraction_orientation=orientation,
            record_groups=[record_group] if record_group is not None else [],
            record_count_estimate=record_group.record_count if record_group is not None else 0,
            confidence=self._combine_confidence(table.confidence, quality),
            provenance=[
                ProvenanceRef(
                    sheet_name=sheet_data.name,
                    cell_range=table.range,
                    rationale="Entity built from detected table region",
                    confidence=self._combine_confidence(table.confidence, quality),
                )
            ],
            metadata={
                "region_role": role.value,
                "source_table_id": table.id,
                "source_range": table.range.excel_range,
                "detection_method": table.detection_method,
            },
        )
        return _EntityCandidate(
            sheet_name=sheet_data.name,
            region_id=region_id,
            region_role=role,
            entity=entity,
        )

    def _build_fields(
        self,
        *,
        sheet_data: SheetData,
        table: TableInfo,
        header_info: HeaderDetectionResult | None,
        df: pd.DataFrame | None,
        headers: list[str],
        orientation: ExtractionOrientation,
        entity_id: str,
    ) -> list[FieldSchema]:
        field_names = headers if headers else self._fallback_headers(table.range, orientation)
        field_id_counts: dict[str, int] = defaultdict(int)
        fields: list[FieldSchema] = []

        for index, display_name in enumerate(field_names):
            base_field_id = self._normalize_field_id(display_name) or f"field_{index + 1}"
            field_id_counts[base_field_id] += 1
            field_id = (
                base_field_id
                if field_id_counts[base_field_id] == 1
                else f"{base_field_id}_{field_id_counts[base_field_id]}"
            )
            source_columns, source_cells = self._field_sources(
                table.range,
                header_info,
                index,
                orientation,
            )
            series = self._field_series(df, index)
            physical_type = self._infer_physical_type(series)
            currency = self._extract_currency(display_name)
            unit = self._extract_unit(display_name)
            semantic_type = self._infer_semantic_type(
                display_name=display_name,
                physical_type=physical_type,
                series=series,
                currency=currency,
                unit=unit,
            )
            nullable = bool(series.isna().any()) if series is not None else None
            field_confidence = self._field_confidence(display_name, physical_type, header_info)

            fields.append(
                FieldSchema(
                    id=field_id,
                    display_name=display_name,
                    semantic_type=semantic_type,
                    physical_type=physical_type,
                    unit=unit,
                    currency=currency,
                    nullable=nullable,
                    confidence=field_confidence,
                    source_columns=source_columns,
                    source_cells=source_cells,
                    provenance=[
                        ProvenanceRef(
                            sheet_name=sheet_data.name,
                            cell_range=table.range,
                            header_cells=source_cells,
                            source_columns=source_columns,
                            source_cells=source_cells,
                            rationale=f"Field normalized for entity {entity_id}",
                            confidence=field_confidence,
                        )
                    ],
                    metadata={
                        "column_index": index,
                        "sample_values": self._sample_values(series),
                    },
                )
            )

        return fields

    def _build_record_group(
        self,
        *,
        sheet_data: SheetData,
        table: TableInfo,
        entity_id: str,
        df: pd.DataFrame | None,
        orientation: ExtractionOrientation,
        confidence: float,
    ) -> RecordGroup | None:
        if df is None:
            return None

        records = self._records_from_dataframe(df)
        return RecordGroup(
            id=f"{entity_id}::records::{table.range.excel_range}",
            entity_id=entity_id,
            orientation=orientation,
            record_count=len(records),
            records=records,
            confidence=confidence,
            provenance=[
                ProvenanceRef(
                    sheet_name=sheet_data.name,
                    cell_range=table.range,
                    rationale="Records extracted from normalized table dataframe",
                    confidence=confidence,
                )
            ],
            metadata={
                "shape": list(df.shape),
                "columns": [str(column) for column in df.columns],
            },
        )

    def _discover_uncovered_regions(
        self,
        sheet_data: SheetData,
        detected_tables: list[TableInfo],
    ) -> list[RegionSchema]:
        covered_cells = {
            (row, col)
            for table in detected_tables
            for row in range(table.range.start_row, table.range.end_row + 1)
            for col in range(table.range.start_col, table.range.end_col + 1)
        }
        uncovered = {
            (cell.row, cell.column): cell
            for cell in sheet_data.get_non_empty_cells()
            if (cell.row, cell.column) not in covered_cells
        }
        regions: list[RegionSchema] = []

        while uncovered:
            start = next(iter(uncovered))
            cluster = self._pop_cluster(start, uncovered)
            table_range = self._cluster_range(cluster)
            role, confidence, suggested_name = self._classify_uncovered_cluster(
                cluster, table_range
            )
            regions.append(
                RegionSchema(
                    id=f"{sheet_data.name}::region::{table_range.excel_range}",
                    sheet_name=sheet_data.name,
                    range=table_range,
                    role=role,
                    confidence=confidence,
                    suggested_name=suggested_name,
                    headers=[],
                    provenance=[
                        ProvenanceRef(
                            sheet_name=sheet_data.name,
                            cell_range=table_range,
                            source_cells=[cell.excel_address for cell in cluster[:8]],
                            rationale="Classified from uncovered worksheet content",
                            confidence=confidence,
                        )
                    ],
                    metadata={
                        "cell_count": len(cluster),
                        "sample_values": [
                            str(cell.value) for cell in cluster[:6] if cell.value is not None
                        ],
                    },
                )
            )

        return regions

    def _merge_entity_candidates(
        self,
        candidates: list[_EntityCandidate],
    ) -> tuple[list[EntitySchema], dict[str, str]]:
        groups: list[list[_EntityCandidate]] = []
        for candidate in candidates:
            placed = False
            for group in groups:
                if self._should_merge(group[0].entity, candidate.entity):
                    group.append(candidate)
                    placed = True
                    break
            if not placed:
                groups.append([candidate])

        merged_entities: list[EntitySchema] = []
        entity_id_map: dict[str, str] = {}
        used_entity_ids: set[str] = set()

        for index, group in enumerate(groups, start=1):
            merged_entity = self._merge_group(group, index, used_entity_ids)
            merged_entities.append(merged_entity)
            for candidate in group:
                entity_id_map[candidate.entity.id] = merged_entity.id

        return merged_entities, entity_id_map

    def _merge_group(
        self,
        group: list[_EntityCandidate],
        ordinal: int,
        used_entity_ids: set[str],
    ) -> EntitySchema:
        first = group[0].entity
        base_name = self._canonical_entity_name(first.suggested_name)
        base_id = self._slugify(base_name) or f"entity_{ordinal}"
        final_id = base_id
        suffix = 2
        while final_id in used_entity_ids:
            final_id = f"{base_id}_{suffix}"
            suffix += 1
        used_entity_ids.add(final_id)

        field_map: dict[str, FieldSchema] = {}
        for candidate in group:
            for field in candidate.entity.fields:
                if field.id not in field_map:
                    field_map[field.id] = field
                    continue
                field_map[field.id] = self._merge_field(field_map[field.id], field)

        record_groups: list[RecordGroup] = []
        for candidate in group:
            for record_group in candidate.entity.record_groups:
                record_groups.append(record_group.model_copy(update={"entity_id": final_id}))

        provenance: list[ProvenanceRef] = []
        for candidate in group:
            provenance.extend(candidate.entity.provenance)

        confidences = [candidate.entity.confidence for candidate in group]
        orientations = [candidate.entity.extraction_orientation for candidate in group]
        dominant_orientation = max(
            set(orientations),
            key=lambda orientation: orientations.count(orientation),
        )

        return EntitySchema(
            id=final_id,
            suggested_name=base_name,
            source_regions=sorted(
                {region_id for candidate in group for region_id in candidate.entity.source_regions}
            ),
            sheet_names=sorted(
                {sheet_name for candidate in group for sheet_name in candidate.entity.sheet_names}
            ),
            fields=list(field_map.values()),
            extraction_orientation=dominant_orientation,
            record_groups=record_groups,
            record_count_estimate=sum(group.record_count for group in record_groups),
            confidence=sum(confidences) / len(confidences),
            provenance=provenance,
            metadata={
                "merged_from": [candidate.entity.id for candidate in group],
                "source_sheet_count": len({candidate.sheet_name for candidate in group}),
            },
        )

    def _merge_field(self, left: FieldSchema, right: FieldSchema) -> FieldSchema:
        return left.model_copy(
            update={
                "confidence": max(left.confidence, right.confidence),
                "source_columns": sorted(set(left.source_columns) | set(right.source_columns)),
                "source_cells": sorted(set(left.source_cells) | set(right.source_cells)),
                "provenance": [*left.provenance, *right.provenance],
                "metadata": {
                    **left.metadata,
                    **right.metadata,
                },
            }
        )

    def _infer_relationships(self, entities: list[EntitySchema]) -> list[RelationshipEdge]:
        relationships: list[RelationshipEdge] = []
        seen_pairs: set[tuple[str, str, tuple[str, ...]]] = set()

        for index, left in enumerate(entities):
            for right in entities[index + 1 :]:
                shared_field_ids = sorted(
                    {field.id for field in left.fields} & {field.id for field in right.fields}
                )
                if not shared_field_ids:
                    continue

                from_entity, to_entity = self._relationship_direction(left, right, shared_field_ids)
                relationship_type, confidence, rationale = self._relationship_kind(
                    from_entity,
                    to_entity,
                    shared_field_ids,
                )
                pair_key = (from_entity.id, to_entity.id, tuple(shared_field_ids))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                provenance = [
                    *from_entity.provenance[:1],
                    *to_entity.provenance[:1],
                ]
                relationships.append(
                    RelationshipEdge(
                        id=f"{from_entity.id}__{to_entity.id}__{relationship_type.value}",
                        from_entity_id=from_entity.id,
                        to_entity_id=to_entity.id,
                        relationship_type=relationship_type,
                        shared_field_ids=shared_field_ids,
                        confidence=confidence,
                        rationale=rationale,
                        provenance=provenance,
                        metadata={
                            "from_records": from_entity.record_count_estimate,
                            "to_records": to_entity.record_count_estimate,
                        },
                    )
                )

        return relationships

    def _relationship_direction(
        self,
        left: EntitySchema,
        right: EntitySchema,
        shared_field_ids: list[str],
    ) -> tuple[EntitySchema, EntitySchema]:
        left_lookup_score = self._lookup_score(left, shared_field_ids)
        right_lookup_score = self._lookup_score(right, shared_field_ids)

        if left_lookup_score > right_lookup_score:
            return right, left
        if right_lookup_score > left_lookup_score:
            return left, right

        if left.record_count_estimate >= right.record_count_estimate:
            return left, right
        return right, left

    def _relationship_kind(
        self,
        from_entity: EntitySchema,
        to_entity: EntitySchema,
        shared_field_ids: list[str],
    ) -> tuple[RelationshipType, float, str]:
        lookup_fields = [
            field_id for field_id in shared_field_ids if self._looks_like_key(field_id)
        ]
        if lookup_fields and to_entity.record_count_estimate <= from_entity.record_count_estimate:
            return (
                RelationshipType.LOOKUP_REFERENCE,
                0.78,
                f"{from_entity.suggested_name} likely references {to_entity.suggested_name} via {', '.join(lookup_fields)}.",
            )

        return (
            RelationshipType.SHARED_FIELD_SET,
            0.62,
            f"{from_entity.suggested_name} and {to_entity.suggested_name} share normalized fields {', '.join(shared_field_ids)}.",
        )

    def _classify_table_role(
        self,
        table: TableInfo,
        header_info: HeaderDetectionResult | None,
        df: pd.DataFrame | None,
    ) -> RegionRole:
        if header_info and header_info.table_type in {"plate_map", "time_series"}:
            return RegionRole.MATRIX_BLOCK
        if header_info and header_info.orientation == "horizontal":
            return RegionRole.DATA_TABLE
        if df is not None and min(df.shape) >= 2:
            return RegionRole.DATA_TABLE
        if table.header_info and table.header_info.is_multi_row:
            return RegionRole.MATRIX_BLOCK
        return RegionRole.DATA_TABLE

    def _determine_orientation(
        self,
        role: RegionRole,
        header_info: HeaderDetectionResult | None,
    ) -> ExtractionOrientation:
        if role == RegionRole.MATRIX_BLOCK:
            return ExtractionOrientation.MATRIX
        if header_info and header_info.orientation == "horizontal":
            return ExtractionOrientation.COLUMN_RECORDS
        return ExtractionOrientation.ROW_RECORDS

    def _resolve_headers(
        self,
        table: TableInfo,
        header_info: HeaderDetectionResult | None,
        df: pd.DataFrame | None,
    ) -> list[str]:
        if header_info and header_info.headers:
            return [str(header) for header in header_info.headers]
        if df is not None:
            return [str(column) for column in df.columns]
        if table.headers:
            return [str(header) for header in table.headers]
        return []

    def _field_sources(
        self,
        table_range: TableRange,
        header_info: HeaderDetectionResult | None,
        index: int,
        orientation: ExtractionOrientation,
    ) -> tuple[list[str], list[str]]:
        if orientation == ExtractionOrientation.COLUMN_RECORDS:
            header_column = table_range.start_col
            column_letter = self._column_letter(header_column)
            cell_row = table_range.start_row + index
            return [column_letter], [f"{column_letter}{cell_row + 1}"]

        source_col = table_range.start_col + index
        column_letter = self._column_letter(source_col)
        header_rows = (
            header_info.header_rows if header_info is not None and header_info.header_rows else 1
        )
        source_cells = [
            f"{column_letter}{table_range.start_row + row_offset + 1}"
            for row_offset in range(header_rows)
        ]
        return [column_letter], source_cells

    def _field_series(self, df: pd.DataFrame | None, index: int) -> pd.Series | None:
        if df is None or index >= len(df.columns):
            return None
        return df.iloc[:, index]

    def _infer_physical_type(self, series: pd.Series | None) -> str:
        if series is None or series.empty:
            return "unknown"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_integer_dtype(series):
            return "integer"
        if pd.api.types.is_float_dtype(series):
            return "number"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        return "string"

    def _field_confidence(
        self,
        display_name: str,
        physical_type: str,
        header_info: HeaderDetectionResult | None,
    ) -> float:
        confidence = 0.65
        if display_name and not display_name.lower().startswith("column_"):
            confidence += 0.1
        if physical_type != "unknown":
            confidence += 0.1
        if header_info is not None:
            confidence += min(0.15, header_info.confidence * 0.15)
        return min(confidence, 0.95)

    def _infer_semantic_type(
        self,
        *,
        display_name: str,
        physical_type: str,
        series: pd.Series | None,
        currency: str | None,
        unit: str | None,
    ) -> str:
        normalized = display_name.lower()
        if any(token in normalized for token in ("id", "code", "sku", "key")):
            return "identifier"
        if "email" in normalized:
            return "email"
        if "phone" in normalized:
            return "phone"
        if "date" in normalized:
            return "date"
        if "time" in normalized:
            return "datetime"
        if currency is not None or "$" in normalized:
            return "currency_amount"
        if "%" in normalized or unit == "%":
            return "percentage"
        if any(token in normalized for token in ("qty", "quantity", "count", "units")):
            return "quantity"
        if any(token in normalized for token in ("price", "revenue", "cost", "amount", "sales")):
            return "measure"
        if any(token in normalized for token in ("name", "title", "label")):
            return "name"
        if physical_type in {"integer", "number"}:
            return "measure"
        if physical_type == "datetime":
            return "datetime"
        if series is not None and series.nunique(dropna=True) <= 20 and len(series) > 0:
            return "category"
        return "text"

    def _extract_currency(self, display_name: str) -> str | None:
        if "$" in display_name:
            return "USD"
        match = _CURRENCY_PATTERN.search(display_name)
        return match.group(1).upper() if match else None

    def _extract_unit(self, display_name: str) -> str | None:
        match = _UNIT_PATTERN.search(display_name)
        if not match:
            return None
        unit = match.group(1) or match.group(2)
        return unit.lower() if unit else None

    def _sample_values(self, series: pd.Series | None) -> list[Any]:
        if series is None:
            return []
        return list(series.dropna().tolist()[:3])

    def _records_from_dataframe(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for record in df.where(pd.notna(df), None).to_dict(orient="records"):
            normalized_record = {}
            for key, value in record.items():
                if hasattr(value, "item"):
                    with suppress(Exception):
                        value = value.item()
                normalized_record[str(key)] = value
            records.append(normalized_record)
        return records

    def _build_provenance_summary(
        self,
        sheet_results: list[SheetSchemaResult],
        entities: list[EntitySchema],
    ) -> dict[str, Any]:
        return {
            "sheet_count": len(sheet_results),
            "region_count": sum(len(sheet.regions) for sheet in sheet_results),
            "entity_count": len(entities),
            "field_count": sum(len(entity.fields) for entity in entities),
            "record_group_count": sum(len(entity.record_groups) for entity in entities),
        }

    def _combine_confidence(self, left: float, right: float) -> float:
        if right <= 0:
            return left
        return min((left + right) / 2, 0.99)

    def _fallback_headers(
        self,
        table_range: TableRange,
        orientation: ExtractionOrientation,
    ) -> list[str]:
        count = (
            table_range.row_count
            if orientation == ExtractionOrientation.COLUMN_RECORDS
            else table_range.col_count
        )
        return [f"column_{index + 1}" for index in range(count)]

    def _suggest_entity_name(
        self,
        sheet_data: SheetData,
        table: TableInfo,
        headers: list[str],
    ) -> str:
        if table.suggested_name:
            return self._humanize_name(table.suggested_name)
        sheet_name = self._canonical_entity_name(self._humanize_name(sheet_data.name))
        if self._slugify(sheet_name) not in {"sheet", "sheet1", "sheet2", "sheet3"}:
            return sheet_name
        if headers:
            lead_header = headers[0].strip()
            return self._humanize_name(lead_header.replace(" id", "").replace("_", " "))
        return self._humanize_name(sheet_data.name)

    def _canonical_entity_name(self, name: str) -> str:
        cleaned = _MONTH_TOKEN_PATTERN.sub(" ", name).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned or name

    def _should_merge(self, left: EntitySchema, right: EntitySchema) -> bool:
        left_name = self._slugify(self._canonical_entity_name(left.suggested_name))
        right_name = self._slugify(self._canonical_entity_name(right.suggested_name))
        left_fields = {field.id for field in left.fields}
        right_fields = {field.id for field in right.fields}
        overlap = self._jaccard(left_fields, right_fields)

        if left_name and right_name and left_name == right_name and overlap >= 0.5:
            return True
        return overlap >= 0.8 and left.extraction_orientation == right.extraction_orientation

    def _lookup_score(self, entity: EntitySchema, shared_field_ids: list[str]) -> float:
        key_bonus = sum(0.1 for field_id in shared_field_ids if self._looks_like_key(field_id))
        size_bonus = 0.15 if entity.record_count_estimate <= 200 else 0.0
        field_bonus = 0.1 if len(entity.fields) <= 8 else 0.0
        return key_bonus + size_bonus + field_bonus

    def _looks_like_key(self, field_id: str) -> bool:
        return any(
            token in field_id for token in ("id", "code", "name", "type", "category", "status")
        )

    def _pop_cluster(
        self,
        start: tuple[int, int],
        uncovered: dict[tuple[int, int], CellData],
    ) -> list[CellData]:
        stack = [start]
        cluster: list[CellData] = []
        while stack:
            position = stack.pop()
            cell = uncovered.pop(position, None)
            if cell is None:
                continue
            cluster.append(cell)
            row, col = position
            for next_row in range(row - 1, row + 2):
                for next_col in range(col - 1, col + 2):
                    if (next_row, next_col) in uncovered:
                        stack.append((next_row, next_col))
        return sorted(cluster, key=lambda cell: (cell.row, cell.column))

    def _cluster_range(self, cluster: list[CellData]) -> TableRange:
        return TableRange(
            start_row=min(cell.row for cell in cluster),
            end_row=max(cell.row for cell in cluster),
            start_col=min(cell.column for cell in cluster),
            end_col=max(cell.column for cell in cluster),
        )

    def _classify_uncovered_cluster(
        self,
        cluster: list[CellData],
        table_range: TableRange,
    ) -> tuple[RegionRole, float, str | None]:
        text = " ".join(str(cell.value) for cell in cluster if cell.value is not None).lower()
        bold_ratio = sum(1 for cell in cluster if cell.is_bold) / len(cluster) if cluster else 0.0
        row_span = table_range.row_count
        col_span = table_range.col_count

        if table_range.start_row <= 2 and row_span <= 2 and bold_ratio >= 0.4:
            return RegionRole.TITLE_BLOCK, 0.82, "sheet_title"
        if "legend" in text or "key" in text:
            return RegionRole.LEGEND_BLOCK, 0.8, "legend"
        if "note" in text or "comment" in text or "remarks" in text:
            return RegionRole.NOTES_BLOCK, 0.78, "notes"
        if col_span <= 2 and row_span >= 2:
            if any(":" in str(cell.value) for cell in cluster if cell.value is not None):
                return RegionRole.FORM_BLOCK, 0.74, "form_block"
            return RegionRole.METADATA_BLOCK, 0.76, "metadata"
        if row_span <= 2 and any(
            len(str(cell.value)) > 20 for cell in cluster if cell.value is not None
        ):
            return RegionRole.NOTES_BLOCK, 0.68, "notes"
        return RegionRole.METADATA_BLOCK, 0.55, None

    def _normalize_field_id(self, value: str) -> str:
        value = value.strip().lower()
        value = _UNIT_PATTERN.sub("", value)
        value = _CURRENCY_PATTERN.sub("", value)
        value = value.replace("$", " usd ")
        return self._slugify(value)

    def _humanize_name(self, value: str) -> str:
        cleaned = value.replace("_", " ").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.title() if cleaned else value

    def _slugify(self, value: str) -> str:
        lowered = value.strip().lower()
        lowered = _NON_ID_CHARS_PATTERN.sub("_", lowered)
        lowered = lowered.strip("_")
        return lowered

    def _column_letter(self, col: int) -> str:
        result = ""
        current = col
        while current >= 0:
            result = chr(current % 26 + ord("A")) + result
            current = current // 26 - 1
        return result

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
