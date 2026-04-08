"""Schema-level tests for workbook schematization."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridgulp.config import Config
from gridgulp.models import DetectionResult, FileInfo, FileType, SheetResult, TableInfo, TableRange
from gridgulp.models.schema import ExtractionOrientation, RegionRole, RelationshipType
from gridgulp.models.sheet_data import CellData, FileData, SheetData
from gridgulp.schematizer import WorkbookSchematizer


def _cell(value, row: int, column: int, *, is_bold: bool = False) -> CellData:
    data_type = "number" if isinstance(value, int | float) else "string"
    return CellData(
        value=value,
        data_type=data_type,
        row=row,
        column=column,
        is_bold=is_bold,
    )


def _sheet_from_rows(
    name: str,
    rows: list[list[object | None]],
    *,
    bold_header: bool = True,
) -> SheetData:
    sheet = SheetData(name=name)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            if value is None:
                continue
            sheet.set_cell(
                row_index,
                column_index,
                _cell(value, row_index, column_index, is_bold=bold_header and row_index == 0),
            )
    return sheet


def _table(
    table_id: str,
    excel_range: str,
    *,
    sheet_name: str,
    suggested_name: str | None = None,
    confidence: float = 0.95,
) -> TableInfo:
    return TableInfo(
        id=table_id,
        range=TableRange.from_excel(excel_range),
        confidence=confidence,
        detection_method="heuristic_table",
        sheet_name=sheet_name,
        suggested_name=suggested_name,
    )


def _detection_result(
    file_path: str, sheets: list[tuple[SheetData, list[TableInfo]]]
) -> DetectionResult:
    return DetectionResult(
        file_info=FileInfo(path=Path(file_path), type=FileType.XLSX, size=1024),
        sheets=[
            SheetResult(
                name=sheet.name,
                tables=tables,
                processing_time=0.1,
                metadata={"methods_used": ["heuristic_table"]},
            )
            for sheet, tables in sheets
        ],
        detection_time=0.3,
        methods_used=["heuristic_table"],
        metadata={"detection_strategy": "heuristic_only"},
    )


@pytest.mark.asyncio
async def test_schematize_single_table_workbook() -> None:
    """A straightforward row-oriented table should become one canonical entity."""
    sheet = _sheet_from_rows(
        "Sales",
        [
            ["Product", "Price", "Quantity"],
            ["Apples", 1.25, 10],
            ["Bananas", 0.8, 12],
        ],
    )
    file_data = FileData(sheets=[sheet], metadata={}, file_format="xlsx")
    detection = _detection_result(
        "sales.xlsx",
        [
            (
                sheet,
                [
                    _table(
                        "sales_table", "A1:C3", sheet_name="Sales", suggested_name="sales_records"
                    )
                ],
            )
        ],
    )

    result = await WorkbookSchematizer(Config(detection_strategy="heuristic_only")).schematize(
        file_path="sales.xlsx",
        file_info=detection.file_info,
        file_data=file_data,
        detection_result=detection,
    )

    assert result.total_entities == 1
    entity = result.entities[0]
    assert entity.suggested_name == "Sales Records"
    assert entity.record_count_estimate == 2
    assert entity.extraction_orientation == ExtractionOrientation.ROW_RECORDS
    assert [field.id for field in entity.fields] == ["product", "price", "quantity"]
    assert result.sheet_results[0].regions[0].role == RegionRole.DATA_TABLE


@pytest.mark.asyncio
async def test_schematize_merges_repeated_entities_across_sheets() -> None:
    """Monthly sheets with the same shape should merge into one canonical entity."""
    january = _sheet_from_rows(
        "Jan Sales",
        [
            ["Customer ID", "Amount"],
            ["C-001", 100],
            ["C-002", 150],
        ],
    )
    february = _sheet_from_rows(
        "Feb Sales",
        [
            ["Customer ID", "Amount"],
            ["C-003", 125],
            ["C-004", 175],
        ],
    )
    file_data = FileData(sheets=[january, february], metadata={}, file_format="xlsx")
    detection = _detection_result(
        "sales_by_month.xlsx",
        [
            (january, [_table("jan_sales", "A1:B3", sheet_name="Jan Sales")]),
            (february, [_table("feb_sales", "A1:B3", sheet_name="Feb Sales")]),
        ],
    )

    result = await WorkbookSchematizer(Config(detection_strategy="heuristic_only")).schematize(
        file_path="sales_by_month.xlsx",
        file_info=detection.file_info,
        file_data=file_data,
        detection_result=detection,
    )

    assert result.total_entities == 1
    entity = result.entities[0]
    assert entity.suggested_name == "Sales"
    assert entity.sheet_names == ["Feb Sales", "Jan Sales"]
    assert len(entity.source_regions) == 2
    assert entity.record_count_estimate == 4
    assert all(sheet.entity_ids == ["sales"] for sheet in result.sheet_results)


@pytest.mark.asyncio
async def test_schematize_classifies_title_and_notes_regions() -> None:
    """Non-table content should still surface as classified worksheet regions."""
    sheet = _sheet_from_rows(
        "Summary",
        [
            ["Quarterly Sales Report", "Quarterly Sales Report", None, None, None, None, None],
            [None, None, None, None, None, None, None],
            ["Product", "Amount", None, None, None, "Note:", "Provisional figures"],
            ["Apples", 100, None, None, None, "Comment", "Board review pending"],
            ["Bananas", 120, None, None, None, None, None],
        ],
        bold_header=False,
    )
    for column in range(2):
        sheet.get_cell(0, column).is_bold = True  # type: ignore[union-attr]

    file_data = FileData(sheets=[sheet], metadata={}, file_format="xlsx")
    detection = _detection_result(
        "summary.xlsx",
        [(sheet, [_table("main_table", "A3:B5", sheet_name="Summary")])],
    )

    result = await WorkbookSchematizer(Config(detection_strategy="heuristic_only")).schematize(
        file_path="summary.xlsx",
        file_info=detection.file_info,
        file_data=file_data,
        detection_result=detection,
    )

    roles = {region.role for region in result.sheet_results[0].regions}
    assert RegionRole.DATA_TABLE in roles
    assert RegionRole.TITLE_BLOCK in roles
    assert RegionRole.NOTES_BLOCK in roles


@pytest.mark.asyncio
async def test_schematize_flags_matrix_like_regions() -> None:
    """Plate-map style layouts should be marked as matrix blocks with review warnings."""
    sheet = _sheet_from_rows(
        "Plate",
        [
            ["", 1, 2, 3],
            ["A", "x", "y", "z"],
            ["B", "q", "r", "s"],
        ],
    )
    file_data = FileData(sheets=[sheet], metadata={}, file_format="xlsx")
    detection = _detection_result(
        "plate.xlsx",
        [(sheet, [_table("plate_map", "A1:D3", sheet_name="Plate")])],
    )

    result = await WorkbookSchematizer(Config(detection_strategy="heuristic_only")).schematize(
        file_path="plate.xlsx",
        file_info=detection.file_info,
        file_data=file_data,
        detection_result=detection,
    )

    assert result.entities[0].extraction_orientation == ExtractionOrientation.MATRIX
    assert result.sheet_results[0].regions[0].role == RegionRole.MATRIX_BLOCK
    assert any("matrix-like" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_schematize_infers_lookup_relationships() -> None:
    """Shared key fields should produce lookup/reference relationships."""
    customers = _sheet_from_rows(
        "Customers",
        [
            ["Customer ID", "Customer Name"],
            ["C-001", "Alice"],
            ["C-002", "Bob"],
        ],
    )
    orders = _sheet_from_rows(
        "Orders",
        [
            ["Order ID", "Customer ID", "Amount"],
            ["O-100", "C-001", 25],
            ["O-101", "C-002", 30],
            ["O-102", "C-001", 45],
        ],
    )
    file_data = FileData(sheets=[customers, orders], metadata={}, file_format="xlsx")
    detection = _detection_result(
        "orders.xlsx",
        [
            (customers, [_table("customers", "A1:B3", sheet_name="Customers")]),
            (orders, [_table("orders", "A1:C4", sheet_name="Orders")]),
        ],
    )

    result = await WorkbookSchematizer(Config(detection_strategy="heuristic_only")).schematize(
        file_path="orders.xlsx",
        file_info=detection.file_info,
        file_data=file_data,
        detection_result=detection,
    )

    assert len(result.relationships) == 1
    relationship = result.relationships[0]
    assert relationship.relationship_type == RelationshipType.LOOKUP_REFERENCE
    assert relationship.from_entity_id == "orders"
    assert relationship.to_entity_id == "customers"
    assert relationship.shared_field_ids == ["customer_id"]


@pytest.mark.asyncio
async def test_schematize_emits_low_confidence_warning() -> None:
    """Low-confidence tables should surface review warnings instead of overcommitting silently."""
    sheet = _sheet_from_rows(
        "Inventory",
        [
            ["Item", "Count"],
            ["Widget", 2],
            ["Gadget", 3],
        ],
    )
    file_data = FileData(sheets=[sheet], metadata={}, file_format="xlsx")
    detection = _detection_result(
        "inventory.xlsx",
        [(sheet, [_table("inventory", "A1:B3", sheet_name="Inventory", confidence=0.4)])],
    )

    result = await WorkbookSchematizer(Config(detection_strategy="heuristic_only")).schematize(
        file_path="inventory.xlsx",
        file_info=detection.file_info,
        file_data=file_data,
        detection_result=detection,
    )

    assert result.total_entities == 1
    assert any("Low-confidence schema inference" in warning for warning in result.warnings)
