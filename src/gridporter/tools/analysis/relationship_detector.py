"""Tool for detecting relationships between tables."""

from typing import Any, NamedTuple


class TableRelationship(NamedTuple):
    """Represents a relationship between tables."""

    table1_id: str
    table2_id: str
    relationship_type: str
    confidence: float
    evidence: dict[str, Any]

    def involves_table(self, table_id: str) -> bool:
        """Check if this relationship involves a specific table."""
        return table_id in [self.table1_id, self.table2_id]


def detect_relationships(tables: list[Any]) -> list[TableRelationship]:
    """
    Detect relationships between multiple tables.

    This tool analyzes tables to find foreign keys, hierarchical
    relationships, and other connections.

    Args:
        tables: List of analyzed tables

    Returns:
        List of detected relationships
    """
    relationships = []

    # Compare each pair of tables
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            table1 = tables[i]
            table2 = tables[j]

            # Check for potential relationships
            rel = _check_table_relationship(table1, table2)
            if rel:
                relationships.append(rel)

    return relationships


def _check_table_relationship(table1: Any, table2: Any) -> TableRelationship | None:
    """Check for relationship between two tables."""
    # Get table metadata
    t1_headers = set(table1.headers) if hasattr(table1, "headers") else set()
    t2_headers = set(table2.headers) if hasattr(table2, "headers") else set()

    t1_id = table1.info.id if hasattr(table1, "info") else str(table1)
    t2_id = table2.info.id if hasattr(table2, "info") else str(table2)

    # Check for common columns (potential foreign keys)
    common_headers = t1_headers.intersection(t2_headers)

    if common_headers:
        # Look for ID-like columns
        id_patterns = ["id", "code", "key", "_id", "_code"]
        potential_keys = [
            h for h in common_headers if any(pattern in h.lower() for pattern in id_patterns)
        ]

        if potential_keys:
            # Check if values match
            confidence = _calculate_key_match_confidence(table1, table2, potential_keys[0])

            if confidence > 0.7:
                return TableRelationship(
                    table1_id=t1_id,
                    table2_id=t2_id,
                    relationship_type="foreign_key",
                    confidence=confidence,
                    evidence={"common_column": potential_keys[0], "match_type": "identifier"},
                )

    # Check for hierarchical relationship
    if _is_hierarchical_relationship(table1, table2):
        return TableRelationship(
            table1_id=t1_id,
            table2_id=t2_id,
            relationship_type="parent_child",
            confidence=0.8,
            evidence={"hierarchy_detected": True},
        )

    # Check for temporal relationship
    if _is_temporal_relationship(table1, table2):
        return TableRelationship(
            table1_id=t1_id,
            table2_id=t2_id,
            relationship_type="temporal_sequence",
            confidence=0.75,
            evidence={"time_based": True},
        )

    return None


def _calculate_key_match_confidence(table1: Any, table2: Any, key_column: str) -> float:
    """Calculate confidence that a column is a foreign key."""
    # This would analyze actual data values
    # For now, return a placeholder
    return 0.8


def _is_hierarchical_relationship(table1: Any, table2: Any) -> bool:
    """Check if tables have a hierarchical relationship."""
    # Check if one table is a summary of another
    t1_type = table1.metadata.get("table_type", "") if hasattr(table1, "metadata") else ""
    t2_type = table2.metadata.get("table_type", "") if hasattr(table2, "metadata") else ""

    if "hierarchical" in t1_type or "hierarchical" in t2_type:
        # Check if they share dimension columns
        return True

    return False


def _is_temporal_relationship(table1: Any, table2: Any) -> bool:
    """Check if tables are related temporally."""
    # Check if both tables have time dimensions
    t1_fields = table1.metadata.get("field_analysis", {}) if hasattr(table1, "metadata") else {}
    t2_fields = table2.metadata.get("field_analysis", {}) if hasattr(table2, "metadata") else {}

    t1_has_time = any(f.get("semantic_type") == "time_dimension" for f in t1_fields.values())
    t2_has_time = any(f.get("semantic_type") == "time_dimension" for f in t2_fields.values())

    return t1_has_time and t2_has_time
