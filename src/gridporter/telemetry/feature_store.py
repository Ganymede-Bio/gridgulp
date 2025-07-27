"""SQLite-backed feature storage for table detection telemetry."""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .feature_models import DetectionFeatures

logger = logging.getLogger(__name__)


class FeatureStore:
    """SQLite-backed storage for detection features."""

    # SQL schema
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS detection_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

        -- Identification
        file_path TEXT NOT NULL,
        file_type TEXT NOT NULL,
        sheet_name TEXT,
        table_range TEXT NOT NULL,
        detection_method TEXT NOT NULL,

        -- Geometric features
        rectangularness REAL,
        filledness REAL,
        density REAL,
        contiguity REAL,
        edge_quality REAL,
        aspect_ratio REAL,
        size_ratio REAL,

        -- Pattern features
        pattern_type TEXT,
        header_density REAL,
        has_multi_headers INTEGER,
        orientation TEXT,
        fill_ratio REAL,

        -- Format features
        header_row_count INTEGER,
        has_bold_headers INTEGER,
        has_totals INTEGER,
        has_subtotals INTEGER,
        section_count INTEGER,
        separator_row_count INTEGER,

        -- Content features
        total_cells INTEGER,
        filled_cells INTEGER,
        numeric_ratio REAL,
        date_columns INTEGER,
        text_columns INTEGER,
        empty_cell_ratio REAL,

        -- Hierarchical features
        max_hierarchy_depth INTEGER,
        has_indentation INTEGER,
        subtotal_count INTEGER,

        -- Results
        confidence REAL NOT NULL,
        detection_success INTEGER NOT NULL,
        error_message TEXT,
        processing_time_ms INTEGER
    );

    CREATE INDEX IF NOT EXISTS idx_detection_timestamp ON detection_features(timestamp);
    CREATE INDEX IF NOT EXISTS idx_detection_confidence ON detection_features(confidence);
    CREATE INDEX IF NOT EXISTS idx_detection_method ON detection_features(detection_method);
    CREATE INDEX IF NOT EXISTS idx_detection_file ON detection_features(file_path);
    CREATE INDEX IF NOT EXISTS idx_detection_success ON detection_features(detection_success);
    """

    def __init__(self, db_path: str = "~/.gridporter/features.db"):
        """Initialize feature store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_database()
        logger.info(f"Feature store initialized at {self.db_path}")

    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(self._SCHEMA)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path), timeout=30.0, check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise
        else:
            self._local.conn.commit()

    def record_features(self, features: DetectionFeatures) -> int:
        """Store feature vector in database.

        Args:
            features: Detection features to store

        Returns:
            Row ID of inserted record
        """
        try:
            data = features.to_db_dict()

            # Build column names and placeholders
            columns = [k for k in data if k != "timestamp"]
            values = [data[k] for k in columns]
            placeholders = ",".join(["?" for _ in columns])
            column_list = ",".join(columns)

            query = f"""
            INSERT INTO detection_features ({column_list})
            VALUES ({placeholders})
            """

            with self._get_connection() as conn:
                cursor = conn.execute(query, values)
                row_id = cursor.lastrowid

            logger.debug(f"Recorded features for {features.file_path} - {features.table_range}")
            return row_id

        except Exception as e:
            logger.error(f"Failed to record features: {str(e)}")
            raise

    def query_features(
        self,
        file_path: str | None = None,
        detection_method: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        success_only: bool = False,
        since: datetime | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[DetectionFeatures]:
        """Query stored features with filters.

        Args:
            file_path: Filter by file path (supports wildcards with %)
            detection_method: Filter by detection method
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            success_only: Only return successful detections
            since: Only return features recorded after this time
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching features
        """
        query = "SELECT * FROM detection_features WHERE 1=1"
        params = []

        if file_path:
            query += " AND file_path LIKE ?"
            params.append(file_path)

        if detection_method:
            query += " AND detection_method = ?"
            params.append(detection_method)

        if min_confidence is not None:
            query += " AND confidence >= ?"
            params.append(min_confidence)

        if max_confidence is not None:
            query += " AND confidence <= ?"
            params.append(max_confidence)

        if success_only:
            query += " AND detection_success = 1"

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [DetectionFeatures.from_db_row(dict(row)) for row in rows]

    def get_summary_statistics(self) -> dict[str, Any]:
        """Get summary statistics of stored features.

        Returns:
            Dictionary with statistics
        """
        with self._get_connection() as conn:
            stats = {}

            # Total records
            stats["total_records"] = conn.execute(
                "SELECT COUNT(*) FROM detection_features"
            ).fetchone()[0]

            # Success rate
            success_count = conn.execute(
                "SELECT COUNT(*) FROM detection_features WHERE detection_success = 1"
            ).fetchone()[0]
            stats["success_rate"] = (
                success_count / stats["total_records"] if stats["total_records"] > 0 else 0
            )

            # Average confidence
            avg_conf = conn.execute("SELECT AVG(confidence) FROM detection_features").fetchone()[0]
            stats["avg_confidence"] = avg_conf or 0.0

            # Detection methods distribution
            method_stats = conn.execute(
                """
                SELECT detection_method, COUNT(*) as count, AVG(confidence) as avg_conf
                FROM detection_features
                GROUP BY detection_method
            """
            ).fetchall()
            stats["by_method"] = {
                row["detection_method"]: {"count": row["count"], "avg_confidence": row["avg_conf"]}
                for row in method_stats
            }

            # Pattern types distribution
            pattern_stats = conn.execute(
                """
                SELECT pattern_type, COUNT(*) as count, AVG(confidence) as avg_conf
                FROM detection_features
                WHERE pattern_type IS NOT NULL
                GROUP BY pattern_type
            """
            ).fetchall()
            stats["by_pattern"] = {
                row["pattern_type"]: {"count": row["count"], "avg_confidence": row["avg_conf"]}
                for row in pattern_stats
            }

            return stats

    def export_to_csv(self, output_path: str, **query_kwargs):
        """Export features to CSV file.

        Args:
            output_path: Path to output CSV file
            **query_kwargs: Same arguments as query_features()
        """
        import csv

        features = self.query_features(**query_kwargs)
        if not features:
            logger.warning("No features to export")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            # Get field names from first feature
            fieldnames = list(features[0].model_dump().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for feature in features:
                writer.writerow(feature.model_dump())

        logger.info(f"Exported {len(features)} features to {output_path}")

    def cleanup_old_data(self, days: int = 30):
        """Remove features older than specified days.

        Args:
            days: Number of days to retain
        """
        cutoff = datetime.now() - timedelta(days=days)

        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM detection_features WHERE timestamp < ?", (cutoff.isoformat(),)
            )
            deleted = cursor.rowcount

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old feature records")

    def close(self):
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
