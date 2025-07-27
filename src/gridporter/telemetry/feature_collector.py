"""Feature collection service for table detection telemetry."""

import logging
from typing import Any, Optional

from .feature_models import DetectionFeatures
from .feature_store import FeatureStore

logger = logging.getLogger(__name__)


class FeatureCollector:
    """Singleton service for collecting detection features."""

    _instance: Optional["FeatureCollector"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize feature collector (only runs once)."""
        if not self._initialized:
            self._feature_store: FeatureStore | None = None
            self._enabled: bool = False
            self._db_path: str = "~/.gridporter/features.db"
            FeatureCollector._initialized = True

    def initialize(self, enabled: bool = False, db_path: str = "~/.gridporter/features.db"):
        """Initialize the feature collector with configuration.

        Args:
            enabled: Whether feature collection is enabled
            db_path: Path to SQLite database
        """
        self._enabled = enabled
        self._db_path = db_path

        if self._enabled:
            try:
                self._feature_store = FeatureStore(db_path)
                logger.info("Feature collection enabled")
            except Exception as e:
                logger.error(f"Failed to initialize feature store: {str(e)}")
                self._enabled = False
                self._feature_store = None
        else:
            logger.info("Feature collection disabled")

    @property
    def enabled(self) -> bool:
        """Check if feature collection is enabled."""
        return self._enabled and self._feature_store is not None

    def record_detection(
        self,
        file_path: str,
        file_type: str,
        table_range: str,
        detection_method: str,
        confidence: float,
        success: bool,
        sheet_name: str | None = None,
        processing_time_ms: int | None = None,
        geometric_features: dict[str, Any] | None = None,
        pattern_features: dict[str, Any] | None = None,
        format_features: dict[str, Any] | None = None,
        content_features: dict[str, Any] | None = None,
        hierarchical_features: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> int | None:
        """Record detection features.

        Args:
            file_path: Path to processed file
            file_type: Type of file (xlsx, csv, etc.)
            table_range: Detected table range
            detection_method: Method used for detection
            confidence: Detection confidence score
            success: Whether detection succeeded
            sheet_name: Sheet name for Excel files
            processing_time_ms: Processing time in milliseconds
            geometric_features: Features from geometry analysis
            pattern_features: Features from pattern detection
            format_features: Features from format analysis
            content_features: Features about cell contents
            hierarchical_features: Features from hierarchy detection
            error_message: Error message if failed

        Returns:
            Row ID if recorded, None if disabled or failed
        """
        if not self.enabled:
            return None

        try:
            # Build feature data
            feature_data = {
                "file_path": file_path,
                "file_type": file_type,
                "sheet_name": sheet_name,
                "table_range": table_range,
                "detection_method": detection_method,
                "confidence": confidence,
                "detection_success": success,
                "error_message": error_message,
                "processing_time_ms": processing_time_ms,
            }

            # Merge feature dictionaries
            if geometric_features:
                feature_data.update(geometric_features)
            if pattern_features:
                feature_data.update(pattern_features)
            if format_features:
                feature_data.update(format_features)
            if content_features:
                feature_data.update(content_features)
            if hierarchical_features:
                feature_data.update(hierarchical_features)

            # Create and validate features
            features = DetectionFeatures(**feature_data)

            # Record to database
            row_id = self._feature_store.record_features(features)
            return row_id

        except Exception as e:
            logger.error(f"Failed to record detection features: {str(e)}")
            return None

    def record_from_geometry_metrics(self, metrics, **kwargs) -> int | None:
        """Record features from GeometryMetrics object.

        Args:
            metrics: GeometryMetrics from region verification
            **kwargs: Additional detection parameters

        Returns:
            Row ID if recorded
        """
        geometric_features = {
            "rectangularness": metrics.rectangularness,
            "filledness": metrics.filledness,
            "density": metrics.density,
            "contiguity": metrics.contiguity,
            "edge_quality": metrics.edge_quality,
            "aspect_ratio": metrics.aspect_ratio,
            "size_ratio": metrics.size_ratio,
        }

        return self.record_detection(geometric_features=geometric_features, **kwargs)

    def record_from_table_pattern(self, pattern, **kwargs) -> int | None:
        """Record features from TablePattern object.

        Args:
            pattern: TablePattern from pattern detection
            **kwargs: Additional detection parameters

        Returns:
            Row ID if recorded
        """
        pattern_features = {
            "pattern_type": pattern.pattern_type.value if pattern.pattern_type else None,
            "orientation": pattern.orientation.value if pattern.orientation else None,
            "has_multi_headers": pattern.has_multi_headers,
            "header_row_count": len(pattern.headers) if pattern.headers else None,
            "fill_ratio": pattern.fill_ratio,
        }

        # Calculate header density if headers present
        if pattern.headers and pattern.bounds:
            total_rows = pattern.bounds[1] - pattern.bounds[0] + 1
            pattern_features["header_density"] = len(pattern.headers) / total_rows

        return self.record_detection(
            pattern_features=pattern_features, table_range=pattern.range, **kwargs
        )

    def get_summary_statistics(self) -> dict[str, Any] | None:
        """Get summary statistics of collected features.

        Returns:
            Statistics dictionary or None if disabled
        """
        if not self.enabled:
            return None

        try:
            return self._feature_store.get_summary_statistics()
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return None

    def export_features(self, output_path: str, format: str = "csv", **query_kwargs):
        """Export collected features.

        Args:
            output_path: Path to output file
            format: Export format (currently only 'csv')
            **query_kwargs: Query parameters for filtering
        """
        if not self.enabled:
            logger.warning("Feature collection is disabled")
            return

        if format == "csv":
            self._feature_store.export_to_csv(output_path, **query_kwargs)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def cleanup(self, days: int = 30):
        """Clean up old feature data.

        Args:
            days: Number of days to retain
        """
        if self.enabled:
            self._feature_store.cleanup_old_data(days)

    def close(self):
        """Close feature store connection."""
        if self._feature_store:
            self._feature_store.close()


# Global instance
_feature_collector = FeatureCollector()


def get_feature_collector() -> FeatureCollector:
    """Get the global feature collector instance.

    Returns:
        The singleton FeatureCollector instance
    """
    return _feature_collector
