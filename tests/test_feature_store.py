"""Tests for feature store and collection."""

import tempfile
import pytest
from pathlib import Path

from gridporter.telemetry.feature_store import FeatureStore
from gridporter.telemetry.feature_models import DetectionFeatures
from gridporter.telemetry.feature_collector import FeatureCollector, get_feature_collector


class TestFeatureStore:
    """Test feature store functionality."""

    def test_feature_store_creation(self):
        """Test creating a feature store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = FeatureStore(str(db_path))

            # Check database was created
            assert db_path.exists()

            # Close the store
            store.close()

    def test_record_features(self):
        """Test recording detection features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = FeatureStore(str(db_path))

            # Create test features
            features = DetectionFeatures(
                file_path="/test/file.xlsx",
                file_type="xlsx",
                sheet_name="Sheet1",
                table_range="A1:D10",
                detection_method="test_method",
                confidence=0.95,
                detection_success=True,
                rectangularness=0.9,
                filledness=0.8,
                total_cells=40,
                filled_cells=32,
            )

            # Record features
            row_id = store.record_features(features)
            assert row_id > 0

            store.close()

    def test_query_features(self):
        """Test querying stored features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = FeatureStore(str(db_path))

            # Record multiple features
            for i in range(5):
                features = DetectionFeatures(
                    file_path=f"/test/file{i}.xlsx",
                    file_type="xlsx",
                    sheet_name=f"Sheet{i}",
                    table_range=f"A1:D{10+i}",
                    detection_method="test_method",
                    confidence=0.8 + i * 0.02,
                    detection_success=True,
                )
                store.record_features(features)

            # Query all features
            results = store.query_features()
            assert len(results) == 5

            # Query with filters
            high_confidence = store.query_features(min_confidence=0.85)
            assert len(high_confidence) < 5

            # Query by file path
            specific_file = store.query_features(file_path="/test/file2.xlsx")
            assert len(specific_file) == 1
            assert specific_file[0].sheet_name == "Sheet2"

            store.close()

    def test_get_summary_statistics(self):
        """Test getting summary statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = FeatureStore(str(db_path))

            # Record features with different methods and patterns
            methods = ["vision", "pattern", "complex"]
            for i in range(9):
                features = DetectionFeatures(
                    file_path=f"/test/file{i}.xlsx",
                    file_type="xlsx",
                    table_range=f"A1:D{10+i}",
                    detection_method=methods[i % 3],
                    confidence=0.7 + (i % 3) * 0.1,
                    detection_success=i % 2 == 0,
                    pattern_type="header_data" if i % 2 == 0 else "matrix",
                )
                store.record_features(features)

            # Get statistics
            stats = store.get_summary_statistics()

            assert stats["total_records"] == 9
            assert 0 <= stats["success_rate"] <= 1
            assert 0 <= stats["avg_confidence"] <= 1
            assert len(stats["by_method"]) == 3
            assert len(stats["by_pattern"]) == 2

            store.close()

    def test_export_to_csv(self):
        """Test exporting features to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = FeatureStore(str(db_path))

            # Record test features
            for i in range(3):
                features = DetectionFeatures(
                    file_path=f"/test/file{i}.xlsx",
                    file_type="xlsx",
                    table_range=f"A1:D{10+i}",
                    detection_method="test_method",
                    confidence=0.8 + i * 0.05,
                    detection_success=True,
                )
                store.record_features(features)

            # Export to CSV
            csv_path = Path(tmpdir) / "export.csv"
            store.export_to_csv(str(csv_path))

            # Check CSV was created
            assert csv_path.exists()

            # Read and verify CSV content
            with open(csv_path) as f:
                content = f.read()
                assert "file_path" in content
                assert "/test/file0.xlsx" in content
                assert "detection_method" in content

            store.close()


class TestFeatureCollector:
    """Test feature collector functionality."""

    def test_singleton_pattern(self):
        """Test that feature collector is a singleton."""
        collector1 = get_feature_collector()
        collector2 = get_feature_collector()
        assert collector1 is collector2

    def test_initialization(self):
        """Test feature collector initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            collector = get_feature_collector()

            # Initialize with custom path
            collector.initialize(enabled=True, db_path=str(db_path))
            assert collector.enabled

            # Check database was created
            assert db_path.exists()

    def test_record_detection(self):
        """Test recording detection through collector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            collector = get_feature_collector()
            collector.initialize(enabled=True, db_path=str(db_path))

            # Record detection with various features
            row_id = collector.record_detection(
                file_path="/test/file.xlsx",
                file_type="xlsx",
                sheet_name="Sheet1",
                table_range="A1:E20",
                detection_method="vision",
                confidence=0.92,
                success=True,
                geometric_features={
                    "rectangularness": 0.95,
                    "filledness": 0.85,
                    "density": 0.88,
                    "contiguity": 0.90,
                    "edge_quality": 0.87,
                    "aspect_ratio": 1.5,
                    "size_ratio": 0.3,
                },
                pattern_features={
                    "pattern_type": "header_data",
                    "orientation": "horizontal",
                    "has_multi_headers": False,
                    "header_row_count": 1,
                    "fill_ratio": 0.85,
                },
                format_features={
                    "header_row_count": 1,
                    "has_bold_headers": True,
                    "has_totals": False,
                    "section_count": 1,
                },
                content_features={
                    "total_cells": 100,
                    "filled_cells": 85,
                    "numeric_ratio": 0.7,
                    "date_columns": 1,
                    "text_columns": 2,
                },
                processing_time_ms=150,
            )

            assert row_id is not None
            assert row_id > 0

            # Verify through statistics
            stats = collector.get_summary_statistics()
            assert stats["total_records"] == 1
            assert stats["avg_confidence"] == 0.92

    def test_disabled_collector(self):
        """Test that disabled collector doesn't record."""
        collector = FeatureCollector()  # New instance
        collector.initialize(enabled=False)

        assert not collector.enabled

        # Try to record - should return None
        row_id = collector.record_detection(
            file_path="/test/file.xlsx",
            file_type="xlsx",
            table_range="A1:B2",
            detection_method="test",
            confidence=0.5,
            success=True,
        )

        assert row_id is None
