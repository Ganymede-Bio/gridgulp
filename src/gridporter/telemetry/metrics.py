"""General metrics collection for GridPorter."""

import logging
import time
from contextlib import contextmanager
from typing import Any

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricExporter,
    PeriodicExportingMetricReader,
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and export metrics using OpenTelemetry."""

    def __init__(
        self,
        service_name: str = "gridporter",
        exporter: MetricExporter | None = None,
        export_interval_millis: int = 60000,  # 1 minute
    ):
        """Initialize metrics collector.

        Args:
            service_name: Name of the service
            exporter: Optional custom metric exporter
            export_interval_millis: Export interval in milliseconds
        """
        # Create metric reader with exporter
        if exporter is None:
            exporter = ConsoleMetricExporter()

        reader = PeriodicExportingMetricReader(
            exporter=exporter, export_interval_millis=export_interval_millis
        )

        # Create meter provider
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)

        # Get meter
        self.meter = metrics.get_meter(service_name)

        # Create common instruments
        self._create_instruments()

    def _create_instruments(self):
        """Create common metric instruments."""
        # Counters
        self.files_processed = self.meter.create_counter(
            name="gridporter.files_processed", description="Number of files processed", unit="files"
        )

        self.tables_detected = self.meter.create_counter(
            name="gridporter.tables_detected",
            description="Number of tables detected",
            unit="tables",
        )

        self.errors = self.meter.create_counter(
            name="gridporter.errors", description="Number of errors encountered", unit="errors"
        )

        # Histograms
        self.file_size = self.meter.create_histogram(
            name="gridporter.file_size", description="Size of processed files", unit="MB"
        )

        self.processing_time = self.meter.create_histogram(
            name="gridporter.processing_time", description="Time to process files", unit="seconds"
        )

        self.detection_confidence = self.meter.create_histogram(
            name="gridporter.detection_confidence",
            description="Confidence scores of detected tables",
            unit="ratio",
        )

        # Gauges (via callbacks)
        self.meter.create_observable_gauge(
            name="gridporter.memory_usage",
            description="Current memory usage",
            unit="MB",
            callbacks=[self._get_memory_usage],
        )

    def _get_memory_usage(self, _options) -> Any:
        """Callback to get current memory usage."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        yield metrics.Observation(value=memory_mb, attributes={})

    def record_file_processed(
        self,
        file_type: str,
        size_mb: float,
        processing_time_seconds: float,
        tables_found: int,
        success: bool = True,
        error_type: str | None = None,
    ):
        """Record metrics for a processed file.

        Args:
            file_type: Type of file (xlsx, csv, etc.)
            size_mb: File size in MB
            processing_time_seconds: Processing time in seconds
            tables_found: Number of tables detected
            success: Whether processing was successful
            error_type: Type of error if not successful
        """
        attributes = {"file_type": file_type}

        # Record file processed
        self.files_processed.add(1, attributes)

        # Record file size
        self.file_size.record(size_mb, attributes)

        # Record processing time
        self.processing_time.record(processing_time_seconds, attributes)

        # Record tables detected
        if tables_found > 0:
            self.tables_detected.add(tables_found, attributes)

        # Record errors
        if not success and error_type:
            error_attributes = {**attributes, "error_type": error_type}
            self.errors.add(1, error_attributes)

    def record_table_detection(
        self,
        detection_method: str,
        confidence: float,
        table_size: tuple[int, int],
        pattern_type: str | None = None,
    ):
        """Record metrics for a detected table.

        Args:
            detection_method: Method used for detection
            confidence: Confidence score (0-1)
            table_size: (rows, columns) tuple
            pattern_type: Type of table pattern detected
        """
        attributes = {
            "detection_method": detection_method,
            "pattern_type": pattern_type or "unknown",
            "size_category": self._categorize_table_size(table_size),
        }

        # Record confidence
        self.detection_confidence.record(confidence, attributes)

    def _categorize_table_size(self, size: tuple[int, int]) -> str:
        """Categorize table size."""
        rows, cols = size
        cells = rows * cols

        if cells < 100:
            return "small"
        elif cells < 1000:
            return "medium"
        elif cells < 10000:
            return "large"
        else:
            return "extra_large"

    @contextmanager
    def measure_time(self, operation: str):
        """Context manager to measure operation time.

        Example:
            with metrics_collector.measure_time("bitmap_generation"):
                # Generate bitmap
                bitmap = generator.generate(sheet_data)
        """
        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time

            # Create a histogram for this specific operation if not exists
            if not hasattr(self, f"time_{operation}"):
                histogram = self.meter.create_histogram(
                    name=f"gridporter.time.{operation}",
                    description=f"Time for {operation} operation",
                    unit="seconds",
                )
                setattr(self, f"time_{operation}", histogram)

            # Record the measurement
            histogram = getattr(self, f"time_{operation}")
            histogram.record(duration)

    def increment(self, metric_name: str, value: int = 1, attributes: dict = None):
        """Increment a counter metric."""
        # Map common metric names to actual counters
        if metric_name == "bitmaps_generated":
            self.tables_detected.add(value, attributes or {})
        elif hasattr(self, metric_name):
            metric = getattr(self, metric_name)
            metric.add(value, attributes or {})

    def record_duration(self, metric_name: str, duration: float, attributes: dict = None):
        """Record a duration/time metric."""
        # Use the processing_time histogram or create a custom one
        self.processing_time.record(duration, attributes or {"operation": metric_name})

    def record_value(self, metric_name: str, value: float, attributes: dict = None):
        """Record a generic value metric."""
        # Map to appropriate histograms
        if "confidence" in metric_name:
            self.detection_confidence.record(value, attributes or {})
        elif "size" in metric_name:
            self.file_size.record(
                value / (1024 * 1024) if value > 1000 else value, attributes or {}
            )
        else:
            # Log it for debugging
            logger.debug(f"Recording {metric_name}: {value}")

    def get_totals(self) -> dict:
        """Get total counts for various metrics."""
        # This is a simplified version - in production you'd query the metric provider
        return {
            "files_processed": 0,
            "tables_detected": 0,
            "errors": 0,
            "calls": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


# Global metrics collector
_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
