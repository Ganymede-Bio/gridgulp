"""Context-aware logging utilities for GridGulp."""

import contextvars
import logging
from collections.abc import MutableMapping
from typing import Any, cast

# Context variables for tracking current processing context
current_file = contextvars.ContextVar[str | None]("current_file", default=None)
current_sheet = contextvars.ContextVar[str | None]("current_sheet", default=None)
current_table = contextvars.ContextVar[str | None]("current_table", default=None)
current_operation = contextvars.ContextVar[str | None]("current_operation", default=None)


class ContextualLogger(logging.LoggerAdapter):
    """Logger adapter that automatically includes context information."""

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        """Add context information to log records."""
        # Get current context values
        file_path = current_file.get()
        sheet_name = current_sheet.get()
        table_range = current_table.get()
        operation = current_operation.get()

        # Build extra context
        extra = cast(dict[str, Any], kwargs.get("extra", {}))
        if file_path:
            extra["file"] = file_path
        if sheet_name:
            extra["sheet"] = sheet_name
        if table_range:
            extra["table"] = table_range
        if operation:
            extra["operation"] = operation

        kwargs["extra"] = extra

        # Add context to message if not using structured logging
        context_parts = []
        if file_path:
            context_parts.append(f"file={file_path}")
        if sheet_name:
            context_parts.append(f"sheet={sheet_name}")
        if table_range:
            context_parts.append(f"table={table_range}")
        if operation:
            context_parts.append(f"op={operation}")

        if context_parts:
            context_str = f"[{', '.join(context_parts)}] "
            msg = context_str + msg

        return msg, kwargs


def get_contextual_logger(name: str) -> ContextualLogger:
    """Get a logger that automatically includes context information.

    Args:
        name: Logger name (usually __name__)

    Returns:
        ContextualLogger instance
    """
    base_logger = logging.getLogger(name)
    return ContextualLogger(base_logger, {})


class FileContext:
    """Context manager for tracking current file being processed."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> "FileContext":
        self.token = current_file.set(self.file_path)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.token:
            current_file.reset(self.token)


class SheetContext:
    """Context manager for tracking current sheet being processed."""

    def __init__(self, sheet_name: str):
        self.sheet_name = sheet_name
        self.token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> "SheetContext":
        self.token = current_sheet.set(self.sheet_name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.token:
            current_sheet.reset(self.token)


class TableContext:
    """Context manager for tracking current table being processed."""

    def __init__(self, table_range: str):
        self.table_range = table_range
        self.token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> "TableContext":
        self.token = current_table.set(self.table_range)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.token:
            current_table.reset(self.token)


class OperationContext:
    """Context manager for tracking current operation."""

    def __init__(self, operation: str):
        self.operation = operation
        self.token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> "OperationContext":
        self.token = current_operation.set(self.operation)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.token:
            current_operation.reset(self.token)


def setup_contextual_logging() -> None:
    """Set up contextual logging with structured format.

    This should be called once at application startup.
    """
    # Configure structured logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s - "
        "%(file)s %(sheet)s %(table)s %(operation)s",
        defaults={"file": "", "sheet": "", "table": "", "operation": ""},
    )

    # Apply to root logger handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
