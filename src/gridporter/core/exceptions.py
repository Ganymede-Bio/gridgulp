"""Custom exceptions for GridPorter."""


class GridPorterError(Exception):
    """Base exception for all GridPorter errors."""

    pass


class FileTypeError(GridPorterError):
    """Raised when file type is not supported or cannot be determined."""

    pass


class DetectionError(GridPorterError):
    """Raised when table detection fails."""

    pass


class ConfigurationError(GridPorterError):
    """Raised when configuration is invalid."""

    pass


class CostLimitError(GridPorterError):
    """Raised when cost limits are exceeded."""

    pass


class ValidationError(GridPorterError):
    """Raised when input validation fails."""

    pass


class TimeoutError(GridPorterError):
    """Raised when processing times out."""

    pass
