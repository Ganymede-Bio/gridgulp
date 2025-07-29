"""Custom exceptions for GridGulp."""


class GridGulpError(Exception):
    """Base exception for all GridGulp errors."""

    pass


class FileTypeError(GridGulpError):
    """Raised when file type is not supported or cannot be determined."""

    pass


class DetectionError(GridGulpError):
    """Raised when table detection fails."""

    pass


class ConfigurationError(GridGulpError):
    """Raised when configuration is invalid."""

    pass


class CostLimitError(GridGulpError):
    """Raised when cost limits are exceeded."""

    pass


class ValidationError(GridGulpError):
    """Raised when input validation fails."""

    pass


class TimeoutError(GridGulpError):
    """Raised when processing times out."""

    pass
