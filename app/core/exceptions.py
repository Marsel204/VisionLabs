"""Stable exception types for user-facing error handling."""


class ApplicationError(Exception):
    """Base class for expected application failures."""


class ServiceError(ApplicationError):
    """Raised when a domain or infrastructure service cannot complete."""
