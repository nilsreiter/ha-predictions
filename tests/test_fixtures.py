"""Shared test fixtures and utilities for ha_predictions tests."""

from typing import Any


class MockLogger:
    """Mock logger for testing.

    This mock logger can be used in tests that require a logger object
    but don't need actual logging functionality. It provides no-op
    implementations of common logger methods.
    """

    def __init__(self) -> None:
        """Initialize the mock logger with call tracking."""
        self.debug_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.info_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.warning_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.error_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.exception_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Mock debug method."""
        self.debug_calls.append((msg, args, kwargs))

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Mock info method."""
        self.info_calls.append((msg, args, kwargs))

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Mock warning method."""
        self.warning_calls.append((msg, args, kwargs))

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Mock error method."""
        self.error_calls.append((msg, args, kwargs))

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Mock exception method."""
        self.exception_calls.append((msg, args, kwargs))
