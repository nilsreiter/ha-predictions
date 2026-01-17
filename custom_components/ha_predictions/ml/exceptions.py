"""Custom exceptions for the ML model component."""


class ModelNotTrainedError(Exception):
    """Exception raised when the model is not trained but an operation requires it."""
