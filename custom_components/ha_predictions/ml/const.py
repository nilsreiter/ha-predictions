"""Constants for machine learning components."""

from enum import Enum


class SamplingStrategy(Enum):
    """Enumeration of sampling strategies."""

    NONE = "none"
    RANDOM_OVER = "random_oversample"
    SMOTE = "smote"
