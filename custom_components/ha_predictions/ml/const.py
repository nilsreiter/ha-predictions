"""Constants for machine learning components."""

from enum import Enum


class SamplingStrategy(Enum):
    """Enumeration of sampling strategies."""

    NONE = "none"
    RANDOM_OVER = "random_oversample"
    SMOTE = "smote"


PRECISION = "precision"
RECALL = "recall"
F_SCORE = "f_score"

MACRO_AVERAGE = "macro_average"

EXCEPTION_SMOTE_NOT_ENOUGH_SAMPLES = (
    "Not enough samples to perform SMOTE. Class %s has only %d samples."
)
