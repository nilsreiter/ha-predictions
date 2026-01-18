"""Tests for accuracy calculation."""

import sys
from pathlib import Path

import numpy as np

# Add the ha_predictions directory to path to enable ml package imports
# This avoids importing the parent package which has homeassistant dependencies
ha_predictions_path = (
    Path(__file__).parent.parent.parent / "custom_components" / "ha_predictions"
)
sys.path.insert(0, str(ha_predictions_path))

from ml import accuracy  # noqa: E402


class TestAccuracy:
    """Test score method."""

    def test_accuracy_perfect(self) -> None:
        """Test accuracy with perfect predictions."""
        labels = np.array([0, 0, 0, 1])

        # Calculate expected accuracy
        score = accuracy(labels, labels)
        assert 0 <= score <= 1
        # For this case, should achieve perfect accuracy
        assert score == 1.0

    def test_score_calculation(self) -> None:
        """Test that score calculates accuracy correctly."""
        # Manually construct y so we know what score to expect
        y_gold = np.array([0, 0, 1, 1])

        score = accuracy(np.array([0, 0, 0, 0]), y_gold)
        # The score should be between 0 and 1
        assert 0 <= score <= 1
        assert score == 0.5

        score = accuracy(np.array([0, 0, 1, 0]), y_gold)
        # The score should be between 0 and 1
        assert 0 <= score <= 1
        assert score == 0.75

        score = accuracy(np.array([1, 1, 0, 1]), y_gold)
        # The score should be between 0 and 1
        assert 0 <= score <= 1
        assert score == 0.25
