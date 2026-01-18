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

from custom_components.ha_predictions.ml.const import MACRO_AVERAGE
from ml.const import F_SCORE, PRECISION, RECALL  # noqa: E402
from ml.evaluation import accuracy, precision_recall_fscore  # noqa: E402


class TestAccuracy:
    """Test accuracy function."""

    def test_accuracy_perfect(self) -> None:
        """Test accuracy with perfect predictions."""
        labels = np.array([0, 0, 0, 1])

        # Calculate expected accuracy
        score = accuracy(labels, labels)
        assert 0 <= score <= 1
        # For this case, should achieve perfect accuracy
        assert score == 1.0

    def test_accuracy_calculation(self) -> None:
        """Test that accuracy is calculated correctly."""
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


class TestPrecisionRecallFscore:
    """Test precision_recall_fscore method."""

    def test_perfect_predictions(self) -> None:
        """Test PRF with perfect predictions."""
        y_gold = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        scores = precision_recall_fscore(y_pred, y_gold)
        for metric in [PRECISION, RECALL, F_SCORE]:
            for cls in [0, 1]:
                assert scores[metric][cls] == 1.0

    def test_imperfect_predictions(self) -> None:
        """Test PRF with imperfect predictions."""
        y_gold = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])

        scores = precision_recall_fscore(y_pred, y_gold)

        # Class 0
        assert scores[PRECISION][0] == 0.5  # 1 TP / (1 TP + 1 FP)
        assert scores[RECALL][0] == 0.5  # 1 TP / (1 TP + 1 FN)
        assert np.isclose(scores[F_SCORE][0], 0.5)  # F1 score

        # Class 1
        assert scores[PRECISION][1] == 0.5  # 1 TP / (1 TP + 1 FP)
        assert scores[RECALL][1] == 0.5  # 1 TP / (1 TP + 1 FN)
        assert np.isclose(scores[F_SCORE][1], 0.5)  # F1 score

    def test_bad_predictions(self) -> None:
        """Test PRF with imperfect predictions."""
        y_gold = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])

        scores = precision_recall_fscore(y_pred, y_gold)

        # Class 0
        assert scores[PRECISION][0] == 0.5  # 2 TP / (2 TP + 2 FP)
        assert scores[RECALL][0] == 1  # 2 TP / (2 TP + 0 FN)
        assert np.isclose(scores[F_SCORE][0], 2 / 3)  # F1 score

        # Class 1
        assert scores[PRECISION][1] == 0  # 0 TP / (0 TP + 0 FP) -> defined as 0
        assert scores[RECALL][1] == 0  # 0 TP / (0 TP + 2 FN)
        assert np.isclose(scores[F_SCORE][1], 0)  # F1 score

    def test_macro_average_simple_example(self) -> None:
        """Test macro average calculation with simple example."""
        y_gold = np.array([0, 0, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 1, 1, 1, 1])

        scores = precision_recall_fscore(y_pred, y_gold)

        assert np.isclose(scores[PRECISION][MACRO_AVERAGE], 2 / 3)
        assert np.isclose(scores[RECALL][MACRO_AVERAGE], 2 / 3)
        assert np.isclose(scores[F_SCORE][MACRO_AVERAGE], 2 / 3)

    def test_macro_average_skewed_example(self) -> None:
        """Test macro average calculation with skewed example."""
        y_gold = np.array([0, 0, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 1, 1, 1, 1])

        scores = precision_recall_fscore(y_pred, y_gold)
        assert np.isclose(scores[PRECISION][MACRO_AVERAGE], 0.92857143)
        assert scores[RECALL][MACRO_AVERAGE] == 0.75
        assert np.isclose(scores[F_SCORE][MACRO_AVERAGE], 0.79487179)

    def test_assign_class_labels(self) -> None:
        """Test that class labels are assigned correctly."""
        y_gold = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        class_labels = ["cat", "dog"]

        scores = precision_recall_fscore(
            y_pred,
            y_gold,
            class_labels=class_labels,
        )

        # Class 'cat'
        assert scores[PRECISION]["cat"] == 0.5
        assert scores[RECALL]["cat"] == 0.5
        assert np.isclose(scores[F_SCORE]["cat"], 0.5)

        # Class 'dog'
        assert scores[PRECISION]["dog"] == 0.5
        assert scores[RECALL]["dog"] == 0.5
        assert np.isclose(scores[F_SCORE]["dog"], 0.5)
