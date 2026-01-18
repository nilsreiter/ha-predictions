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

from ml.const import F_SCORE, MACRO_AVERAGE, PRECISION, RECALL  # noqa: E402
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
                assert scores[cls][metric] == 1.0

    def test_imperfect_predictions(self) -> None:
        """Test PRF with imperfect predictions."""
        y_gold = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])

        scores = precision_recall_fscore(y_pred, y_gold)

        # Class 0
        assert scores[0][PRECISION] == 0.5  # 1 TP / (1 TP + 1 FP)
        assert scores[0][RECALL] == 0.5  # 1 TP / (1 TP + 1 FN)
        assert np.isclose(scores[0][F_SCORE], 0.5)  # F1 score

        # Class 1
        assert scores[1][PRECISION] == 0.5  # 1 TP / (1 TP + 1 FP)
        assert scores[1][RECALL] == 0.5  # 1 TP / (1 TP + 1 FN)
        assert np.isclose(scores[1][F_SCORE], 0.5)  # F1 score

    def test_bad_predictions(self) -> None:
        """Test PRF with all predictions being class 0."""
        y_gold = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])

        scores = precision_recall_fscore(y_pred, y_gold)

        # Class 0
        assert scores[0][PRECISION] == 0.5  # 2 TP / (2 TP + 2 FP)
        assert scores[0][RECALL] == 1  # 2 TP / (2 TP + 0 FN)
        assert np.isclose(scores[0][F_SCORE], 2 / 3)  # F1 score

        # Class 1
        assert scores[1][PRECISION] == 0  # 0 TP / (0 TP + 0 FP) -> defined as 0
        assert scores[1][RECALL] == 0  # 0 TP / (0 TP + 2 FN)
        assert np.isclose(scores[1][F_SCORE], 0)  # F1 score

    def test_macro_average_simple_example(self) -> None:
        """Test macro average calculation with simple example."""
        y_gold = np.array([0, 0, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 1, 1, 1, 1])

        scores = precision_recall_fscore(y_pred, y_gold)

        assert np.isclose(scores[MACRO_AVERAGE][PRECISION], 2 / 3)
        assert np.isclose(scores[MACRO_AVERAGE][RECALL], 2 / 3)
        assert np.isclose(scores[MACRO_AVERAGE][F_SCORE], 2 / 3)

    def test_macro_average_skewed_example(self) -> None:
        """Test macro average calculation with skewed example."""
        y_gold = np.array([0, 0, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 1, 1, 1, 1])

        scores = precision_recall_fscore(y_pred, y_gold)
        assert np.isclose(scores[MACRO_AVERAGE][PRECISION], 0.92857143)
        assert scores[MACRO_AVERAGE][RECALL] == 0.75
        assert np.isclose(scores[MACRO_AVERAGE][F_SCORE], 0.79487179)

    def test_assign_class_labels(self) -> None:
        """Test that class labels are assigned correctly."""
        y_gold = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        class_labels = np.array(["cat", "dog"])

        scores = precision_recall_fscore(
            y_pred,
            y_gold,
            class_labels=class_labels,
        )

        # Class 'cat'
        assert scores["cat"][PRECISION] == 0.5
        assert scores["cat"][RECALL] == 0.5
        assert np.isclose(scores["cat"][F_SCORE], 0.5)

        # Class 'dog'
        assert scores["dog"][PRECISION] == 0.5
        assert scores["dog"][RECALL] == 0.5
        assert np.isclose(scores["dog"][F_SCORE], 0.5)
