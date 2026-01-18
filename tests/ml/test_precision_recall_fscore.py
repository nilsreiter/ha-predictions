"""Tests for PRF calculation."""

import sys
from pathlib import Path

import numpy as np

# Add the ha_predictions directory to path to enable ml package imports
# This avoids importing the parent package which has homeassistant dependencies
ha_predictions_path = (
    Path(__file__).parent.parent.parent / "custom_components" / "ha_predictions"
)
sys.path.insert(0, str(ha_predictions_path))

from ml import precision_recall_fscore  # noqa: E402
from ml.const import F_SCORE, PRECISION, RECALL  # noqa: E402


class TestPrecisionRecallFscore:
    """Test precision_recall_fscore method."""

    def test_perfect_predictions(self) -> None:
        """Test PRF with perfect predictions."""
        y_gold = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        scores = precision_recall_fscore(y_pred, y_gold)

        for metric in [PRECISION, RECALL, F_SCORE]:
            for cls in ["0", "1"]:
                assert scores[metric][cls] == 1.0

    def test_imperfect_predictions(self) -> None:
        """Test PRF with imperfect predictions."""
        y_gold = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])

        scores = precision_recall_fscore(y_pred, y_gold)

        # Class 0
        assert scores[PRECISION]["0"] == 0.5  # 1 TP / (1 TP + 1 FP)
        assert scores[RECALL]["0"] == 0.5  # 1 TP / (1 TP + 1 FN)
        assert np.isclose(scores[F_SCORE]["0"], 0.5)  # F1 score

        # Class 1
        assert scores[PRECISION]["1"] == 0.5  # 1 TP / (1 TP + 1 FP)
        assert scores[RECALL]["1"] == 0.5  # 1 TP / (1 TP + 1 FN)
        assert np.isclose(scores[F_SCORE]["1"], 0.5)  # F1 score

    def test_bad_predictions(self) -> None:
        """Test PRF with imperfect predictions."""
        y_gold = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])

        scores = precision_recall_fscore(y_pred, y_gold)

        # Class 0
        assert scores[PRECISION]["0"] == 0.5  # 1 TP / (1 TP + 1 FP)
        assert scores[RECALL]["0"] == 1  # 1 TP / (1 TP + 1 FN)
        assert np.isclose(scores[F_SCORE]["0"], 2 / 3)  # F1 score

        # Class 1
        assert scores[PRECISION]["1"] == 0  # 1 TP / (1 TP + 1 FP)
        assert scores[RECALL]["1"] == 0  # 1 TP / (1 TP + 1 FN)
        assert np.isclose(scores[F_SCORE]["1"], 0)  # F1 score
