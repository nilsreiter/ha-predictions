"""Evaluation metrics for ML models."""

import numpy as np

from .const import F_SCORE, MACRO_AVERAGE, PRECISION, RECALL


def accuracy(y_pred: np.ndarray, y_gold: np.ndarray) -> float:
    """
    Calculate accuracy of the model.

    Arguments:
        y_pred (np.ndarray): Predicted labels.
        y_gold (np.ndarray): True labels.

    Returns:
        float: Accuracy score.

    Raises:
        ValueError: If ``y_gold`` is empty or if ``y_pred`` and ``y_gold`` have
            different shapes.

    """
    if len(y_gold) == 0:
        raise ValueError("Cannot compute accuracy on empty ground-truth labels (y_gold).")

    if y_pred.shape != y_gold.shape:
        raise ValueError("y_pred and y_gold must have the same shape to compute accuracy.")
    matches = (y_gold == y_pred).sum()
    total = len(y_gold)
    return matches / total


def precision_recall_fscore(
    y_pred: np.ndarray, y_gold: np.ndarray, beta: float = 1.0
) -> dict[str, dict[str, float]]:
    """
    Calculate precision, recall and f-score of the model.

    Arguments:
        y_pred (np.ndarray): Predicted labels.
        y_gold (np.ndarray): True labels.
        beta (float): Beta value for F-score calculation.
            Default is 1.0, higher values represent more emphasis on recall.

    Returns:
        dict: Precision scores per class and macro average. Outer dict keys are
        'precision', 'recall', 'f_score'.

    """
    classes = [0, 1]
    scores: dict[str, dict[str, float]] = {PRECISION: {}, RECALL: {}, F_SCORE: {}}
    for cls in classes:
        true_positives = ((y_pred == cls) & (y_gold == cls)).sum()
        predicted_positives = (y_pred == cls).sum()
        gold_positives = (y_gold == cls).sum()

        # Precision
        if predicted_positives == 0:
            scores[PRECISION][str(cls)] = 0.0
        else:
            scores[PRECISION][str(cls)] = true_positives / predicted_positives

        # Recall
        if gold_positives == 0:
            scores[RECALL][str(cls)] = 0.0
        else:
            scores[RECALL][str(cls)] = true_positives / gold_positives

        # F1 Score
        prec = scores[PRECISION][str(cls)]
        rec = scores[RECALL][str(cls)]
        if prec + rec == 0:
            scores[F_SCORE][str(cls)] = 0.0
        else:
            scores[F_SCORE][str(cls)] = (
                (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)
            )
    scores[PRECISION][MACRO_AVERAGE] = sum(
        scores[PRECISION][str(cls)] for cls in classes
    ) / len(classes)

    scores[RECALL][MACRO_AVERAGE] = sum(
        scores[RECALL][str(cls)] for cls in classes
    ) / len(classes)

    scores[F_SCORE][MACRO_AVERAGE] = sum(
        scores[F_SCORE][str(cls)] for cls in classes
    ) / len(classes)

    return scores
