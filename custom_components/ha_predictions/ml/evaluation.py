"""Evaluation metrics for ML models."""

from typing import Any

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
        msg = "Cannot compute accuracy on empty ground-truth labels (y_gold)."
        raise ValueError(msg)

    if y_pred.shape != y_gold.shape:
        msg = "y_pred and y_gold must have the same shape to compute accuracy."
        raise ValueError(msg)
    matches = (y_gold == y_pred).sum()
    total = len(y_gold)
    return matches / total


def precision_recall_fscore(
    y_pred: np.ndarray,
    y_gold: np.ndarray,
    class_labels: np.ndarray | None = None,
    beta: float = 1.0,
) -> dict[Any, dict[str, float]]:
    """
    Calculate precision, recall and f-score of the model.

    This function currently assumes **binary classification** with class labels
    ``0`` and ``1``. Both ``y_pred`` and ``y_gold`` are expected to contain only
    these two classes. If ``class_labels`` is provided, it must be indexable by
    these integer class IDs (i.e. it must have at least two elements where
    index ``0`` and ``1`` correspond to the negative and positive class labels,
    respectively).

    Arguments:
        y_pred (np.ndarray): Predicted labels.
        y_gold (np.ndarray): True labels.
        class_labels (np.ndarray | None): Optional array of human-readable class
            labels corresponding to the integer class IDs used in ``y_pred`` and
            ``y_gold`` (currently assumed to be ``0`` and ``1``). If provided,
            these labels will be used as keys in the returned dictionaries.
        beta (float): Beta value for F-score calculation.
            Default is 1.0, higher values represent more emphasis on recall.

    Returns:
        dict: Nested dictionary of scores per class and macro average. The outer
        dict keys are the class labels (or ``MACRO_AVERAGE``), and the inner dict
        keys are ``'precision'``, ``'recall'``, and ``'f_score'``.


    """
    classes = [0, 1]
    scores: dict[str, dict[Any, float]] = {PRECISION: {}, RECALL: {}, F_SCORE: {}}
    for cls in classes:
        cls_label = class_labels[cls] if class_labels is not None else cls

        true_positives = ((y_pred == cls) & (y_gold == cls)).sum()
        predicted_positives = (y_pred == cls).sum()
        gold_positives = (y_gold == cls).sum()

        # Precision
        if predicted_positives == 0:
            scores[PRECISION][cls_label] = 0.0
        else:
            scores[PRECISION][cls_label] = true_positives / predicted_positives

        # Recall
        if gold_positives == 0:
            scores[RECALL][cls_label] = 0.0
        else:
            scores[RECALL][cls_label] = true_positives / gold_positives

        # F1 Score
        prec = scores[PRECISION][cls_label]
        rec = scores[RECALL][cls_label]
        if prec + rec == 0:
            scores[F_SCORE][cls_label] = 0.0
        else:
            scores[F_SCORE][cls_label] = (
                (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)
            )

    scores[PRECISION][MACRO_AVERAGE] = sum(scores[PRECISION].values()) / len(classes)
    scores[RECALL][MACRO_AVERAGE] = sum(scores[RECALL].values()) / len(classes)
    scores[F_SCORE][MACRO_AVERAGE] = sum(scores[F_SCORE].values()) / len(classes)

    all_inner_keys = {k for d in scores.values() for k in d}

    # transpose, such that the class is the outer key
    return {
        inner_key: {
            outer_key: scores[outer_key][inner_key]
            for outer_key in scores
            if inner_key in scores[outer_key]
        }
        for inner_key in all_inner_keys
    }
