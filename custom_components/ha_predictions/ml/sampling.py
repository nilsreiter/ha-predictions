"""Sampling strategies for handling imbalanced datasets."""

from typing import Any

import numpy as np


def random_oversample(
    x: np.ndarray, y: np.ndarray, target_class: Any | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly oversample minority class(es) to match majority class size.

    x: features (n_samples, n_features)
    y: labels (n_samples,)
    target_class: class to oversample (None = all minority classes)
    """
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    rng = np.random.default_rng(1337)

    x_resampled = []
    y_resampled = []

    for cls in unique:
        # Get indices for this class
        cls_indices = np.where(y == cls)[0]
        cls_count = len(cls_indices)

        # How many samples to add
        n_samples_needed = max_count - cls_count

        if n_samples_needed > 0 and (target_class is None or cls == target_class):
            # Random sampling with replacement
            additional_indices = rng.choice(
                cls_indices, size=n_samples_needed, replace=True
            )
            combined_indices = np.concatenate([cls_indices, additional_indices])
        else:
            combined_indices = cls_indices

        x_resampled.append(x[combined_indices])
        y_resampled.append(y[combined_indices])

    x_resampled = np.vstack(x_resampled)
    y_resampled = np.concatenate(y_resampled)

    # Shuffle
    shuffle_idx = rng.permutation(len(y_resampled))
    return x_resampled[shuffle_idx], y_resampled[shuffle_idx]


def smote(
    x: np.ndarray, y: np.ndarray, k_neighbors: int = 5, target_class: Any | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to oversample minority class(es) to match majority class size.

    Args:
        x: features (n_samples, n_features)
        y: labels (n_samples,)
        k_neighbors: number of nearest neighbors to use for synthetic sample generation
        target_class: class to oversample (None = all minority classes)

    Returns:
        Resampled features and labels as (x_resampled, y_resampled)

    """
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    rng = np.random.default_rng(1337)

    x_resampled = [x]
    y_resampled = [y]

    for cls in unique:
        cls_indices = np.where(y == cls)[0]
        cls_count = len(cls_indices)
        n_samples_needed = max_count - cls_count

        if (
            n_samples_needed > 0
            and (target_class is None or cls == target_class)
            and cls_count > k_neighbors
        ):
            x_cls = x[cls_indices]

            # Generate synthetic samples
            synthetic_samples = []

            for _ in range(n_samples_needed):
                # Pick random sample
                idx = rng.integers(0, len(x_cls))
                sample = x_cls[idx]

                # Find k nearest neighbors (excluding itself)
                distances = np.linalg.norm(x_cls - sample, axis=1)
                nearest_indices = np.argsort(distances)[1 : k_neighbors + 1]

                # Pick random neighbor
                neighbor_idx = rng.choice(nearest_indices)
                neighbor = x_cls[neighbor_idx]

                # Generate synthetic sample (interpolate)
                alpha = rng.random()
                synthetic = sample + alpha * (neighbor - sample)
                synthetic_samples.append(synthetic)

            if synthetic_samples:
                synthetic_samples = np.array(synthetic_samples)
                x_resampled.append(synthetic_samples)
                y_resampled.append(np.full(len(synthetic_samples), cls))

    x_resampled = np.vstack(x_resampled)
    y_resampled = np.concatenate(y_resampled)

    # Shuffle
    shuffle_idx = rng.permutation(len(y_resampled))
    return x_resampled[shuffle_idx], y_resampled[shuffle_idx]
