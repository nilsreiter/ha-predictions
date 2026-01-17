"""Unit tests for sampling strategies."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add the ml directory to the path to avoid importing homeassistant dependencies
ml_path = (
    Path(__file__).parent.parent.parent / "custom_components" / "ha_predictions" / "ml"
)
sys.path.insert(0, str(ml_path))

from sampling import random_oversample, smote  # noqa: E402


class TestRandomOversample:
    """Test random oversampling function."""

    def test_basic_functionality(self) -> None:
        """Test basic oversampling of minority class."""
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 0, 1, 1])

        x_resampled, y_resampled = random_oversample(x, y)

        # Should have same number of samples for each class
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == counts[1] == 3

        # Total samples should be 6 (3 + 3)
        assert len(y_resampled) == 6
        assert x_resampled.shape[0] == 6

    def test_target_class_parameter(self) -> None:
        """Test oversampling only a specific target class."""
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 0, 1, 1])

        x_resampled, y_resampled = random_oversample(x, y, target_class=1)

        # Class 1 should be oversampled to match class 0
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert counts[0] == 3  # Class 0 unchanged
        assert counts[1] == 3  # Class 1 oversampled

    def test_already_balanced(self) -> None:
        """Test with already balanced dataset."""
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])

        x_resampled, y_resampled = random_oversample(x, y)

        # Should remain the same size
        assert len(y_resampled) == 4
        assert x_resampled.shape[0] == 4

        # Class distribution should remain balanced
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert counts[0] == counts[1] == 2

    def test_output_shapes(self) -> None:
        """Test that output shapes are correct."""
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 0, 0, 1])

        x_resampled, y_resampled = random_oversample(x, y)

        # Features should maintain same number of columns
        assert x_resampled.shape[1] == x.shape[1]
        # Number of samples should match between X and y
        assert x_resampled.shape[0] == y_resampled.shape[0]

    def test_small_dataset(self) -> None:
        """Test with very small dataset."""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        x_resampled, y_resampled = random_oversample(x, y)

        # Should remain balanced
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == counts[1] == 1

    def test_single_sample_minority(self) -> None:
        """Test with single sample in minority class."""
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 0, 1])

        x_resampled, y_resampled = random_oversample(x, y)

        # Minority class should be oversampled
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert counts[0] == 3
        assert counts[1] == 3

    def test_multiple_classes(self) -> None:
        """Test with more than two classes."""
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([0, 0, 0, 1, 1, 2])

        x_resampled, y_resampled = random_oversample(x, y)

        # All classes should be balanced to majority class count
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 3
        assert all(count == 3 for count in counts)

    def test_shuffling(self) -> None:
        """Test that output is shuffled."""
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])

        x_resampled, y_resampled = random_oversample(x, y)

        # With fixed seed, output should be consistent but shuffled
        # We can't test exact order, but we can verify samples exist
        assert len(y_resampled) == 4
        assert x_resampled.shape[0] == 4

    def test_single_class(self) -> None:
        """Test with dataset containing only one class."""
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 0, 0])

        x_resampled, y_resampled = random_oversample(x, y)

        # Should remain unchanged
        assert len(y_resampled) == 3
        assert x_resampled.shape[0] == 3
        assert np.all(y_resampled == 0)

    def test_preserves_feature_values(self) -> None:
        """Test that oversampling preserves original feature values."""
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 0, 1])

        x_resampled, y_resampled = random_oversample(x, y)

        # All resampled features should exist in original data
        for sample in x_resampled:
            # Check if sample exists in original data
            found = False
            for original in x:
                if np.allclose(sample, original):
                    found = True
                    break
            assert found, f"Sample {sample} not found in original data"


class TestSMOTE:
    """Test SMOTE (Synthetic Minority Over-sampling Technique)."""

    def test_basic_functionality(self) -> None:
        """Test basic SMOTE oversampling."""
        x = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12]])
        y = np.array([0, 0, 0, 1, 1])

        x_resampled, y_resampled = smote(x, y)

        # Should have same number of samples for each class
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == counts[1] == 3

        # Total samples should be 6 (3 + 3)
        assert len(y_resampled) == 6
        assert x_resampled.shape[0] == 6

    def test_target_class_parameter(self) -> None:
        """Test SMOTE with target class parameter."""
        x = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12]])
        y = np.array([0, 0, 0, 1, 1])

        x_resampled, y_resampled = smote(x, y, target_class=1)

        # Class 1 should be oversampled to match class 0
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert counts[0] == 3  # Class 0 unchanged
        assert counts[1] == 3  # Class 1 oversampled

    def test_k_neighbors_parameter(self) -> None:
        """
        Test SMOTE with different k_neighbors values.

        Note: SMOTE requires at least (k_neighbors + 1) samples in the minority
        class to work, since it needs k_neighbors after excluding the sample itself.
        """
        # Need at least k_neighbors+1 samples in minority class for SMOTE to work
        x = np.array(
            [
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [10, 11],
                [11, 12],
                [12, 13],
                [13, 14],
            ]
        )
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

        # Should work with k_neighbors=3 (minority class has 4 samples)
        x_resampled, y_resampled = smote(x, y, k_neighbors=3)

        unique, counts = np.unique(y_resampled, return_counts=True)
        assert counts[0] == 5
        assert counts[1] == 5

    def test_k_neighbors_exceeds_minority_size(self) -> None:
        """
        Test SMOTE when k_neighbors is close to minority class size.

        Note: SMOTE requires at least (k_neighbors + 1) samples in the minority
        class to work, since it needs k_neighbors after excluding the sample itself.
        """
        # Minority class has 3 samples, k_neighbors=2 should work (needs 2+1=3)
        x = np.array(
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [10, 11], [11, 12], [12, 13]]
        )
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1])

        # k_neighbors=2, minority has 3 samples - should work
        x_resampled, y_resampled = smote(x, y, k_neighbors=2)

        # Should oversample minority class
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert counts[0] == 5
        assert counts[1] == 5

    def test_synthetic_samples_are_interpolations(self) -> None:
        """Test that SMOTE creates synthetic samples via interpolation."""
        # Need at least 2 samples in minority class for SMOTE to work
        x = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11]])
        y = np.array([0, 0, 0, 1, 1])

        x_resampled, y_resampled = smote(x, y, k_neighbors=1)

        # Get synthetic samples (class 1 samples)
        class_1_mask = y_resampled == 1
        class_1_samples = x_resampled[class_1_mask]

        # At least one sample should be [10, 10] or [11, 11] (original)
        assert np.any(np.all(class_1_samples == [10, 10], axis=1)) or np.any(
            np.all(class_1_samples == [11, 11], axis=1)
        )

        # Check that synthetic samples are reasonable
        # Should have 3 samples total (2 original + 1 synthetic)
        assert class_1_samples.shape[0] == 3

    def test_small_dataset(self) -> None:
        """Test SMOTE with very small dataset."""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        x_resampled, y_resampled = smote(x, y, k_neighbors=1)

        # Should remain balanced
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == counts[1] == 1

    def test_single_sample_minority(self) -> None:
        """
        Test SMOTE with single sample in minority class.

        Note: SMOTE requires at least (k_neighbors + 1) samples in the minority
        class to work properly. With the default k_neighbors=5, this means at least
        6 samples. With a single sample, there are no neighbors available, so SMOTE
        will fail with a ValueError.
        """
        x = np.array([[1, 2], [2, 3], [3, 4], [10, 11]])
        y = np.array([0, 0, 0, 1])

        # This should raise an error with current implementation
        with pytest.raises(
            ValueError, match="Cannot apply SMOTE to class 1 with only 1 sample"
        ):
            x_resampled, y_resampled = smote(x, y, k_neighbors=1)

    def test_output_shapes(self) -> None:
        """Test that SMOTE output shapes are correct.

        Note: Need at least 2 samples in minority class (k_neighbors=5 default
        requires 6, but we use the minimum viable case).
        """
        # Need at least 2 samples in minority class
        x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [10, 11, 12], [11, 12, 13]])
        y = np.array([0, 0, 0, 1, 1])

        x_resampled, y_resampled = smote(x, y)

        # Features should maintain same number of columns
        assert x_resampled.shape[1] == x.shape[1]
        # Number of samples should match between X and y
        assert x_resampled.shape[0] == y_resampled.shape[0]

    def test_already_balanced(self) -> None:
        """Test SMOTE with already balanced dataset."""
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])

        x_resampled, y_resampled = smote(x, y)

        # Should remain the same size
        assert len(y_resampled) == 4
        assert x_resampled.shape[0] == 4

        # Class distribution should remain balanced
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert counts[0] == counts[1] == 2

    def test_multiple_classes(self) -> None:
        """Test SMOTE with more than two classes.

        Note: Each minority class needs at least 2 samples for SMOTE to work.
        """
        # Each minority class needs at least 2 samples
        x = np.array(
            [
                [1, 2],
                [2, 3],
                [3, 4],  # Class 0 (3 samples)
                [10, 11],
                [11, 12],  # Class 1 (2 samples)
                [20, 21],
                [21, 22],  # Class 2 (2 samples)
            ]
        )
        y = np.array([0, 0, 0, 1, 1, 2, 2])

        x_resampled, y_resampled = smote(x, y)

        # All classes should be balanced to majority class count
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 3
        assert all(count == 3 for count in counts)

    def test_shuffling(self) -> None:
        """Test that SMOTE output is shuffled."""
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])

        x_resampled, y_resampled = smote(x, y)

        # With fixed seed, output should be consistent but shuffled
        assert len(y_resampled) == 4
        assert x_resampled.shape[0] == 4

    def test_single_class(self) -> None:
        """Test SMOTE with dataset containing only one class."""
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 0, 0])

        x_resampled, y_resampled = smote(x, y)

        # Should remain unchanged
        assert len(y_resampled) == 3
        assert x_resampled.shape[0] == 3
        assert np.all(y_resampled == 0)

    def test_preserves_original_samples(self) -> None:
        """Test that SMOTE preserves all original samples.

        Note: Need at least 2 samples in minority class for SMOTE to work.
        """
        # Need at least 2 samples in minority class
        x = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12]])
        y = np.array([0, 0, 0, 1, 1])

        x_resampled, y_resampled = smote(x, y)

        # All original samples should be in resampled data
        for i, sample in enumerate(x):
            found = False
            for j, resampled_sample in enumerate(x_resampled):
                if np.allclose(sample, resampled_sample) and y[i] == y_resampled[j]:
                    found = True
                    break
            assert found, f"Original sample {sample} not found in resampled data"

    def test_synthetic_samples_reasonable_range(self) -> None:
        """Test that synthetic samples fall within reasonable range.

        Note: Need at least 2 samples in minority class for SMOTE to work.
        """
        # Need at least 2 samples in minority class
        x = np.array([[1, 1], [2, 2], [3, 3], [100, 100], [101, 101]])
        y = np.array([0, 0, 0, 1, 1])

        x_resampled, y_resampled = smote(x, y, k_neighbors=1)

        # Get synthetic samples for class 1
        class_1_samples = x_resampled[y_resampled == 1]

        # All class 1 samples should have reasonable values
        # They should be interpolations between [100, 100] and [101, 101]
        for sample in class_1_samples:
            # Each dimension should be between 100 and 101
            assert 100 <= sample[0] <= 101
            assert 100 <= sample[1] <= 101
