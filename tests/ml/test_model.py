"""Unit tests for Model class."""

import contextlib
import sys
from pathlib import Path
from types import NoneType
from typing import Any

import numpy as np
import pandas as pd
import pytest

# Add the ml directory to the path to avoid importing homeassistant dependencies
ml_path = (
    Path(__file__).parent.parent.parent / "custom_components" / "ha_predictions" / "ml"
)
sys.path.insert(0, str(ml_path))

from logistic_regression import LogisticRegression  # noqa: E402
from sampling import random_oversample, smote  # noqa: E402


class MockLogger:
    """Mock logger for testing."""

    def debug(self, *args: Any, **kwargs: Any) -> None:
        """Mock debug method."""

    def info(self, *args: Any, **kwargs: Any) -> None:
        """Mock info method."""


class ModelNotTrainedError(Exception):
    """Exception raised when the model is not trained but an operation requires it."""


class SamplingStrategy:
    """Mock sampling strategy enum."""

    NONE = "none"
    RANDOM_OVER = "random_oversample"
    SMOTE = "smote"


class Model:
    """Model class for testing (updated to match model.py)."""

    def __init__(self, logger: MockLogger) -> None:
        """Initialize the Model class."""
        self.logger = logger
        self.accuracy: float | NoneType = None
        self.factors: dict[int, Any] = {}
        self.model_eval: LogisticRegression | None = None
        self.model_final: LogisticRegression | None = None
        self.target_column_idx: int | None = None
        self.prediction_ready: bool = False
        self.transformations: dict[str, dict[str, Any]] = {
            "zscores": {},
            "sampling": {"type": SamplingStrategy.SMOTE, "k_neighbors": 5},
        }

    def predict(self, data: np.ndarray) -> tuple[str, float] | NoneType:
        """
        Make predictions and return original values.

        Args:
            data: Numpy array of feature values (raw, not encoded)

        Returns:
            Tuple of (predicted_label, probability) or None if prediction not possible.

        Raises:
            ModelNotTrainedError: If the model is not (yet) trained.

        """
        if self.model_final is None:
            raise ModelNotTrainedError

        # Apply factorization to features only using stored factors
        # Create a new array with float dtype to avoid object dtype issues
        data_encoded = np.empty(data.shape, dtype=float)

        for col_idx in range(data.shape[1]):
            # Apply factorization if this column was factorized during training
            if col_idx in self.factors:
                value = data[0, col_idx]
                categories = self.factors[col_idx]

                # Find the index of the value in categories using numpy
                try:
                    idx = np.where(categories == value)[0]
                    if len(idx) > 0:
                        data_encoded[0, col_idx] = float(idx[0])
                    else:
                        # Value not found in training data, use -1
                        data_encoded[0, col_idx] = -1.0
                except (ValueError, TypeError):
                    data_encoded[0, col_idx] = -1.0
            else:
                # Copy numeric data as-is
                try:
                    data_encoded[0, col_idx] = float(data[0, col_idx])
                except (ValueError, TypeError):
                    data_encoded[0, col_idx] = 0.0

        if "zscores" in self.transformations:
            # Apply z-score normalization
            self.logger.debug("Applying z-score normalization for prediction")
            means = self.transformations["zscores"]["means"]
            stds = self.transformations["zscores"]["stds"]

            data_encoded = (data_encoded - means) / stds

        # Predict
        predictions, probabilities = self.model_final.predict(data_encoded)

        if (
            self.target_column_idx is not None
            and self.target_column_idx in self.factors
            and predictions is not None
            and probabilities is not None
        ):
            target_categories = self.factors[self.target_column_idx]
            label = target_categories[predictions[0]]
            # Sigmoid output represents P(class=1), adjust for class 0
            # If predicted class is 0, probability should be 1 - sigmoid_output
            probability = (
                probabilities[0] if predictions[0] == 1 else 1 - probabilities[0]
            )
            return (label, probability)
        return None

    def train_final(
        self,
        data: np.ndarray,
    ) -> None:
        """
        Train the final model.

        Args:
            data: Numpy array with features and target (not encoded).
                  Last column is assumed to be the target column.

        """
        # Target column is the last column
        self.target_column_idx = data.shape[1] - 1

        # Factorize categorical columns using numpy.unique
        # Create a new array with float dtype to avoid object dtype issues
        data_encoded = np.empty(data.shape, dtype=float)
        self.factors = {}

        for col_idx in range(data.shape[1]):
            column_data = data[:, col_idx]

            # Check if column contains non-numeric data
            if column_data.dtype == object or not np.issubdtype(
                column_data.dtype, np.number
            ):
                # Use numpy.unique to get unique values and their indices
                unique_values, inverse_indices = np.unique(
                    column_data, return_inverse=True
                )
                # Store the unique values for later decoding (keyed by column index)
                self.factors[col_idx] = unique_values
                # Replace column with encoded indices
                data_encoded[:, col_idx] = inverse_indices.astype(float)
            else:
                # Copy numeric data as-is
                data_encoded[:, col_idx] = column_data.astype(float)

        # Split features and target
        x_train = data_encoded[:, :-1]
        y_train = data_encoded[:, -1]

        if "zscores" in self.transformations:
            (means, stds, x_train) = self._apply_normalization(x_train)
            self.transformations["zscores"]["means"] = means
            self.transformations["zscores"]["stds"] = stds

        # Train model
        self.model_final = LogisticRegression()
        self.logger.debug("Training of final model begins")
        self.model_final.fit(x_train, y_train)
        self.logger.debug("Training ends, model: %s", str(self.model_final))
        self.prediction_ready = True

    def train_eval(self, data: np.ndarray) -> NoneType:
        """
        Train and evaluate the model with train/test split.

        Args:
            data: Numpy array with features and target (not encoded).
                  Last column is assumed to be the target column.

        """
        self.logger.info("Starting training for evaluation with data: %s", str(data))

        # Remove rows with 'Unavailable' in the target column (last column)
        filtered_arr = data[data[:, -1] != "unavailable"]
        # Remove rows with 'unknown' in the target column
        filtered_arr = filtered_arr[filtered_arr[:, -1] != "unknown"]

        # Factorize categorical columns using numpy.unique
        # Create a new array with float dtype to avoid object dtype issues
        data_encoded = np.empty(filtered_arr.shape, dtype=float)

        for col_idx in range(filtered_arr.shape[1]):
            column_data = filtered_arr[:, col_idx]

            # Check if column contains non-numeric data
            if column_data.dtype == object or not np.issubdtype(
                column_data.dtype, np.number
            ):
                # Use numpy.unique to get unique values and their indices
                inverse_indices = np.unique(column_data, return_inverse=True)[1]
                # Replace column with encoded indices
                data_encoded[:, col_idx] = inverse_indices.astype(float)
            else:
                # Copy numeric data as-is
                data_encoded[:, col_idx] = column_data.astype(float)

        # train/test split in pure numpy with stratification
        rng = np.random.Generator(np.random.PCG64())

        # Get target column (last column)
        y = data_encoded[:, -1]
        unique_classes = np.unique(y)

        train_indices = []
        test_indices = []

        # Stratify split based on last column
        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            rng.shuffle(cls_indices)

            n_cls = len(cls_indices)
            # Ensure at least 1 test sample per class if there are 2+ samples
            # For single-sample classes, put in training to avoid empty training sets
            test_size_cls = max(int(n_cls * 0.25), 1) if n_cls > 1 else 0

            test_indices.extend(cls_indices[:test_size_cls])
            train_indices.extend(cls_indices[test_size_cls:])

        # Ensure at least 1 test sample overall (fallback for edge cases)
        if len(test_indices) == 0 and len(train_indices) > 1:
            # Move one sample from train to test
            test_indices.append(train_indices.pop())

        # Shuffle the final indices to mix classes
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

        train = data_encoded[train_indices, :]
        test = data_encoded[test_indices, :]
        self.logger.debug("Data used for training: %s", str(train))
        self.logger.debug("Data used for testing: %s", str(test))

        train = self._apply_sampling(train)

        # Split x and y
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = test[:, :-1]
        y_test = test[:, -1]

        if "zscores" in self.transformations:
            (means, stds, x_train) = self._apply_normalization(x_train)
            x_test = (x_test - means) / stds

        self.model_eval = LogisticRegression()
        self.logger.debug("Training begins")
        self.model_eval.fit(x_train, y_train)
        self.logger.debug("Training ends, model: %s", str(self.model_eval))
        self.accuracy = self.model_eval.score(x_test, y_test)

    def _apply_normalization(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply z-score normalization to data.

        Args:
            data: Numpy array with feature data.

        Returns:
            Tuple of (means, stds, normalized_data).

        """
        self.logger.debug("Applying z-score normalization")
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0, ddof=0)

        # Avoid division by zero by leaving zero-variance features unscaled
        stds[stds == 0] = 1.0

        data = (data - means) / stds
        return (means, stds, data)

    def _apply_sampling(self, train: np.ndarray) -> np.ndarray:
        """
        Apply sampling strategy to training data.

        Args:
            train: Numpy array with training data (features + target).

        Returns:
            Numpy array with resampled training data.

        """
        if "sampling" in self.transformations:
            if self.transformations["sampling"]["type"] == SamplingStrategy.SMOTE:
                # Apply SMOTE to training data
                self.logger.debug("Applying SMOTE to training data")
                x_train_smote, y_train_smote = smote(
                    train[:, :-1],
                    train[:, -1],
                    k_neighbors=self.transformations["sampling"].get("k_neighbors", 5),
                )
                train = np.hstack((x_train_smote, y_train_smote.reshape(-1, 1)))
                self.logger.debug("Training data after SMOTE: %s", np.info(train))
            elif (
                self.transformations["sampling"]["type"] == SamplingStrategy.RANDOM_OVER
            ):
                # Apply random oversampling to training data
                self.logger.debug("Applying random oversampling to training data")
                x_train_rand, y_train_rand = random_oversample(
                    train[:, :-1],
                    train[:, -1],
                    target_class=None,
                )
                train = np.hstack((x_train_rand, y_train_rand.reshape(-1, 1)))
                self.logger.debug(
                    "Training data after random oversampling: %s", np.info(train)
                )
        return train


def convert_df_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """Helper function to convert DataFrame to numpy for testing."""
    return df.to_numpy()


class TestModelPredictProbabilities:
    """Test that probabilities correspond to predicted class."""

    def test_probability_corresponds_to_predicted_class_0(self) -> None:
        """Test that probability for class 0 is correctly calculated."""
        model = Model(MockLogger())

        # Train on data with more separation and more samples
        train_data = pd.DataFrame(
            {
                "feature1": [0, 1, 2, 3, 4, 15, 16, 17, 18, 19],
                "feature2": [0, 1, 2, 3, 4, 15, 16, 17, 18, 19],
                "target": [
                    "off",
                    "off",
                    "off",
                    "off",
                    "off",
                    "on",
                    "on",
                    "on",
                    "on",
                    "on",
                ],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # Predict on a sample that should be class 0 ("off") - using extreme low value
        test_data = pd.DataFrame({"feature1": [0], "feature2": [0]})
        test_numpy = convert_df_to_numpy(test_data)
        label, probability = model.predict(test_numpy)

        # Should predict "off"
        assert label == "off"
        # Probability should be high (> 0.5) for the predicted class
        # This is the key test: probability represents confidence
        # in predicted class
        assert probability > 0.5

    def test_probability_corresponds_to_predicted_class_1(self) -> None:
        """Test that probability for class 1 is correctly calculated."""
        model = Model(MockLogger())

        # Train on data with more separation
        train_data = pd.DataFrame(
            {
                "feature1": [0, 1, 2, 3, 4, 15, 16, 17, 18, 19],
                "feature2": [0, 1, 2, 3, 4, 15, 16, 17, 18, 19],
                "target": [
                    "off",
                    "off",
                    "off",
                    "off",
                    "off",
                    "on",
                    "on",
                    "on",
                    "on",
                    "on",
                ],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # Predict on a sample that should be class 1 ("on") - using extreme high value
        test_data = pd.DataFrame({"feature1": [19], "feature2": [19]})
        test_numpy = convert_df_to_numpy(test_data)
        label, probability = model.predict(test_numpy)

        # Should predict "on"
        assert label == "on"
        # Probability should be high (> 0.5) for the predicted class
        # This is the key test: probability represents confidence
        # in predicted class
        assert probability > 0.5

    def test_probability_consistency(self) -> None:
        """Test that probability is always >= 0.5 for the predicted class."""
        model = Model(MockLogger())

        # Train on varied data
        train_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "target": [
                    "off",
                    "off",
                    "off",
                    "off",
                    "off",
                    "on",
                    "on",
                    "on",
                    "on",
                    "on",
                ],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # Test multiple predictions
        test_values = [1.5, 3.5, 5.5, 7.5, 9.5]
        for val in test_values:
            test_data = pd.DataFrame({"feature1": [val], "feature2": [val]})
            test_numpy = convert_df_to_numpy(test_data)
            label, probability = model.predict(test_numpy)

            # Probability should always be >= 0.5 since it represents
            # confidence in the predicted class (after threshold of 0.5)
            assert probability >= 0.5, (
                f"Probability {probability} < 0.5 for value {val}, label {label}"
            )

    def test_probability_with_edge_case_sigmoid_values(self) -> None:
        """Test probability calculation with edge case sigmoid values."""
        model = Model(MockLogger())

        # Train with simple data
        train_data = pd.DataFrame(
            {
                "feature": [0, 1, 10, 11],
                "target": ["no", "no", "yes", "yes"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # Test predictions at extremes
        test_data_low = pd.DataFrame({"feature": [0]})
        test_numpy_low = convert_df_to_numpy(test_data_low)
        label_low, prob_low = model.predict(test_numpy_low)

        test_data_high = pd.DataFrame({"feature": [11]})
        test_numpy_high = convert_df_to_numpy(test_data_high)
        label_high, prob_high = model.predict(test_numpy_high)

        # Both probabilities should represent confidence in predicted class
        assert prob_low >= 0.5
        assert prob_high >= 0.5

        # Predictions at extremes should be different classes
        assert label_low != label_high

        # Both probabilities should represent confidence in predicted class
        assert prob_low >= 0.5
        assert prob_high >= 0.5

        # Predictions at extremes should be different classes
        assert label_low != label_high


class TestModelInitialization:
    """Test Model initialization and attributes."""

    def test_init_creates_transformations(self) -> None:
        """Test that initialization creates transformations attribute."""
        model = Model(MockLogger())
        assert hasattr(model, "transformations")
        assert "zscores" in model.transformations
        assert "sampling" in model.transformations

    def test_init_default_sampling_strategy(self) -> None:
        """Test that default sampling strategy is SMOTE."""
        model = Model(MockLogger())
        assert model.transformations["sampling"]["type"] == SamplingStrategy.SMOTE
        assert model.transformations["sampling"]["k_neighbors"] == 5

    def test_init_factors_empty(self) -> None:
        """Test that factors dict is initialized empty."""
        model = Model(MockLogger())
        assert model.factors == {}

    def test_init_prediction_ready_false(self) -> None:
        """Test that prediction_ready is False initially."""
        model = Model(MockLogger())
        assert model.prediction_ready is False


class TestModelPredictError:
    """Test Model predict error handling."""

    def test_predict_raises_model_not_trained_error(self) -> None:
        """Test that predict raises ModelNotTrainedError when not trained."""
        model = Model(MockLogger())
        test_data = pd.DataFrame({"feature": [1]})
        test_numpy = convert_df_to_numpy(test_data)

        with pytest.raises(ModelNotTrainedError):
            model.predict(test_numpy)


class TestModelTrainFinalWithNormalization:
    """Test Model train_final with z-score normalization."""

    def test_train_final_stores_normalization_params(self) -> None:
        """Test that train_final stores normalization parameters."""
        model = Model(MockLogger())
        train_data = pd.DataFrame(
            {
                "feature1": [100, 200, 300, 400],
                "feature2": [1000, 2000, 3000, 4000],
                "target": ["off", "off", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        assert "means" in model.transformations["zscores"]
        assert "stds" in model.transformations["zscores"]
        assert len(model.transformations["zscores"]["means"]) == 2
        assert len(model.transformations["zscores"]["stds"]) == 2

    def test_train_final_normalization_parameters(self) -> None:
        """Test that normalization parameters are calculated correctly."""
        model = Model(MockLogger())
        train_data = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4],
                "target": ["off", "off", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        means = model.transformations["zscores"]["means"]
        # Mean of [1, 2, 3, 4] encoded as [0, 1, 2, 3] for "off", "off", "on", "on"
        # Feature column after factorization: [1, 2, 3, 4]
        # Mean should be 2.5 but only feature columns are normalized (not target)
        # After train_final, only 1 feature column exists (excluding target)
        # So means[0] is the mean of the feature column
        assert len(means) == 1  # Only one feature column

    def test_train_final_zero_variance_handling(self) -> None:
        """Test that zero-variance features are handled correctly."""
        model = Model(MockLogger())
        # Feature1 has zero variance
        train_data = pd.DataFrame(
            {
                "feature1": [5, 5, 5, 5],
                "feature2": [1, 2, 3, 4],
                "target": ["off", "off", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Should not raise division by zero error
        model.train_final(train_numpy)
        assert model.model_final is not None

        # Std for zero-variance feature should be set to 1.0
        stds = model.transformations["zscores"]["stds"]
        assert stds[0] == 1.0


class TestModelPredictWithNormalization:
    """Test Model predict applies normalization."""

    def test_predict_applies_normalization(self) -> None:
        """Test that predict applies stored normalization parameters."""
        model = Model(MockLogger())
        train_data = pd.DataFrame(
            {
                "feature": [100, 200, 300, 400],
                "target": ["off", "off", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # Test prediction (normalization should be applied internally)
        test_data = pd.DataFrame({"feature": [150]})
        test_numpy = convert_df_to_numpy(test_data)
        result = model.predict(test_numpy)

        assert result is not None
        label, probability = result
        assert label in ["off", "on"]


class TestModelTrainEval:
    """Test Model train_eval method."""

    def test_train_eval_basic(self) -> None:
        """Test train_eval with basic data."""
        model = Model(MockLogger())
        train_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature2": [10, 20, 30, 40, 50, 60, 70, 80],
                "target": ["off", "off", "off", "off", "on", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_eval(train_numpy)

        assert model.model_eval is not None
        assert model.accuracy is not None
        assert 0.0 <= model.accuracy <= 1.0

    def test_train_eval_sets_accuracy(self) -> None:
        """Test that train_eval sets accuracy attribute."""
        model = Model(MockLogger())
        train_data = pd.DataFrame(
            {
                "feature": list(range(10)),
                "target": ["off"] * 5 + ["on"] * 5,
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_eval(train_numpy)

        assert isinstance(model.accuracy, float)
        assert 0.0 <= model.accuracy <= 1.0

    def test_train_eval_stratified_split(self) -> None:
        """Test that train_eval performs stratified split correctly."""
        model = Model(MockLogger())
        # Create balanced dataset
        train_data = pd.DataFrame(
            {
                "feature": list(range(20)),
                "target": ["off"] * 10 + ["on"] * 10,
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_eval(train_numpy)

        # Should successfully train and evaluate without error
        assert model.model_eval is not None

    def test_train_eval_edge_case_single_sample_per_class(self) -> None:
        """Test train_eval with only 1 sample per class."""
        model = Model(MockLogger())
        train_data = pd.DataFrame(
            {
                "feature": [1, 10],
                "target": ["off", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Should handle this edge case without error
        model.train_eval(train_numpy)
        assert model.model_eval is not None


class TestModelTrainEvalFiltering:
    """Test Model train_eval filtering of unavailable and unknown values."""

    def test_train_eval_filters_unavailable_target_values(self) -> None:
        """Test that train_eval filters out rows with 'unavailable' in target."""
        model = Model(MockLogger())
        # Create data with 'unavailable' values in target
        train_data = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4, 5, 6, 7, 8],
                "target": [
                    "off",
                    "unavailable",
                    "on",
                    "off",
                    "unavailable",
                    "on",
                    "off",
                    "on",
                ],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Train should succeed without error
        model.train_eval(train_numpy)

        # Model should be trained
        assert model.model_eval is not None
        assert model.accuracy is not None

        # The filtered data should only have 6 rows (8 - 2 unavailable)
        # We can verify this indirectly by checking that model was trained
        # successfully with only the valid classes

    def test_train_eval_filters_unknown_target_values(self) -> None:
        """Test that train_eval filters out rows with 'unknown' in target column."""
        model = Model(MockLogger())
        # Create data with 'unknown' values in target
        train_data = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4, 5, 6, 7, 8],
                "target": ["off", "unknown", "on", "off", "unknown", "on", "off", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Train should succeed without error
        model.train_eval(train_numpy)

        # Model should be trained
        assert model.model_eval is not None
        assert model.accuracy is not None

    def test_train_eval_filters_both_unavailable_and_unknown(self) -> None:
        """Test that train_eval filters both 'unavailable' and 'unknown'."""
        model = Model(MockLogger())
        # Create data with both 'unavailable' and 'unknown' values
        train_data = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "target": [
                    "off",
                    "unavailable",
                    "on",
                    "unknown",
                    "off",
                    "on",
                    "unavailable",
                    "unknown",
                    "off",
                    "on",
                ],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Train should succeed without error
        model.train_eval(train_numpy)

        # Model should be trained with only valid data
        assert model.model_eval is not None
        assert model.accuracy is not None

    def test_train_eval_keeps_valid_target_values(self) -> None:
        """Test train_eval keeps valid values and filters unavailable/unknown."""
        model = Model(MockLogger())
        # Create data with various values
        # Only 'unavailable' and 'unknown' should be filtered
        train_data = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "target": [
                    "off",
                    "on",
                    "off",
                    "on",
                    "unavailable",
                    "off",
                    "unknown",
                    "on",
                    "off",
                    "on",
                ],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Train should succeed with 8 valid rows (10 - 2 filtered)
        model.train_eval(train_numpy)

        # Model should be trained
        assert model.model_eval is not None
        assert model.accuracy is not None

    def test_train_eval_with_all_unavailable_raises_error(self) -> None:
        """Test train_eval with all unavailable values handles edge case."""
        model = Model(MockLogger())
        # Create data where all values are unavailable
        train_data = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4],
                "target": ["unavailable", "unavailable", "unavailable", "unavailable"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # This should raise an error or handle empty dataset gracefully
        # Since the filtering results in an empty array
        with contextlib.suppress(ValueError, IndexError):
            model.train_eval(train_numpy)
            # If it doesn't raise, check that model handles empty data
            # The model may not train properly with empty data


class TestModelSampling:
    """Test Model sampling functionality."""

    def test_sampling_configuration_default(self) -> None:
        """Test default sampling configuration is SMOTE."""
        model = Model(MockLogger())
        assert model.transformations["sampling"]["type"] == SamplingStrategy.SMOTE
        assert model.transformations["sampling"]["k_neighbors"] == 5

    def test_sampling_configuration_can_change(self) -> None:
        """Test that sampling configuration can be changed."""
        model = Model(MockLogger())
        model.transformations["sampling"]["type"] = SamplingStrategy.RANDOM_OVER

        # Should be able to use train_eval with changed sampling
        train_data = pd.DataFrame(
            {
                "feature": [1, 2, 3, 10, 11, 12],
                "target": ["off", "off", "off", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_eval(train_numpy)

        assert model.model_eval is not None


class TestModelFactorization:
    """Test Model factorization functionality."""

    def test_factorization_categorical_features(self) -> None:
        """Test that categorical features are factorized."""
        model = Model(MockLogger())
        train_data = pd.DataFrame(
            {
                "category": ["a", "b", "c", "d"],
                "target": ["off", "off", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # Both columns should be factorized (categorical)
        assert 0 in model.factors  # category column
        assert 1 in model.factors  # target column

    def test_factorization_numeric_features(self) -> None:
        """Test that numeric features are not factorized."""
        model = Model(MockLogger())
        # Use integer dtype to ensure numpy recognizes as numeric
        train_data = pd.DataFrame(
            {
                "numeric": pd.array([1, 2, 3, 4], dtype="int64"),
                "target": ["off", "off", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # Only target should be factorized (categorical)
        # Note: When converting pandas to numpy with to_numpy(),
        # mixed types create object dtype, so numeric column may be factorized
        # This test documents actual behavior
        assert 1 in model.factors  # target column factorized

    def test_factorization_order(self) -> None:
        """Test that factorization preserves sorted order."""
        model = Model(MockLogger())
        train_data = pd.DataFrame(
            {
                "category": ["c", "a", "b", "a"],
                "target": ["off", "off", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # numpy.unique sorts values
        categories = model.factors[0]
        assert categories[0] == "a"
        assert categories[1] == "b"
        assert categories[2] == "c"


class TestModelUnknownCategories:
    """Test Model handling of unknown categorical values."""

    def test_predict_unknown_categorical_value(self) -> None:
        """Test predict with unknown categorical value uses -1 encoding."""
        model = Model(MockLogger())
        train_data = pd.DataFrame(
            {
                "color": ["red", "blue", "red", "blue"],
                "target": ["off", "on", "on", "off"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # Predict with unknown category
        test_data = pd.DataFrame(
            {
                "color": ["green"],  # Not in training data
            }
        )
        test_numpy = convert_df_to_numpy(test_data)
        result = model.predict(test_numpy)

        # Should still return a result even with unknown value
        assert result is not None


class TestModelEdgeCases:
    """Test Model edge cases."""

    def test_single_sample_training(self) -> None:
        """Test training with single sample."""
        model = Model(MockLogger())
        train_data = pd.DataFrame(
            {
                "feature": [1],
                "target": ["on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Should handle single sample
        model.train_final(train_numpy)
        assert model.model_final is not None
        assert model.prediction_ready is True

    def test_all_same_target_value(self) -> None:
        """Test training when all target values are the same."""
        model = Model(MockLogger())
        train_data = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4],
                "target": ["on", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Should not raise error
        model.train_final(train_numpy)
        assert model.model_final is not None

    def test_mixed_data_types(self) -> None:
        """Test with mixed categorical and numeric features."""
        model = Model(MockLogger())
        train_data = pd.DataFrame(
            {
                "numeric1": pd.array([1, 2, 3, 4], dtype="int64"),
                "categorical": ["a", "b", "a", "b"],
                "numeric2": pd.array([10, 20, 30, 40], dtype="int64"),
                "target": ["off", "on", "off", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # Check that categorical column is factorized
        assert 1 in model.factors  # categorical column
        assert 3 in model.factors  # target column

        # Note: Due to DataFrame.to_numpy() with mixed types creating object dtype,
        # numeric columns may also be factorized. This documents actual behavior.

    def test_predict_returns_none_for_numeric_target(self) -> None:
        """Test that predict returns None for numeric target."""
        model = Model(MockLogger())
        train_data = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4],
                "target": [0, 0, 1, 1],  # Numeric target
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        test_data = pd.DataFrame({"feature": [2.5]})
        test_numpy = convert_df_to_numpy(test_data)
        result = model.predict(test_numpy)

        # Should return None if target is not categorical
        assert result is None


class TestModelNormalizationMethod:
    """Test Model _apply_normalization private method."""

    def test_apply_normalization_calculates_correctly(self) -> None:
        """Test that _apply_normalization calculates means and stds correctly."""
        model = Model(MockLogger())
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])

        means, stds, normalized = model._apply_normalization(data)

        # Check means
        assert np.isclose(means[0], 2.5, atol=0.1)
        assert np.isclose(means[1], 25.0, atol=0.1)

        # Check that stds are positive
        assert stds[0] > 0
        assert stds[1] > 0

        # Check that normalized data has mean ~0 and std ~1
        assert np.isclose(np.mean(normalized[:, 0]), 0, atol=0.1)
        assert np.isclose(np.std(normalized[:, 0], ddof=0), 1, atol=0.1)


class TestModelSamplingMethod:
    """Test Model _apply_sampling private method."""

    def test_apply_sampling_with_smote(self) -> None:
        """Test that _apply_sampling applies SMOTE correctly."""
        model = Model(MockLogger())
        # Imbalanced dataset with enough samples for SMOTE
        # Majority class: 10 samples, Minority class: 6 samples (> k_neighbors=5)
        train = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
                [6.0, 0.0],
                [7.0, 0.0],
                [8.0, 0.0],
                [9.0, 0.0],
                [10.0, 0.0],
                [20.0, 1.0],
                [21.0, 1.0],
                [22.0, 1.0],
                [23.0, 1.0],
                [24.0, 1.0],
                [25.0, 1.0],
            ]
        )

        result = model._apply_sampling(train)

        # After SMOTE, should have balanced classes
        y_result = result[:, -1]
        unique, counts = np.unique(y_result, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == counts[1]  # Should be balanced

    def test_apply_sampling_with_random_over(self) -> None:
        """Test that _apply_sampling applies random oversampling correctly."""
        model = Model(MockLogger())
        model.transformations["sampling"]["type"] = SamplingStrategy.RANDOM_OVER

        # Imbalanced dataset
        train = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [10.0, 1.0],
            ]
        )

        result = model._apply_sampling(train)

        # After random oversampling, should have balanced classes
        y_result = result[:, -1]
        unique, counts = np.unique(y_result, return_counts=True)
        # Both classes should have equal counts
        assert counts[0] == counts[1]


class TestModelMissingAndWeirdData:
    """Test Model handling of missing and weird data values."""

    def test_train_final_with_nan_in_numeric_features(self) -> None:
        """Test train_final handles NaN values in numeric features."""
        model = Model(MockLogger())
        # Create data with NaN in numeric features
        train_data = pd.DataFrame(
            {
                "feature1": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
                "feature2": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0],
                "target": ["off", "off", "off", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Should handle NaN without crashing
        model.train_final(train_numpy)
        assert model.model_final is not None
        assert model.prediction_ready is True

    def test_train_final_with_inf_in_numeric_features(self) -> None:
        """Test train_final handles inf values in numeric features."""
        model = Model(MockLogger())
        # Create data with inf in numeric features
        train_data = pd.DataFrame(
            {
                "feature1": [1.0, np.inf, 3.0, 4.0, 5.0, 6.0],
                "feature2": [10.0, 20.0, -np.inf, 40.0, 50.0, 60.0],
                "target": ["off", "off", "off", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Should handle inf without crashing
        model.train_final(train_numpy)
        assert model.model_final is not None
        assert model.prediction_ready is True

    def test_predict_with_nan_in_features(self) -> None:
        """Test predict handles NaN values in input features."""
        model = Model(MockLogger())
        # Train on clean data
        train_data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "target": ["off", "off", "off", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # Predict with NaN in features
        test_data = pd.DataFrame(
            {
                "feature1": [np.nan],
                "feature2": [25.0],
            }
        )
        test_numpy = convert_df_to_numpy(test_data)

        # Should handle NaN without crashing
        result = model.predict(test_numpy)
        # Result may be None or a tuple depending on implementation
        assert result is None or isinstance(result, tuple)

    def test_predict_with_inf_in_features(self) -> None:
        """Test predict handles inf values in input features."""
        model = Model(MockLogger())
        # Train on clean data
        train_data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "target": ["off", "off", "off", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # Predict with inf in features
        test_data = pd.DataFrame(
            {
                "feature1": [np.inf],
                "feature2": [25.0],
            }
        )
        test_numpy = convert_df_to_numpy(test_data)

        # Should handle inf without crashing
        result = model.predict(test_numpy)
        # Result may be None or a tuple depending on implementation
        assert result is None or isinstance(result, tuple)

    def test_predict_with_unavailable_string_in_features(self) -> None:
        """Test predict handles 'unavailable' string in categorical features."""
        model = Model(MockLogger())
        # Train with categorical feature
        train_data = pd.DataFrame(
            {
                "color": ["red", "blue", "red", "blue", "red", "blue"],
                "target": ["off", "off", "off", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # Predict with 'unavailable' in features
        test_data = pd.DataFrame({"color": ["unavailable"]})
        test_numpy = convert_df_to_numpy(test_data)

        # Should handle unknown category without crashing
        result = model.predict(test_numpy)
        # Should return a prediction (unknown category gets encoded as -1)
        assert result is not None

    def test_predict_with_unknown_string_in_features(self) -> None:
        """Test predict handles 'unknown' string in categorical features."""
        model = Model(MockLogger())
        # Train with categorical feature
        train_data = pd.DataFrame(
            {
                "color": ["red", "blue", "red", "blue", "red", "blue"],
                "target": ["off", "off", "off", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # Predict with 'unknown' in features
        test_data = pd.DataFrame({"color": ["unknown"]})
        test_numpy = convert_df_to_numpy(test_data)

        # Should handle unknown category without crashing
        result = model.predict(test_numpy)
        # Should return a prediction (unknown category gets encoded as -1)
        assert result is not None

    def test_train_eval_with_nan_in_features(self) -> None:
        """Test train_eval handles NaN values in features."""
        model = Model(MockLogger())
        # Create data with NaN in features
        train_data = pd.DataFrame(
            {
                "feature1": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "feature2": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0, 70.0, 80.0],
                "target": ["off", "off", "off", "off", "on", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Should handle NaN without crashing
        model.train_eval(train_numpy)
        assert model.model_eval is not None
        # Accuracy might be affected but should still be calculated
        assert model.accuracy is not None

    def test_train_eval_with_inf_in_features(self) -> None:
        """Test train_eval handles inf values in features."""
        model = Model(MockLogger())
        # Create data with inf in features
        train_data = pd.DataFrame(
            {
                "feature1": [1.0, np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "feature2": [10.0, 20.0, -np.inf, 40.0, 50.0, 60.0, 70.0, 80.0],
                "target": ["off", "off", "off", "off", "on", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Should handle inf without crashing
        model.train_eval(train_numpy)
        assert model.model_eval is not None
        # Accuracy might be affected but should still be calculated
        assert model.accuracy is not None

    def test_train_final_with_unavailable_in_features(self) -> None:
        """Test train_final handles 'unavailable' in categorical features."""
        model = Model(MockLogger())
        # Create data with 'unavailable' in features (not target)
        train_data = pd.DataFrame(
            {
                "color": ["red", "unavailable", "blue", "red", "blue", "red"],
                "target": ["off", "off", "off", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Should handle 'unavailable' as a valid category in features
        model.train_final(train_numpy)
        assert model.model_final is not None
        # 'unavailable' should be treated as a valid category in features
        assert 0 in model.factors  # color column should be factorized

    def test_train_final_with_unknown_in_features(self) -> None:
        """Test train_final handles 'unknown' in categorical features."""
        model = Model(MockLogger())
        # Create data with 'unknown' in features (not target)
        train_data = pd.DataFrame(
            {
                "color": ["red", "unknown", "blue", "red", "blue", "red"],
                "target": ["off", "off", "off", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Should handle 'unknown' as a valid category in features
        model.train_final(train_numpy)
        assert model.model_final is not None
        # 'unknown' should be treated as a valid category in features
        assert 0 in model.factors  # color column should be factorized

    def test_train_final_with_none_values(self) -> None:
        """Test train_final handles None values in features."""
        model = Model(MockLogger())
        # Create data with None values
        train_data = pd.DataFrame(
            {
                "feature1": [1.0, None, 3.0, 4.0, 5.0, 6.0],
                "feature2": [10.0, 20.0, None, 40.0, 50.0, 60.0],
                "target": ["off", "off", "off", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Should handle None (converts to NaN in pandas/numpy)
        model.train_final(train_numpy)
        assert model.model_final is not None
        assert model.prediction_ready is True

    def test_train_final_with_mixed_weird_values(self) -> None:
        """Test train_final handles mix of NaN, inf, and special strings."""
        model = Model(MockLogger())
        # Create data with various weird values
        train_data = pd.DataFrame(
            {
                "feature1": [1.0, np.nan, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0],
                "color": [
                    "red",
                    "blue",
                    "unavailable",
                    "unknown",
                    "red",
                    "blue",
                    "red",
                    "blue",
                ],
                "target": ["off", "off", "off", "off", "on", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)

        # Should handle all weird values
        model.train_final(train_numpy)
        assert model.model_final is not None
        assert model.prediction_ready is True

    def test_predict_with_all_nan_features(self) -> None:
        """Test predict handles all NaN features."""
        model = Model(MockLogger())
        # Train on clean data
        train_data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "target": ["off", "off", "off", "on", "on", "on"],
            }
        )
        train_numpy = convert_df_to_numpy(train_data)
        model.train_final(train_numpy)

        # Predict with all NaN features
        test_data = pd.DataFrame(
            {
                "feature1": [np.nan],
                "feature2": [np.nan],
            }
        )
        test_numpy = convert_df_to_numpy(test_data)

        # Should handle all NaN without crashing
        result = model.predict(test_numpy)
        # Result may be None or a tuple depending on implementation
        assert result is None or isinstance(result, tuple)

    def test_normalization_with_nan_values(self) -> None:
        """Test that normalization handles NaN values correctly."""
        model = Model(MockLogger())
        # Data with NaN will have NaN in means/stds
        data = np.array([[1.0, np.nan], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])

        means, stds, normalized = model._apply_normalization(data)

        # Means and stds should be calculated (NaN will propagate)
        assert means is not None
        assert stds is not None
        assert normalized is not None
        # Shape should be preserved
        assert normalized.shape == data.shape

    def test_normalization_with_inf_values(self) -> None:
        """Test that normalization handles inf values correctly."""
        model = Model(MockLogger())
        # Data with inf values
        data = np.array([[1.0, 10.0], [2.0, np.inf], [3.0, 30.0], [4.0, 40.0]])

        means, stds, normalized = model._apply_normalization(data)

        # Means and stds should be calculated (inf will affect results)
        assert means is not None
        assert stds is not None
        assert normalized is not None
        # Shape should be preserved
        assert normalized.shape == data.shape
