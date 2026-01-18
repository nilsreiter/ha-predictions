"""ML Model management."""

from logging import Logger
from types import NoneType
from typing import Any

import numpy as np

from .const import SamplingStrategy
from .evaluation import accuracy, precision_recall_fscore
from .exceptions import ModelNotTrainedError
from .logistic_regression import LogisticRegression
from .sampling import random_oversample, smote


class Model:
    """Class to manage the ML model instance."""

    def __init__(self, logger: Logger) -> None:
        """Initialize the Model class."""
        self.logger = logger
        self.scores: tuple[float, dict[Any, dict[str, float]]] | NoneType = None
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

        data = self._apply_filtering(data)

        # Factorize categorical columns using numpy.unique
        # Create a new array with float dtype to avoid object dtype issues
        data_encoded, self.factors = self._factorize(data)

        data_encoded = self._apply_sampling(data_encoded)

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

        filtered_arr = self._apply_filtering(data)
        self.logger.debug("Filtered data: %s", str(filtered_arr))

        # Factorize categorical columns using numpy.unique
        # Create a new array with float dtype to avoid object dtype issues
        data_encoded, factors = self._factorize(filtered_arr)

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

        y_pred = self.model_eval.predict(x_test)[0]
        if y_pred is not None:
            # Use the actual target column index (last column) to get class labels
            target_col_idx = data_encoded.shape[1] - 1
            class_labels = factors.get(target_col_idx)
            self.scores = (
                accuracy(y_pred, y_test),
                precision_recall_fscore(
                    y_pred,
                    y_test,
                    class_labels=class_labels,
                ),
            )
            self.logger.debug(
                "Evaluation results - Accuracy: %s",
                str(self.scores[0]),
            )
        else:
            self.scores = None

    def _factorize(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Factorize categorical columns in data.

        Args:
            data: Numpy array with feature data.

        Returns:
            Tuple of (encoded_data, factors).

        """
        data_encoded = np.empty(data.shape, dtype=float)
        factors = {}
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
                factors[col_idx] = unique_values
                # Replace column with encoded indices
                data_encoded[:, col_idx] = inverse_indices.astype(float)
            else:
                # Copy numeric data as-is
                data_encoded[:, col_idx] = column_data.astype(float)
        return data_encoded, factors

    def _apply_filtering(
        self,
        data: np.ndarray,
        *,
        filter_unavailable: bool = True,
        filter_unknown: bool = True,
    ) -> np.ndarray:
        """
        Apply filtering to data.

        Args:
            data: Numpy array with feature data.
            filter_unavailable: Whether to filter out 'Unavailable' target rows.
            filter_unknown: Whether to filter out 'unknown' target rows.

        Returns:
            Numpy array with filtered data.

        """
        # Placeholder for future filtering logic
        self.logger.debug("Applying filters on %i instances", data.shape[0])

        # Remove rows with 'unavailable' in the target column (last column)
        if filter_unavailable:
            data = data[data[:, -1] != "unavailable"]
            data = data[data[:, -1] != "Unavailable"]

        # Remove rows with 'unknown' in the target column
        if filter_unknown:
            data = data[data[:, -1] != "unknown"]
            data = data[data[:, -1] != "Unknown"]

        self.logger.debug("Number of instances after filtering: %i", data.shape[0])
        return data

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
                self.logger.debug("Training data after SMOTE: %s", train)
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
                self.logger.debug("Training data after random oversampling: %s", train)
        return train
