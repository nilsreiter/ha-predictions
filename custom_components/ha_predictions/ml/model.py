"""ML Model management."""

from logging import Logger
from types import NoneType
from typing import Any

import numpy as np

from .exceptions import ModelNotTreainedError
from .logistic_regression import LogisticRegression
class Model:
    """Class to manage the ML model instance."""

    def __init__(self, logger: Logger) -> None:
        """Initialize the Model class."""
        self.logger = logger
        self.accuracy: float | NoneType = None
        self.factors: dict[str, Any] = {}
        self.model_eval: LogisticRegression | None = None
        self.model_final: LogisticRegression | None = None
        self.target_column: str | None = None
        self.prediction_ready: bool = False

    def predict(self, data: np.ndarray) -> tuple[str, float] | NoneType:
        """Make predictions and return original values.
        
        Args:
            data: Numpy array of feature values (already encoded/factorized)
        
        Returns:
            Tuple of (predicted_label, probability) or None
        """
        if self.model_final is None:
            raise ModelNotTreainedError

        # Predict
        predictions, probabilities = self.model_final.predict(data)

        if (
            self.target_column in self.factors
            and predictions is not None
            and probabilities is not None
        ):
            target_categories = self.factors[self.target_column]
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
        factors: dict[str, Any],
        target_column: str,
    ) -> None:
        """Train the final model.
        
        Args:
            data: Numpy array with features and target (already encoded/factorized)
            factors: Dictionary mapping column names to their category mappings
            target_column: Name of the target column
        """
        # Store factors and target column for decoding predictions
        self.factors = factors
        self.target_column = target_column

        # Split features and target
        x_train = data[:, :-1]
        y_train = data[:, -1]

        # Train model
        self.model_final = LogisticRegression()
        self.logger.debug("Training of final model begins")
        self.model_final.fit(x_train, y_train)
        self.logger.debug("Training ends, model: %s", str(self.model_final))
        self.prediction_ready = True

    def train_eval(self, data: np.ndarray) -> NoneType:
        """Train and evaluate the model with train/test split.
        
        Args:
            data: Numpy array with features and target (already encoded/factorized)
        """
        self.logger.info("Starting training for evaluation with data: %s", str(data))

        # train/test split in pure numpy with stratification
        rng = np.random.Generator(np.random.PCG64())

        # Get target column (last column)
        y = data[:, -1]
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
            test_size_cls = max(int(n_cls * 0.25), 1) if n_cls >= 2 else 0

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

        train = data[train_indices, :]
        test = data[test_indices, :]
        self.logger.debug("Data used for training: %s", str(train))
        self.logger.debug("Data used for testing: %s", str(test))

        # Split x and y
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = test[:, :-1]
        y_test = test[:, -1]

        self.model_eval = LogisticRegression()
        self.logger.debug("Training begins")
        self.model_eval.fit(x_train, y_train)
        self.logger.debug("Training ends, model: %s", str(self.model_eval))
        self.accuracy = self.model_eval.score(x_test, y_test)
