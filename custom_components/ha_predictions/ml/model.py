"""ML Model management."""

from logging import Logger
from types import NoneType
from typing import Any

import numpy as np
import pandas as pd

from .LogisticRegression import LogisticRegression


class Model:
    """Class to manage the ML model instance."""

    accuracy: float | NoneType = None
    factors: dict[str, Any] = {}
    model_eval: LogisticRegression | None = None
    model_final: LogisticRegression | None = None
    target_column: str | None = None
    prediction_ready: bool = False

    def __init__(self, logger: Logger) -> None:
        """Initialize the Model class."""
        self.logger = logger

    def predict(self, data: pd.DataFrame) -> tuple[str, float] | NoneType:
        """Make predictions and return original values."""
        if self.model_final is None:
            raise ValueError("Model not trained yet.")

        data_copy = data.copy()

        # Apply factorization to features only
        for col, categories in self.factors.items():
            if col == self.target_column:
                continue
            if col in data_copy.columns:
                category_to_code = {val: idx for idx, val in enumerate(categories)}
                data_copy[col] = (
                    data_copy[col].map(category_to_code).fillna(-1).astype(int)
                )

        # Predict
        x_pred = data_copy.to_numpy()
        predictions, probabilities = self.model_final.predict(x_pred)

        # TODO: Check if probabilities correspond to predicted class and if predicted class is correct. Maybe probabilities need to be adjusted.
        # Decode to original values and get probability for predicted class
        if (
            self.target_column in self.factors
            and predictions is not None
            and probabilities is not None
        ):  # Changed from prediction_codes to predictions
            target_categories = self.factors[self.target_column]
            label = target_categories[predictions[0]]
            probability = probabilities[0]  # Probability of the predicted class
            return (label, probability)
        return None

    def train_final(self, data: pd.DataFrame, target_col: str | None = None) -> None:
        """Train the final model."""
        data_copy = data.copy()

        # Determine target column
        if target_col is None:
            self.target_column = data_copy.columns.tolist()[-1]
        else:
            self.target_column = target_col

        # Factorize categorical columns
        for col in data_copy.select_dtypes(include=["object"]).columns:
            codes, uniques = pd.factorize(data_copy[col])
            data_copy[col] = codes
            self.factors[col] = uniques

        # Convert to numpy
        dfn = data_copy.to_numpy()
        x_train = dfn[:, :-1]
        y_train = dfn[:, -1]

        # Train model
        self.model_final = LogisticRegression()
        self.logger.debug("Training of final model begins")
        self.model_final.fit(x_train, y_train)
        self.logger.debug("Training ends, model: %s", str(self.model_final))
        self.prediction_ready = True

    def train_eval(self, df: pd.DataFrame) -> NoneType:
        """Train and evaluate the model with train/test split."""
        self.logger.info("Starting training for evaluation with data: %s", str(df))
        categories = {}
        for col in df.select_dtypes(include=["object"]).columns:
            codes, uniques = pd.factorize(df[col])
            df[col] = codes
            categories[col] = uniques

        # train/test split in pure numpy with stratification
        dfn = df.to_numpy()
        rng = np.random.Generator(np.random.PCG64())

        # Get target column (last column)
        y = dfn[:, -1]
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

        train = dfn[train_indices, :]
        test = dfn[test_indices, :]
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
