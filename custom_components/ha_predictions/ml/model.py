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

    def __init__(self, logger: Logger):
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
        """
        Train the model.

        Parameters
        ----------
        - data: DataFrame with features and target
        - target_col: Name of target column (if None, assumes last column)

        """
        data_copy = data.copy()

        # Determine target column
        if target_col is None:
            self.target_column = data_copy.columns[-1]
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
        self.logger.info("Starting training for evaluation with data: %s", str(df))
        categories = {}
        for col in df.select_dtypes(include=["object"]).columns:
            codes, uniques = pd.factorize(df[col])
            df[col] = codes
            categories[col] = uniques

        # train/test split in pure numpy
        # TODO: Stratify split based on last column
        dfn = df.to_numpy()
        np.random.Generator(np.random.PCG64()).shuffle(dfn)

        nrows = dfn.shape[0]
        test_size = max(int(nrows * 0.25), 1)
        train, test = dfn[:test_size, :], dfn[test_size:, :]
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
