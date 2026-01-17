"""Unit tests for Model class."""

import sys
from pathlib import Path
from types import NoneType
from typing import Any, ClassVar

import numpy as np
import pandas as pd

# Add the ml directory to the path to avoid importing homeassistant dependencies
ml_path = (
    Path(__file__).parent.parent.parent
    / "custom_components"
    / "ha_predictions"
    / "ml"
)
sys.path.insert(0, str(ml_path))

from logistic_regression import LogisticRegression  # noqa: E402


class MockLogger:
    """Mock logger for testing."""

    def debug(self, *args: Any, **kwargs: Any) -> None:
        """Mock debug method."""

    def info(self, *args: Any, **kwargs: Any) -> None:
        """Mock info method."""


class Model:
    """Minimal Model class for testing (copied from model.py)."""

    accuracy: float | NoneType = None
    factors: ClassVar[dict[str, Any]] = {}
    model_eval: LogisticRegression | None = None
    model_final: LogisticRegression | None = None
    target_column: str | None = None
    prediction_ready: bool = False

    def __init__(self, logger: MockLogger) -> None:
        """Initialize the Model class."""
        self.logger = logger
        self.factors = {}

    def predict(self, data: np.ndarray) -> tuple[str, float] | NoneType:
        """Make predictions and return original values.
        
        Args:
            data: Numpy array of feature values (already encoded/factorized)
        
        Returns:
            Tuple of (predicted_label, probability) or None
        """
        msg = "Model not trained yet."
        if self.model_final is None:
            raise ValueError(msg)

        # Predict
        predictions, probabilities = self.model_final.predict(data)

        # Decode to original values and get probability for predicted class
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


def factorize_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], str]:
    """Helper function to factorize a DataFrame for testing."""
    df_copy = df.copy()
    factors: dict[str, Any] = {}
    target_column = df_copy.columns.tolist()[-1]
    
    # Factorize categorical columns
    for col in df_copy.select_dtypes(include=["object"]).columns:
        codes, uniques = pd.factorize(df_copy[col])
        df_copy[col] = codes
        factors[col] = uniques
    
    return df_copy.to_numpy(), factors, target_column


def encode_instance(instance_df: pd.DataFrame, factors: dict[str, Any], target_column: str) -> np.ndarray:
    """Helper function to encode an instance using factors."""
    instance_copy = instance_df.copy()
    
    # Apply factorization to features only
    for col, categories in factors.items():
        if col == target_column:
            continue
        if col in instance_copy.columns:
            category_to_code = {val: idx for idx, val in enumerate(categories)}
            instance_copy[col] = (
                instance_copy[col].map(category_to_code).fillna(-1).astype(int)
            )
    
    return instance_copy.to_numpy()


class TestModelPredictProbabilities:
    """Test that probabilities correspond to predicted class."""

    def test_probability_corresponds_to_predicted_class_0(self) -> None:
        """Test that probability for class 0 is correctly calculated."""
        model = Model(MockLogger())

        # Train on data with more separation and more samples
        train_data = pd.DataFrame({
            "feature1": [0, 1, 2, 3, 4, 15, 16, 17, 18, 19],
            "feature2": [0, 1, 2, 3, 4, 15, 16, 17, 18, 19],
            "target": ["off", "off", "off", "off", "off", "on", "on", "on", "on", "on"],
        })
        train_numpy, factors, target_col = factorize_dataframe(train_data)
        model.train_final(train_numpy, factors, target_col)

        # Predict on a sample that should be class 0 ("off") - using extreme low value
        test_data = pd.DataFrame({"feature1": [0], "feature2": [0]})
        test_numpy = encode_instance(test_data, factors, target_col)
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
        train_data = pd.DataFrame({
            "feature1": [0, 1, 2, 3, 4, 15, 16, 17, 18, 19],
            "feature2": [0, 1, 2, 3, 4, 15, 16, 17, 18, 19],
            "target": ["off", "off", "off", "off", "off", "on", "on", "on", "on", "on"],
        })
        train_numpy, factors, target_col = factorize_dataframe(train_data)
        model.train_final(train_numpy, factors, target_col)

        # Predict on a sample that should be class 1 ("on") - using extreme high value
        test_data = pd.DataFrame({"feature1": [19], "feature2": [19]})
        test_numpy = encode_instance(test_data, factors, target_col)
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
        train_data = pd.DataFrame({
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
        })
        train_numpy, factors, target_col = factorize_dataframe(train_data)
        model.train_final(train_numpy, factors, target_col)

        # Test multiple predictions
        test_values = [1.5, 3.5, 5.5, 7.5, 9.5]
        for val in test_values:
            test_data = pd.DataFrame({"feature1": [val], "feature2": [val]})
            test_numpy = encode_instance(test_data, factors, target_col)
            label, probability = model.predict(test_numpy)

            # Probability should always be >= 0.5 since it represents
            # confidence in the predicted class (after threshold of 0.5)
            assert (
                probability >= 0.5
            ), f"Probability {probability} < 0.5 for value {val}, label {label}"

    def test_probability_with_edge_case_sigmoid_values(self) -> None:
        """Test probability calculation with edge case sigmoid values."""
        model = Model(MockLogger())

        # Train with simple data
        train_data = pd.DataFrame({
            "feature": [0, 1, 10, 11],
            "target": ["no", "no", "yes", "yes"],
        })
        train_numpy, factors, target_col = factorize_dataframe(train_data)
        model.train_final(train_numpy, factors, target_col)

        # Test predictions at extremes
        test_data_low = pd.DataFrame({"feature": [0]})
        test_numpy_low = encode_instance(test_data_low, factors, target_col)
        label_low, prob_low = model.predict(test_numpy_low)

        test_data_high = pd.DataFrame({"feature": [11]})
        test_numpy_high = encode_instance(test_data_high, factors, target_col)
        label_high, prob_high = model.predict(test_numpy_high)

        # Both probabilities should represent confidence in predicted class
        assert prob_low >= 0.5
        assert prob_high >= 0.5

        # Predictions at extremes should be different classes
        assert label_low != label_high
