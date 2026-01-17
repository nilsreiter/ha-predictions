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

    def predict(
        self, data: np.ndarray, col_names: list[str] | None = None
    ) -> tuple[str, float] | NoneType:
        """Make predictions and return original values.
        
        Args:
            data: Numpy array of feature values (raw, not encoded)
            col_names: List of column names corresponding to data columns
        
        Returns:
            Tuple of (predicted_label, probability) or None
        """
        msg = "Model not trained yet."
        if self.model_final is None:
            raise ValueError(msg)

        # Apply factorization to features only using stored factors
        data_encoded = data.copy()
        
        if col_names:
            for col_idx, col_name in enumerate(col_names):
                if col_name in self.factors and col_name != self.target_column:
                    value = data[0, col_idx]
                    categories = self.factors[col_name]
                    
                    # Find the index of the value in categories using numpy
                    try:
                        idx = np.where(categories == value)[0]
                        if len(idx) > 0:
                            data_encoded[0, col_idx] = idx[0]
                        else:
                            # Value not found in training data, use -1
                            data_encoded[0, col_idx] = -1
                    except (ValueError, TypeError):
                        data_encoded[0, col_idx] = -1

        # Predict
        predictions, probabilities = self.model_final.predict(data_encoded)

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
        col_names: list[str],
        target_column: str | None = None,
    ) -> None:
        """Train the final model.
        
        Args:
            data: Numpy array with features and target (not encoded)
            col_names: List of column names corresponding to data columns
            target_column: Name of the target column (optional, defaults to last column)
        """
        # Determine target column
        if target_column is None:
            self.target_column = col_names[-1]
        else:
            self.target_column = target_column

        # Factorize categorical columns using numpy.unique
        data_encoded = data.copy()
        self.factors = {}
        
        for col_idx, col_name in enumerate(col_names):
            column_data = data[:, col_idx]
            
            # Check if column contains non-numeric data
            if column_data.dtype == object or not np.issubdtype(column_data.dtype, np.number):
                # Use numpy.unique to get unique values and their indices
                unique_values, inverse_indices = np.unique(column_data, return_inverse=True)
                # Store the unique values for later decoding
                self.factors[col_name] = unique_values
                # Replace column with encoded indices
                data_encoded[:, col_idx] = inverse_indices

        # Convert to appropriate numeric type
        data_encoded = data_encoded.astype(float)

        # Split features and target
        x_train = data_encoded[:, :-1]
        y_train = data_encoded[:, -1]

        # Train model
        self.model_final = LogisticRegression()
        self.logger.debug("Training of final model begins")
        self.model_final.fit(x_train, y_train)
        self.logger.debug("Training ends, model: %s", str(self.model_final))
        self.prediction_ready = True


def factorize_with_numpy(df: pd.DataFrame) -> tuple[np.ndarray, list[str], str]:
    """Helper function to convert DataFrame to numpy for testing."""
    # Simply convert to numpy and return column names
    data_array = df.to_numpy()
    col_names = df.columns.tolist()
    target_column = col_names[-1]
    
    return data_array, col_names, target_column


def encode_instance(instance_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Helper function to convert instance DataFrame to numpy."""
    # Simply convert to numpy and return column names
    instance_array = instance_df.to_numpy()
    col_names = instance_df.columns.tolist()
    
    return instance_array, col_names


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
        train_numpy, col_names, target_col = factorize_with_numpy(train_data)
        model.train_final(train_numpy, col_names, target_col)

        # Predict on a sample that should be class 0 ("off") - using extreme low value
        test_data = pd.DataFrame({"feature1": [0], "feature2": [0]})
        test_numpy, test_col_names = encode_instance(test_data)
        label, probability = model.predict(test_numpy, test_col_names)

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
        train_numpy, col_names, target_col = factorize_with_numpy(train_data)
        model.train_final(train_numpy, col_names, target_col)

        # Predict on a sample that should be class 1 ("on") - using extreme high value
        test_data = pd.DataFrame({"feature1": [19], "feature2": [19]})
        test_numpy, test_col_names = encode_instance(test_data)
        label, probability = model.predict(test_numpy, test_col_names)

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
        train_numpy, col_names, target_col = factorize_with_numpy(train_data)
        model.train_final(train_numpy, col_names, target_col)

        # Test multiple predictions
        test_values = [1.5, 3.5, 5.5, 7.5, 9.5]
        for val in test_values:
            test_data = pd.DataFrame({"feature1": [val], "feature2": [val]})
            test_numpy, test_col_names = encode_instance(test_data)
            label, probability = model.predict(test_numpy, test_col_names)

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
        train_numpy, col_names, target_col = factorize_with_numpy(train_data)
        model.train_final(train_numpy, col_names, target_col)

        # Test predictions at extremes
        test_data_low = pd.DataFrame({"feature": [0]})
        test_numpy_low, test_col_names_low = encode_instance(test_data_low)
        label_low, prob_low = model.predict(test_numpy_low, test_col_names_low)

        test_data_high = pd.DataFrame({"feature": [11]})
        test_numpy_high, test_col_names_high = encode_instance(test_data_high)
        label_high, prob_high = model.predict(test_numpy_high, test_col_names_high)

        # Both probabilities should represent confidence in predicted class
        assert prob_low >= 0.5
        assert prob_high >= 0.5

        # Predictions at extremes should be different classes
        assert label_low != label_high
        assert label_low != label_high
