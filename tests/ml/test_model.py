"""Unit tests for Model class."""

import sys
from pathlib import Path
from types import NoneType
from typing import Any, ClassVar

import pandas as pd

# Add the ml directory to the path to avoid importing homeassistant dependencies
ml_path = (
    Path(__file__).parent.parent.parent
    / "custom_components"
    / "ha_predictions"
    / "ml"
)
sys.path.insert(0, str(ml_path))

from LogisticRegression import LogisticRegression  # noqa: E402


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

    def predict(self, data: pd.DataFrame) -> tuple[str, float] | NoneType:
        """Make predictions and return original values."""
        msg = "Model not trained yet."
        if self.model_final is None:
            raise ValueError(msg)

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
        model.train_final(train_data)

        # Predict on a sample that should be class 0 ("off") - using extreme low value
        test_data = pd.DataFrame({"feature1": [0], "feature2": [0]})
        label, probability = model.predict(test_data)

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
        model.train_final(train_data)

        # Predict on a sample that should be class 1 ("on") - using extreme high value
        test_data = pd.DataFrame({"feature1": [19], "feature2": [19]})
        label, probability = model.predict(test_data)

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
        model.train_final(train_data)

        # Test multiple predictions
        test_values = [1.5, 3.5, 5.5, 7.5, 9.5]
        for val in test_values:
            test_data = pd.DataFrame({"feature1": [val], "feature2": [val]})
            label, probability = model.predict(test_data)

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
        model.train_final(train_data)

        # Test predictions at extremes
        test_data_low = pd.DataFrame({"feature": [0]})
        label_low, prob_low = model.predict(test_data_low)

        test_data_high = pd.DataFrame({"feature": [11]})
        label_high, prob_high = model.predict(test_data_high)

        # Both probabilities should represent confidence in predicted class
        assert prob_low >= 0.5
        assert prob_high >= 0.5

        # Predictions at extremes should be different classes
        assert label_low != label_high
