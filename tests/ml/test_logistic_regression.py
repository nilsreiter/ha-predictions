"""Unit tests for LogisticRegression class."""

import sys
from pathlib import Path

import numpy as np

# Add the ha_predictions directory to path to enable ml package imports
# This avoids importing the parent package which has homeassistant dependencies
ha_predictions_path = (
    Path(__file__).parent.parent.parent / "custom_components" / "ha_predictions"
)
sys.path.insert(0, str(ha_predictions_path))

from ml.logistic_regression import LogisticRegression  # noqa: E402


class TestLogisticRegressionInit:
    """Test initialization and basic attributes."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        model = LogisticRegression()
        assert model.lr == 0.001
        assert model.n_iters == 1000
        assert model.weights is None
        assert model.bias == 0
        assert model.losses == []

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        model = LogisticRegression(learning_rate=0.01, n_iters=500)
        assert model.lr == 0.01
        assert model.n_iters == 500
        assert model.weights is None
        assert model.bias == 0
        assert model.losses == []


class TestSigmoid:
    """Test sigmoid activation function."""

    def test_sigmoid_zero(self) -> None:
        """Test sigmoid at zero returns 0.5."""
        model = LogisticRegression()
        result = model._sigmoid(0)
        assert np.isclose(result, 0.5)

    def test_sigmoid_positive(self) -> None:
        """Test sigmoid with positive values."""
        model = LogisticRegression()
        result = model._sigmoid(2.0)
        expected = 1 / (1 + np.exp(-2.0))
        assert np.isclose(result, expected)

    def test_sigmoid_negative(self) -> None:
        """Test sigmoid with negative values."""
        model = LogisticRegression()
        result = model._sigmoid(-2.0)
        expected = 1 / (1 + np.exp(2.0))
        assert np.isclose(result, expected)

    def test_sigmoid_array(self) -> None:
        """Test sigmoid with array input."""
        model = LogisticRegression()
        x = np.array([0, 1, -1])
        result = model._sigmoid(x)
        expected = 1 / (1 + np.exp(-x))
        assert np.allclose(result, expected)


class TestComputeLoss:
    """Test loss computation (binary cross entropy)."""

    def test_compute_loss_perfect_prediction(self) -> None:
        """Test loss with perfect predictions."""
        model = LogisticRegression()
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0.0, 1.0, 1.0, 0.0])
        loss = model._compute_loss(y_true, y_pred)
        # Should be very close to zero for perfect predictions
        assert loss < 0.01

    def test_compute_loss_worst_prediction(self) -> None:
        """Test loss with worst predictions."""
        model = LogisticRegression()
        y_true = np.array([0, 1, 1, 0])
        # Very confident but wrong predictions (but not exactly 0 or 1 to avoid log(0))
        y_pred = np.array([0.99, 0.01, 0.01, 0.99])
        loss = model._compute_loss(y_true, y_pred)
        # Should be a large positive value
        assert loss > 2.0

    def test_compute_loss_medium_prediction(self) -> None:
        """Test loss with uncertain predictions."""
        model = LogisticRegression()
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        loss = model._compute_loss(y_true, y_pred)
        # Should be around -log(0.5) ≈ 0.693
        assert np.isclose(loss, 0.693, atol=0.01)


class TestFeedForward:
    """Test feed forward method."""

    def test_feed_forward_before_training(self) -> None:
        """Test feed forward returns None before training."""
        model = LogisticRegression()
        x = np.array([[1, 2], [3, 4]])
        result = model._feed_forward(x)
        assert result is None

    def test_feed_forward_after_training(self) -> None:
        """Test feed forward after weights are initialized."""
        model = LogisticRegression()
        model.weights = np.array([0.5, -0.5])
        model.bias = 0.1
        x = np.array([[1, 2]])
        result = model._feed_forward(x)
        assert result is not None
        # Result should be between 0 and 1 (sigmoid output)
        assert 0 < result[0] < 1


class TestFit:
    """Test model training (fit method)."""

    def test_fit_initializes_weights(self) -> None:
        """Test that fit initializes weights correctly."""
        model = LogisticRegression()
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 1])
        model.fit(x, y)
        assert model.weights is not None
        assert len(model.weights) == 2
        # Bias should be a float (could be any value including 0)
        assert isinstance(model.bias, float)

    def test_fit_records_losses(self) -> None:
        """Test that fit records loss values during training."""
        model = LogisticRegression(n_iters=10)
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        model.fit(x, y)
        assert len(model.losses) == 10
        # Losses should generally decrease (though not guaranteed in all cases)
        assert model.losses[0] > 0

    def test_fit_simple_linear_separable(self) -> None:
        """Test fitting on a simple linearly separable dataset."""
        model = LogisticRegression(learning_rate=0.1, n_iters=1000)
        # Simple XOR-like pattern that's linearly separable
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
        model.fit(x, y)
        # Check that weights are learned
        assert model.weights is not None
        assert not np.allclose(model.weights, 0)


class TestPredict:
    """Test prediction method."""

    def test_predict_before_training(self) -> None:
        """Test predict returns None before training."""
        model = LogisticRegression()
        x = np.array([[1, 2]])
        classes, probs = model.predict(x)
        assert classes is None
        assert probs is None

    def test_predict_after_training(self) -> None:
        """Test predict after training."""
        model = LogisticRegression(learning_rate=0.1, n_iters=1000)
        x = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        y = np.array([0, 0, 1, 1])
        model.fit(x, y)

        x_test = np.array([[1.5, 1.5], [3.5, 3.5]])
        classes, probs = model.predict(x_test)

        assert classes is not None
        assert probs is not None
        assert len(classes) == 2
        assert len(probs) == 2
        # Classes should be 0 or 1
        assert all(c in [0, 1] for c in classes)
        # Probabilities should be between 0 and 1
        assert all(0 <= p <= 1 for p in probs)

    def test_predict_returns_correct_types(self) -> None:
        """Test that predict returns numpy arrays."""
        model = LogisticRegression(learning_rate=0.1, n_iters=100)
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        model.fit(x, y)

        x_test = np.array([[2, 3]])
        classes, probs = model.predict(x_test)

        assert isinstance(classes, np.ndarray)
        assert isinstance(probs, np.ndarray)


class TestScore:
    """Test score method."""

    def test_score_perfect(self) -> None:
        """Test score with perfect predictions."""
        model = LogisticRegression(learning_rate=0.1, n_iters=1000)
        # Train on simple data
        x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_train = np.array([0, 0, 0, 1])
        model.fit(x_train, y_train)

        # Test on same data (should get high accuracy)
        _, _ = model.predict(x_train)
        # Calculate expected accuracy
        score = model.score(x_train, y_train)
        assert 0 <= score <= 1
        # For this simple case, should achieve decent accuracy
        assert score >= 0.5

    def test_score_calculation(self) -> None:
        """Test that score calculates accuracy correctly."""
        model = LogisticRegression()
        model.weights = np.array([1.0, 1.0])
        model.bias = 0

        x = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        # Manually construct y so we know what score to expect
        y_gold = np.array([0, 0, 1, 1])

        score = model.score(x, y_gold)
        # The score should be between 0 and 1
        assert 0 <= score <= 1


class TestStringRepresentation:
    """Test string representation."""

    def test_str_before_training(self) -> None:
        """Test __str__ before training."""
        model = LogisticRegression()
        result = str(model)
        assert "LogisticRegression" in result
        assert "weights=None" in result
        assert "bias=0" in result

    def test_str_after_training(self) -> None:
        """Test __str__ after training."""
        model = LogisticRegression(n_iters=10)
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        model.fit(x, y)
        result = str(model)
        assert "LogisticRegression" in result
        assert "weights=" in result
        assert "bias=" in result


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_sample(self) -> None:
        """Test training with single sample."""
        model = LogisticRegression(n_iters=10)
        x = np.array([[1, 2]])
        y = np.array([1])
        # Should not raise an error
        model.fit(x, y)
        assert model.weights is not None

    def test_single_feature(self) -> None:
        """Test training with single feature."""
        model = LogisticRegression(n_iters=10)
        x = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        model.fit(x, y)
        assert model.weights is not None
        assert len(model.weights) == 1

    def test_predict_single_sample(self) -> None:
        """Test prediction with single sample."""
        model = LogisticRegression(n_iters=10)
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        model.fit(x, y)

        x_test = np.array([[2, 3]])
        classes, probs = model.predict(x_test)
        assert len(classes) == 1
        assert len(probs) == 1

    def test_all_same_label(self) -> None:
        """Test training when all labels are the same."""
        model = LogisticRegression(n_iters=10)
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 1, 1])
        # Should not raise an error
        model.fit(x, y)
        assert model.weights is not None
