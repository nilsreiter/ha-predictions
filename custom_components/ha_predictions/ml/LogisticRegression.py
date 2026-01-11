from types import NoneType

import numpy as np


# TODO: Add multi-class support
# TODO: Add regularization
# TODO: Add better convergence checks
# TODO: Optimize performance with vectorized operations, but without introducing additional dependencies
# TODO: Deal with missing data (gracefully handle NaNs)
class LogisticRegression:
    weights: np.ndarray | NoneType = None
    bias: float = 0

    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.losses: list[float] = []

    # Sigmoid method
    def _sigmoid(self, x: float) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # binary cross entropy
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        return float(-np.mean(y1 + y2))

    def _feed_forward(self, x: np.ndarray):
        if self.weights is not None:
            z = np.dot(x, self.weights) + self.bias
            return self._sigmoid(z)
        return None

    def fit(self, x: np.ndarray, y: np.ndarray) -> NoneType:
        n_samples, n_features = x.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            A = self._feed_forward(x)
            self.losses.append(self._compute_loss(y, A))  # type: ignore
            dz = A - y  # derivative of sigmoid and bce X.T*(A-y)
            # compute gradients
            dw = (1 / n_samples) * np.dot(x.T, dz)
            db = (1 / n_samples) * np.sum(dz)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | tuple[NoneType, NoneType]:
        """
        Make predictions and return classes and probabilities.

        Returns:
            tuple: (predicted_classes, probabilities) or None

        """
        if self.weights is not None:
            threshold = 0.5
            y_hat = np.dot(x, self.weights) + self.bias
            y_predicted = self._sigmoid(y_hat)  # Probabilities
            y_predicted_cls = np.array([1 if i > threshold else 0 for i in y_predicted])

            return y_predicted_cls, y_predicted
        return (None, None)

    def score(self, x: np.ndarray, y_gold: np.ndarray) -> float:
        y_pred_classes, _ = self.predict(x)
        matches = (y_gold == y_pred_classes).sum()
        total = len(y_gold)
        return matches / total

    def __str__(self) -> str:
        """Generate string representation of the model."""
        return f"LogisticRegression(weights={self.weights},bias={self.bias})"
