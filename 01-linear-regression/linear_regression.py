"""Linear regression from scratch with NumPy.

Model
-----
    f(x) = w . x + b          w in R^n, b in R, x in R^n

Cost (mean squared error, halved so the 2 cancels when differentiating)

    J(w, b) = (1 / 2m) * sum_i (f(x^(i)) - y^(i))^2

Gradients

    dJ/dw_j = (1 / m) * sum_i (f(x^(i)) - y^(i)) * x_j^(i)
    dJ/db   = (1 / m) * sum_i (f(x^(i)) - y^(i))

Gradient descent updates every parameter *simultaneously*:

    w := w - alpha * dJ/dw
    b := b - alpha * dJ/db

Shapes used throughout: X is (m, n), y is (m,), w is (n,), b is a scalar.
A single feature is just n = 1, so nothing here is special-cased.
"""

import numpy as np


def predict(X, w, b):
    """Model output f(x) = X @ w + b for every row of X. Returns shape (m,)."""
    return X @ w + b


def compute_cost(X, y, w, b):
    """Mean squared error cost J(w, b). Returns a Python float."""
    m = X.shape[0]
    error = predict(X, w, b) - y          # (m,)
    return float(error @ error / (2 * m))  # error @ error == sum of squares


def compute_gradient(X, y, w, b):
    """Partial derivatives of J. Returns (dj_dw of shape (n,), dj_db float)."""
    m = X.shape[0]
    error = predict(X, w, b) - y          # (m,)
    dj_dw = (X.T @ error) / m             # (n, m) @ (m,) -> (n,)
    dj_db = float(error.sum() / m)
    return dj_dw, dj_db


def gradient_descent(X, y, w, b, alpha, num_iters, record_every=1):
    """Run batch gradient descent.

    Returns
    -------
    w, b : the learned parameters
    history : dict with "cost" and "iter" lists, for plotting the learning curve
    """
    w = np.asarray(w, dtype=float).copy()
    b = float(b)
    history = {"iter": [], "cost": []}

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        # Simultaneous update: both use the gradient at the *old* parameters.
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % record_every == 0 or i == num_iters - 1:
            history["iter"].append(i)
            history["cost"].append(compute_cost(X, y, w, b))

    return w, b, history


def zscore_normalize(X, mu=None, sigma=None):
    """Scale each feature to zero mean and unit standard deviation.

    Pass the training mu/sigma back in when transforming new data, so test
    points land in the same coordinate system the model was fitted in.
    """
    if mu is None:
        mu = X.mean(axis=0)
    if sigma is None:
        sigma = X.std(axis=0)
        sigma = np.where(sigma == 0, 1.0, sigma)  # constant feature -> leave as is
    return (X - mu) / sigma, mu, sigma


def normal_equation(X, y):
    """Closed-form solution, used only to check gradient descent's answer.

    Solves (X_b^T X_b) theta = X_b^T y where X_b is X with a column of ones.
    """
    m = X.shape[0]
    X_b = np.hstack([np.ones((m, 1)), X])
    theta = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)
    return theta[1:], float(theta[0])  # w, b


class LinearRegression:
    """Small sklearn-shaped wrapper around the functions above."""

    def __init__(self, alpha=0.01, num_iters=1000, normalize=False):
        self.alpha = alpha
        self.num_iters = num_iters
        self.normalize = normalize
        self.w = None
        self.b = None
        self.mu = None
        self.sigma = None
        self.history = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.normalize:
            X, self.mu, self.sigma = zscore_normalize(X)

        w0 = np.zeros(X.shape[1])
        self.w, self.b, self.history = gradient_descent(
            X, y, w0, 0.0, self.alpha, self.num_iters
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.normalize:
            X, _, _ = zscore_normalize(X, self.mu, self.sigma)
        return predict(X, self.w, self.b)

    def score(self, X, y):
        """R^2: 1 means perfect, 0 means no better than predicting the mean."""
        y = np.asarray(y, dtype=float)
        ss_res = np.sum((y - self.predict(X)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / ss_tot)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m = 100
    X = rng.uniform(0, 10, size=(m, 1))
    y = 3.5 * X[:, 0] - 2.0 + rng.normal(0, 1.0, size=m)  # true w=3.5, b=-2

    model = LinearRegression(alpha=0.01, num_iters=5000).fit(X, y)
    w_exact, b_exact = normal_equation(X, y)

    print(f"gradient descent : w={model.w[0]:.4f}  b={model.b:.4f}")
    print(f"normal equation  : w={w_exact[0]:.4f}  b={b_exact:.4f}")
    print(f"final cost       : {compute_cost(X, y, model.w, model.b):.4f}")
    print(f"R^2              : {model.score(X, y):.4f}")
