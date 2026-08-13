"""Logistic regression from scratch with NumPy.

Binary classification. The model outputs a probability, not a number on a line.

Model
-----
    z = w . x + b                 the linear score, also called the logit
    f(x) = sigmoid(z)             a probability in (0, 1)

    sigmoid(z) = 1 / (1 + exp(-z))

Cost (binary cross-entropy, also called log loss)

    J(w, b) = (1 / m) * sum_i [ -y^(i) * log(f(x^(i))) - (1 - y^(i)) * log(1 - f(x^(i))) ]

Gradients (identical in form to linear regression, which is not a coincidence)

    dJ/dw_j = (1 / m) * sum_i (f(x^(i)) - y^(i)) * x_j^(i)
    dJ/db   = (1 / m) * sum_i (f(x^(i)) - y^(i))

Shapes: X is (m, n), y is (m,) holding 0 or 1, w is (n,), b is a scalar.
"""

import numpy as np


def sigmoid(z):
    """Map any real number into (0, 1), computed so large inputs cannot overflow.

    For z very negative, exp(-z) overflows to inf. For z very positive, exp(z)
    overflows. Each branch below uses only the exponential that stays small.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def predict_proba(X, w, b):
    """Predicted probability that y = 1, for every row of X. Returns shape (m,)."""
    return sigmoid(X @ w + b)


def compute_cost(X, y, w, b):
    """Binary cross-entropy, computed from the logits so it never returns inf or nan.

    Writing the loss in terms of z instead of sigmoid(z) avoids taking log(0):

        loss = log(1 + exp(z)) - y * z = logaddexp(0, z) - y * z

    np.logaddexp(0, z) computes log(1 + exp(z)) without overflowing.
    """
    z = X @ w + b
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def compute_gradient(X, y, w, b):
    """Partial derivatives of J. Returns (dj_dw of shape (n,), dj_db float)."""
    m = X.shape[0]
    error = predict_proba(X, w, b) - y        # (m,), the probability minus the label
    dj_dw = (X.T @ error) / m                 # (n, m) times (m,) gives (n,)
    dj_db = float(error.sum() / m)
    return dj_dw, dj_db


def gradient_descent(X, y, w, b, alpha, num_iters, record_every=1):
    """Run batch gradient descent.

    Returns the learned w and b, plus a history dict with "iter" and "cost" lists.
    """
    w = np.asarray(w, dtype=float).copy()
    b = float(b)
    history = {"iter": [], "cost": []}

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        # Simultaneous update: both use the gradient at the current parameters.
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % record_every == 0 or i == num_iters - 1:
            history["iter"].append(i)
            history["cost"].append(compute_cost(X, y, w, b))

    return w, b, history


def zscore_normalize(X, mu=None, sigma=None):
    """Scale each feature to zero mean and unit standard deviation.

    Pass the training mu and sigma back in when transforming validation or test data,
    so those points land in the same coordinate system the model was fitted in.
    """
    if mu is None:
        mu = X.mean(axis=0)
    if sigma is None:
        sigma = X.std(axis=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma


def accuracy(y_true, y_pred):
    """Fraction of labels predicted correctly."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred):
    """Counts as a 2 by 2 integer array, laid out as [[TN, FP], [FN, TP]].

    Row index is the true label, column index is the predicted label.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t in (0, 1):
        for p in (0, 1):
            cm[t, p] = np.sum((y_true == t) & (y_pred == p))
    return cm


def precision_recall_f1(y_true, y_pred):
    """Precision, recall and F1 for the positive class. Returns three floats."""
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1


class LogisticRegression:
    """Small sklearn-shaped wrapper around the functions above."""

    def __init__(self, alpha=0.1, num_iters=2000, normalize=False, threshold=0.5):
        self.alpha = alpha
        self.num_iters = num_iters
        self.normalize = normalize
        self.threshold = threshold
        self.w = None
        self.b = None
        self.mu = None
        self.sigma = None
        self.history = None

    def _prepare(self, X, fitting=False):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.normalize:
            if fitting:
                X, self.mu, self.sigma = zscore_normalize(X)
            else:
                X, _, _ = zscore_normalize(X, self.mu, self.sigma)
        return X

    def fit(self, X, y):
        X = self._prepare(X, fitting=True)
        y = np.asarray(y, dtype=float)
        self.w, self.b, self.history = gradient_descent(
            X, y, np.zeros(X.shape[1]), 0.0, self.alpha, self.num_iters
        )
        return self

    def predict_proba(self, X):
        """Probability that y = 1 for each row. Returns shape (m,)."""
        return predict_proba(self._prepare(X), self.w, self.b)

    def predict(self, X, threshold=None):
        """Hard 0 or 1 labels, obtained by thresholding the probabilities."""
        t = self.threshold if threshold is None else threshold
        return (self.predict_proba(X) >= t).astype(int)

    def score(self, X, y):
        """Classification accuracy."""
        return accuracy(y, self.predict(X))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m = 200
    # Two clusters of points, one per class.
    X = np.vstack([
        rng.normal([-1.5, -1.0], 1.0, size=(m // 2, 2)),
        rng.normal([1.5, 1.0], 1.0, size=(m // 2, 2)),
    ])
    y = np.concatenate([np.zeros(m // 2), np.ones(m // 2)])

    model = LogisticRegression(alpha=0.5, num_iters=3000).fit(X, y)
    y_pred = model.predict(X)
    precision, recall, f1 = precision_recall_f1(y, y_pred)

    print(f"learned w = {model.w.round(3)}, b = {model.b:.3f}")
    print(f"final cost = {model.history['cost'][-1]:.4f}")
    print(f"accuracy   = {model.score(X, y):.4f}")
    print(f"precision  = {precision:.4f}   recall = {recall:.4f}   F1 = {f1:.4f}")
    print("confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y, y_pred))
