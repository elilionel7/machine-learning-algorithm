"""Regularisation from scratch with NumPy: ridge (L2) and lasso (L1).

A model flexible enough to fit any training set will fit the noise in it. Regularisation
adds a penalty on the size of the weights, so the optimiser has to trade fitting the data
against keeping the weights small.

Ridge, also called L2 regularisation, penalises the squared length of w

    J(w, b) = (1 / 2m) * [ sum_i (f(x^(i)) - y^(i))^2 + lam * sum_j w_j^2 ]

Lasso, also called L1 regularisation, penalises the sum of absolute values

    J(w, b) = (1 / 2m) * sum_i (f(x^(i)) - y^(i))^2 + (lam / m) * sum_j |w_j|

Regularised logistic regression uses the same L2 penalty on top of cross-entropy

    J(w, b) = (1 / m) * sum_i loss_i + (lam / 2m) * sum_j w_j^2

The bias b is never penalised. It sets the overall level of the predictions and shrinking
it towards zero would bias every prediction towards zero for no benefit.

Shapes: X is (m, n), y is (m,), w is (n,), b is a scalar, lam is a non negative float.
"""

import numpy as np


# ----------------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------------

def zscore_normalize(X, mu=None, sigma=None):
    """Scale each feature to zero mean and unit standard deviation.

    Mandatory when regularising. The penalty treats every weight equally, so a feature
    measured in thousands and a feature measured in units would be penalised very
    differently for no principled reason.
    """
    if mu is None:
        mu = X.mean(axis=0)
    if sigma is None:
        sigma = X.std(axis=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma


def polynomial_features(x, degree):
    """Build [x, x^2, ..., x^degree] from a 1D array of length m.

    Returns an array of shape (m, degree). Column j holds x raised to the power j + 1.
    """
    x = np.asarray(x, dtype=float).ravel()
    return np.column_stack([x ** p for p in range(1, degree + 1)])


def train_test_split(X, y, test_fraction=0.3, seed=0):
    """Shuffle the rows, then cut them into a training part and a test part."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[0])
    n_test = int(round(test_fraction * X.shape[0]))
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def k_fold_indices(m, k=5, seed=0):
    """Yield (train_idx, validation_idx) pairs for k-fold cross validation."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(m)
    folds = np.array_split(order, k)
    for i in range(k):
        validation_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, validation_idx


def sigmoid(z):
    """Map any real number into (0, 1), computed so large inputs cannot overflow."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


# ----------------------------------------------------------------------------------
# linear regression with an L2 penalty
# ----------------------------------------------------------------------------------

def compute_cost_linear(X, y, w, b, lam=0.0):
    """Mean squared error plus the ridge penalty. Returns a Python float."""
    m = X.shape[0]
    error = X @ w + b - y
    data_term = error @ error
    penalty = lam * (w @ w)          # b is deliberately absent from this line
    return float((data_term + penalty) / (2 * m))


def compute_gradient_linear(X, y, w, b, lam=0.0):
    """Gradient of the ridge cost. Returns (dj_dw of shape (n,), dj_db float).

    Differentiating (lam / 2m) * w . w gives (lam / m) * w, so the penalty adds a term
    proportional to w itself. That is why L2 is also called weight decay: every step
    pulls each weight a little towards zero.
    """
    m = X.shape[0]
    error = X @ w + b - y
    dj_dw = (X.T @ error + lam * w) / m
    dj_db = float(error.sum() / m)   # no penalty term here
    return dj_dw, dj_db


def ridge_normal_equation(X, y, lam=0.0):
    """Closed form ridge solution. Returns (w of shape (n,), b float).

    Solves (X_b^T X_b + lam * P) theta = X_b^T y, where X_b is X with a leading column of
    ones and P is the identity with its first diagonal entry zeroed so the bias escapes
    the penalty.

    A useful side effect: lam > 0 makes the matrix invertible even when X has fewer rows
    than columns, or has perfectly correlated features. Plain least squares fails there.
    """
    m, n = X.shape
    X_b = np.hstack([np.ones((m, 1)), X])
    penalty = np.eye(n + 1)
    penalty[0, 0] = 0.0
    theta = np.linalg.solve(X_b.T @ X_b + lam * penalty, X_b.T @ y)
    return theta[1:], float(theta[0])


# ----------------------------------------------------------------------------------
# logistic regression with an L2 penalty
# ----------------------------------------------------------------------------------

def compute_cost_logistic(X, y, w, b, lam=0.0):
    """Cross-entropy plus the ridge penalty, computed from the logits for stability."""
    m = X.shape[0]
    z = X @ w + b
    data_term = np.mean(np.logaddexp(0.0, z) - y * z)
    return float(data_term + lam * (w @ w) / (2 * m))


def compute_gradient_logistic(X, y, w, b, lam=0.0):
    """Gradient of the regularised cross-entropy cost."""
    m = X.shape[0]
    error = sigmoid(X @ w + b) - y
    dj_dw = (X.T @ error + lam * w) / m
    dj_db = float(error.sum() / m)
    return dj_dw, dj_db


# ----------------------------------------------------------------------------------
# gradient descent, shared by both models
# ----------------------------------------------------------------------------------

def gradient_descent(X, y, w, b, alpha, num_iters, grad_fn, cost_fn, lam=0.0,
                     record_every=1):
    """Batch gradient descent for any (cost, gradient) pair that takes lam.

    Returns the learned w and b, plus a history dict with "iter" and "cost" lists.

    Note on stability: the penalty contributes a term (lam / m) * w to the gradient, so
    the weight update contains the factor (1 - alpha * lam / m). That factor must stay
    inside (-1, 1), which requires

        alpha < 2 * m / lam

    Raising lam without lowering alpha is a common way to make a run diverge.
    """
    w = np.asarray(w, dtype=float).copy()
    b = float(b)
    history = {"iter": [], "cost": []}

    for i in range(num_iters):
        dj_dw, dj_db = grad_fn(X, y, w, b, lam)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % record_every == 0 or i == num_iters - 1:
            history["iter"].append(i)
            history["cost"].append(cost_fn(X, y, w, b, lam))

    return w, b, history


# ----------------------------------------------------------------------------------
# lasso, solved by proximal gradient descent
# ----------------------------------------------------------------------------------

def soft_threshold(x, t):
    """Shrink each entry of x towards zero by t, clamping at exactly zero.

        soft_threshold(x, t) = sign(x) * max(|x| - t, 0)

    This is the proximal operator of the L1 norm, and it is the reason lasso produces
    exact zeros while ridge only produces small numbers.
    """
    # adding 0.0 turns any -0.0 produced by sign(x) * 0.0 into +0.0, which only affects
    # how the result prints, not its value
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0) + 0.0


def compute_cost_lasso(X, y, w, b, lam=0.0):
    """Mean squared error plus the L1 penalty."""
    m = X.shape[0]
    error = X @ w + b - y
    return float(error @ error / (2 * m) + lam * np.abs(w).sum() / m)


def lasso_gradient_descent(X, y, w, b, alpha, num_iters, lam=0.0, record_every=1):
    """Proximal gradient descent, also known as ISTA.

    The absolute value has no derivative at zero, so plain gradient descent cannot reach
    an exact zero and would oscillate around it. Instead each iteration takes an ordinary
    gradient step on the squared error, then applies soft thresholding to handle the
    penalty exactly.
    """
    w = np.asarray(w, dtype=float).copy()
    b = float(b)
    m = X.shape[0]
    history = {"iter": [], "cost": []}

    for i in range(num_iters):
        error = X @ w + b - y
        w = w - alpha * (X.T @ error) / m          # step on the squared error only
        w = soft_threshold(w, alpha * lam / m)      # then handle the L1 penalty exactly
        b = b - alpha * float(error.sum() / m)

        if i % record_every == 0 or i == num_iters - 1:
            history["iter"].append(i)
            history["cost"].append(compute_cost_lasso(X, y, w, b, lam))

    return w, b, history


# ----------------------------------------------------------------------------------
# model classes
# ----------------------------------------------------------------------------------

class _BaseModel:
    """Shared fitting plumbing. Subclasses supply the cost, gradient and solver."""

    def __init__(self, lam=1.0, alpha=0.1, num_iters=3000, normalize=True):
        self.lam = lam
        self.alpha = alpha
        self.num_iters = num_iters
        self.normalize = normalize
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


class RidgeRegression(_BaseModel):
    """Linear regression with an L2 penalty."""

    def fit(self, X, y):
        X = self._prepare(X, fitting=True)
        y = np.asarray(y, dtype=float)
        self.w, self.b, self.history = gradient_descent(
            X, y, np.zeros(X.shape[1]), 0.0, self.alpha, self.num_iters,
            compute_gradient_linear, compute_cost_linear, self.lam,
        )
        return self

    def predict(self, X):
        return self._prepare(X) @ self.w + self.b

    def score(self, X, y):
        """R squared. 1 is perfect, 0 is no better than predicting the mean."""
        y = np.asarray(y, dtype=float)
        residual = np.sum((y - self.predict(X)) ** 2)
        total = np.sum((y - y.mean()) ** 2)
        return float(1 - residual / total)

    def mse(self, X, y):
        return float(np.mean((np.asarray(y, dtype=float) - self.predict(X)) ** 2))


class LassoRegression(RidgeRegression):
    """Linear regression with an L1 penalty, solved by proximal gradient descent."""

    def fit(self, X, y):
        X = self._prepare(X, fitting=True)
        y = np.asarray(y, dtype=float)
        self.w, self.b, self.history = lasso_gradient_descent(
            X, y, np.zeros(X.shape[1]), 0.0, self.alpha, self.num_iters, self.lam,
        )
        return self


class RegularizedLogisticRegression(_BaseModel):
    """Logistic regression with an L2 penalty."""

    def __init__(self, lam=1.0, alpha=0.1, num_iters=3000, normalize=True, threshold=0.5):
        super().__init__(lam, alpha, num_iters, normalize)
        self.threshold = threshold

    def fit(self, X, y):
        X = self._prepare(X, fitting=True)
        y = np.asarray(y, dtype=float)
        self.w, self.b, self.history = gradient_descent(
            X, y, np.zeros(X.shape[1]), 0.0, self.alpha, self.num_iters,
            compute_gradient_logistic, compute_cost_logistic, self.lam,
        )
        return self

    def predict_proba(self, X):
        return sigmoid(self._prepare(X) @ self.w + self.b)

    def predict(self, X, threshold=None):
        t = self.threshold if threshold is None else threshold
        return (self.predict_proba(X) >= t).astype(int)

    def score(self, X, y):
        return float(np.mean(np.asarray(y) == self.predict(X)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 40 noisy samples from a smooth curve, fitted with a degree 12 polynomial
    m = 40
    x = np.sort(rng.uniform(-1, 1, m))
    y = np.sin(2.5 * x) + rng.normal(0, 0.25, m)
    X_poly = polynomial_features(x, 12)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, 0.5, seed=2)

    # The closed form is used here so the sweep shows the effect of lambda alone, with no
    # confounding from how far gradient descent happened to get.
    X_train_n, mu, sigma = zscore_normalize(X_train)
    X_test_n, _, _ = zscore_normalize(X_test, mu, sigma)

    print("ridge, solved exactly by the normal equation")
    print(f"{'lambda':>10}{'train MSE':>12}{'test MSE':>12}{'||w||':>12}")
    for lam in (0.0, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        w, b = ridge_normal_equation(X_train_n, y_train, lam)
        train_mse = np.mean((X_train_n @ w + b - y_train) ** 2)
        test_mse = np.mean((X_test_n @ w + b - y_test) ** 2)
        print(f"{lam:>10.0e}{train_mse:>12.4f}{test_mse:>12.4f}{np.linalg.norm(w):>12.3f}")
    print("train error rises with lambda while test error dips, which is the whole point")

    print("\nlasso against ridge, counting weights that are exactly zero")
    print(f"{'lambda':>10}{'lasso kept':>13}{'ridge kept':>13}{'lasso test MSE':>17}")
    for lam in (0.01, 0.1, 1.0, 10.0):
        w_l, b_l, _ = lasso_gradient_descent(X_train_n, y_train, np.zeros(12), 0.0,
                                             0.02, 50_000, lam)
        w_r, _ = ridge_normal_equation(X_train_n, y_train, lam)
        test_mse = np.mean((X_test_n @ w_l + b_l - y_test) ** 2)
        print(f"{lam:>10}{int(np.sum(np.abs(w_l) > 1e-8)):>13}"
              f"{int(np.sum(np.abs(w_r) > 1e-8)):>13}{test_mse:>17.4f}")
    print("lasso drives weights to exactly zero, ridge keeps all 12 alive")
