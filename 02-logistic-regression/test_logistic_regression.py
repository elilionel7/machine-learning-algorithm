"""Tests for lesson 02. Run with:  python3 -m pytest 02-logistic-regression -q"""

import numpy as np
import pytest

from logistic_regression import (
    LogisticRegression,
    accuracy,
    compute_cost,
    compute_gradient,
    confusion_matrix,
    gradient_descent,
    precision_recall_f1,
    predict_proba,
    sigmoid,
)


@pytest.fixture
def separable():
    """Two clearly separated clusters, so a good model reaches perfect accuracy."""
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal([-3.0, -3.0], 0.6, size=(60, 2)),
        rng.normal([3.0, 3.0], 0.6, size=(60, 2)),
    ])
    y = np.concatenate([np.zeros(60), np.ones(60)])
    return X, y


def test_sigmoid_known_values():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(np.array([0.0, 2.0]))[1] == pytest.approx(0.8807970779778823)


def test_sigmoid_is_symmetric_about_zero():
    z = np.array([-4.0, -0.5, 0.0, 0.5, 4.0])
    assert np.allclose(sigmoid(-z), 1.0 - sigmoid(z))


def test_sigmoid_is_monotonic_and_bounded():
    z = np.linspace(-20, 20, 200)
    s = sigmoid(z)
    assert np.all(np.diff(s) > 0)
    assert np.all((s > 0.0) & (s < 1.0))


def test_sigmoid_does_not_overflow_on_extreme_inputs():
    """The naive 1/(1+exp(-z)) overflows here and emits a RuntimeWarning.

    Underflow is allowed: exp(-1000) becoming exactly 0.0 is the correct answer to
    working precision. Only overflow and invalid operations signal a real problem.
    """
    with np.errstate(over="raise", invalid="raise"):
        s = sigmoid(np.array([-1000.0, -50.0, 50.0, 1000.0]))
    assert np.all(np.isfinite(s))
    assert s[0] == pytest.approx(0.0)
    assert s[-1] == pytest.approx(1.0)


def test_cost_is_near_zero_when_predictions_are_confident_and_right():
    X = np.array([[10.0], [-10.0]])
    y = np.array([1.0, 0.0])
    assert compute_cost(X, y, np.array([1.0]), 0.0) < 1e-4


def test_cost_stays_finite_when_predictions_are_confident_and_wrong():
    """sigmoid saturates to exactly 1.0, so a naive log(1 - f) would give inf."""
    X = np.array([[100.0]])
    y = np.array([0.0])
    cost = compute_cost(X, y, np.array([1.0]), 0.0)
    assert np.isfinite(cost)
    assert cost == pytest.approx(100.0, rel=1e-6)  # loss approaches z for a confident miss


def test_cost_of_a_chance_level_model():
    """With w = 0 and b = 0 every probability is 0.5, so the cost is log(2)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = (rng.random(30) > 0.5).astype(float)
    assert compute_cost(X, y, np.zeros(3), 0.0) == pytest.approx(np.log(2))


def test_analytic_gradient_matches_numerical_gradient():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(25, 3))
    y = (rng.random(25) > 0.5).astype(float)
    w = rng.normal(size=3)
    b = 0.3
    eps = 1e-6

    dj_dw, dj_db = compute_gradient(X, y, w, b)

    for j in range(3):
        w_up, w_dn = w.copy(), w.copy()
        w_up[j] += eps
        w_dn[j] -= eps
        numeric = (compute_cost(X, y, w_up, b) - compute_cost(X, y, w_dn, b)) / (2 * eps)
        assert dj_dw[j] == pytest.approx(numeric, abs=1e-6)

    numeric_b = (compute_cost(X, y, w, b + eps) - compute_cost(X, y, w, b - eps)) / (2 * eps)
    assert dj_db == pytest.approx(numeric_b, abs=1e-6)


def test_gradient_descent_decreases_cost_and_classifies_correctly(separable):
    X, y = separable
    w, b, history = gradient_descent(X, y, np.zeros(2), 0.0, alpha=0.1, num_iters=2000)

    assert np.all(np.diff(history["cost"]) <= 1e-12)
    predictions = (predict_proba(X, w, b) >= 0.5).astype(int)
    assert accuracy(y, predictions) == 1.0


def test_gradient_descent_does_not_modify_the_caller_arrays(separable):
    X, y = separable
    w0 = np.zeros(2)
    gradient_descent(X, y, w0, 0.0, alpha=0.1, num_iters=50)
    assert np.array_equal(w0, np.zeros(2))


def test_confusion_matrix_layout():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 1])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.tolist() == [[2, 1], [1, 3]]  # [[TN, FP], [FN, TP]]
    assert cm.sum() == len(y_true)


def test_precision_recall_f1_against_hand_calculation():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 1])
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    assert precision == pytest.approx(3 / 4)   # 3 true positives, 1 false positive
    assert recall == pytest.approx(3 / 4)      # 3 true positives, 1 false negative
    assert f1 == pytest.approx(3 / 4)


def test_threshold_trades_precision_against_recall(separable):
    """Raising the threshold makes the model more reluctant to predict class 1."""
    X, y = separable
    model = LogisticRegression(alpha=0.1, num_iters=500).fit(X, y)
    n_positive_low = model.predict(X, threshold=0.1).sum()
    n_positive_high = model.predict(X, threshold=0.9).sum()
    assert n_positive_low >= n_positive_high


def test_model_fits_a_one_dimensional_feature():
    X = np.array([-4.0, -3.0, -2.0, 2.0, 3.0, 4.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(alpha=0.5, num_iters=3000).fit(X, y)
    assert model.score(X, y) == 1.0
    assert model.predict_proba(np.array([5.0]))[0] > 0.5


def test_normalization_reuses_training_statistics():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(0, 1, 100), rng.normal(500, 200, 100)])
    y = (X[:, 0] + X[:, 1] / 200 > 2.5).astype(float)
    model = LogisticRegression(alpha=0.5, num_iters=2000, normalize=True).fit(X, y)
    assert model.score(X, y) > 0.9
    # The stored statistics come from the training data, not from whatever we predict on.
    assert np.allclose(model.mu, X.mean(axis=0))
    assert np.allclose(model.sigma, X.std(axis=0))
