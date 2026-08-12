"""Tests for lesson 01. Run with:  python3 -m pytest 01-linear-regression -q"""

import numpy as np
import pytest

from linear_regression import (
    LinearRegression,
    compute_cost,
    compute_gradient,
    gradient_descent,
    normal_equation,
    predict,
    zscore_normalize,
)


@pytest.fixture
def toy():
    """y = 2x + 1 exactly, so the optimum is w=[2], b=1 with zero cost."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])
    return X, y


def test_predict_matches_hand_calculation(toy):
    X, _ = toy
    assert np.allclose(predict(X, np.array([2.0]), 1.0), [3, 5, 7, 9])


def test_cost_is_zero_at_the_optimum(toy):
    X, y = toy
    assert compute_cost(X, y, np.array([2.0]), 1.0) == pytest.approx(0.0)


def test_cost_matches_the_formula_by_hand(toy):
    X, y = toy
    # w=0, b=0 -> errors are -3,-5,-7,-9 -> sum sq = 164 -> /(2*4) = 20.5
    assert compute_cost(X, y, np.array([0.0]), 0.0) == pytest.approx(20.5)


def test_gradient_is_zero_at_the_optimum(toy):
    X, y = toy
    dj_dw, dj_db = compute_gradient(X, y, np.array([2.0]), 1.0)
    assert np.allclose(dj_dw, 0.0)
    assert dj_db == pytest.approx(0.0)


def test_analytic_gradient_matches_numerical_gradient():
    """The check that catches almost every derivative mistake."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=20)
    w = rng.normal(size=3)
    b = 0.7
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


def test_gradient_descent_converges_to_the_closed_form_solution():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 5, size=(200, 2))
    y = X @ np.array([1.5, -2.0]) + 4.0 + rng.normal(0, 0.1, size=200)

    w, b, history = gradient_descent(X, y, np.zeros(2), 0.0, alpha=0.02, num_iters=20_000)
    w_exact, b_exact = normal_equation(X, y)

    assert np.allclose(w, w_exact, atol=1e-3)
    assert b == pytest.approx(b_exact, abs=1e-3)
    # Cost must decrease monotonically for a well-chosen learning rate.
    assert np.all(np.diff(history["cost"]) <= 1e-12)


def test_zscore_normalize_gives_zero_mean_unit_std():
    rng = np.random.default_rng(3)
    X = rng.normal(10, 4, size=(50, 2))
    Xn, mu, sigma = zscore_normalize(X)
    assert np.allclose(Xn.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Xn.std(axis=0), 1.0)
    # Reusing training statistics reproduces the same transform.
    assert np.allclose(zscore_normalize(X, mu, sigma)[0], Xn)


def test_normalization_does_not_change_the_predictions():
    rng = np.random.default_rng(4)
    X = np.column_stack([rng.uniform(0, 1, 100), rng.uniform(0, 1000, 100)])
    y = X @ np.array([3.0, 0.05]) + 1.0

    scaled = LinearRegression(alpha=0.1, num_iters=5000, normalize=True).fit(X, y)
    assert scaled.score(X, y) > 0.999


def test_fit_accepts_a_1d_feature_array(toy):
    X, y = toy
    model = LinearRegression(alpha=0.05, num_iters=5000).fit(X.ravel(), y)
    assert model.predict(np.array([5.0]))[0] == pytest.approx(11.0, abs=1e-2)
