"""Tests for lesson 03. Run with:  python3 -m pytest 03-regularization -q"""

import numpy as np
import pytest

from regularization import (
    LassoRegression,
    RegularizedLogisticRegression,
    RidgeRegression,
    compute_cost_linear,
    compute_cost_lasso,
    compute_cost_logistic,
    compute_gradient_linear,
    compute_gradient_logistic,
    gradient_descent,
    k_fold_indices,
    lasso_gradient_descent,
    polynomial_features,
    ridge_normal_equation,
    soft_threshold,
    train_test_split,
    zscore_normalize,
)


@pytest.fixture
def noisy_curve():
    """40 noisy samples of a smooth curve, expanded to degree 8 polynomial features."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-1, 1, 40))
    y = np.sin(2.5 * x) + rng.normal(0, 0.25, 40)
    X, _, _ = zscore_normalize(polynomial_features(x, 8))
    return X, y


# ---------------------------------------------------------------- helpers

def test_polynomial_features_shape_and_values():
    X = polynomial_features(np.array([2.0, 3.0]), 3)
    assert X.shape == (2, 3)
    assert np.allclose(X, [[2, 4, 8], [3, 9, 27]])


def test_train_test_split_is_a_partition():
    X = np.arange(40).reshape(20, 2).astype(float)
    y = np.arange(20).astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_fraction=0.25, seed=0)
    assert len(y_tr) == 15 and len(y_te) == 5
    assert sorted(np.concatenate([y_tr, y_te])) == list(range(20))


def test_k_fold_indices_uses_every_row_for_validation_exactly_once():
    seen = np.concatenate([val for _, val in k_fold_indices(23, k=5, seed=0)])
    assert sorted(seen) == list(range(23))
    for train_idx, val_idx in k_fold_indices(23, k=5, seed=0):
        assert set(train_idx).isdisjoint(set(val_idx))
        assert len(train_idx) + len(val_idx) == 23


# ---------------------------------------------------------------- the penalty term

def test_cost_with_lam_zero_is_plain_mean_squared_error():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    w, b = np.array([1.5]), 0.5
    error = X @ w + b - y
    expected = float(error @ error / (2 * len(y)))
    assert compute_cost_linear(X, y, w, b, lam=0.0) == pytest.approx(expected)


def test_penalty_adds_the_expected_amount():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    w, b, lam = np.array([2.0]), 0.5, 3.0
    plain = compute_cost_linear(X, y, w, b, lam=0.0)
    penalised = compute_cost_linear(X, y, w, b, lam=lam)
    assert penalised - plain == pytest.approx(lam * (w @ w) / (2 * len(y)))


def test_bias_is_not_penalised():
    """Changing b alone must change the cost only through the data term."""
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    w = np.array([1.0])
    for lam in (0.0, 5.0, 100.0):
        gap_plain = (compute_cost_linear(X, y, w, 1.0, 0.0)
                     - compute_cost_linear(X, y, w, 0.0, 0.0))
        gap_penalised = (compute_cost_linear(X, y, w, 1.0, lam)
                         - compute_cost_linear(X, y, w, 0.0, lam))
        assert gap_plain == pytest.approx(gap_penalised)


def test_huge_lambda_drives_weights_to_zero_and_bias_to_the_mean(noisy_curve):
    X, y = noisy_curve
    w, b = ridge_normal_equation(X, y, lam=1e12)
    assert np.allclose(w, 0.0, atol=1e-6)
    assert b == pytest.approx(y.mean(), abs=1e-6)


# ---------------------------------------------------------------- gradients

def test_linear_gradient_matches_numerical_gradient_with_penalty():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 4))
    y = rng.normal(size=20)
    w, b, lam, eps = rng.normal(size=4), 0.4, 7.0, 1e-6

    dj_dw, dj_db = compute_gradient_linear(X, y, w, b, lam)
    for j in range(4):
        step = np.zeros(4)
        step[j] = eps
        numeric = (compute_cost_linear(X, y, w + step, b, lam)
                   - compute_cost_linear(X, y, w - step, b, lam)) / (2 * eps)
        assert dj_dw[j] == pytest.approx(numeric, abs=1e-6)

    numeric_b = (compute_cost_linear(X, y, w, b + eps, lam)
                 - compute_cost_linear(X, y, w, b - eps, lam)) / (2 * eps)
    assert dj_db == pytest.approx(numeric_b, abs=1e-6)


def test_logistic_gradient_matches_numerical_gradient_with_penalty():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(25, 3))
    y = (rng.random(25) > 0.5).astype(float)
    w, b, lam, eps = rng.normal(size=3), 0.2, 4.0, 1e-6

    dj_dw, dj_db = compute_gradient_logistic(X, y, w, b, lam)
    for j in range(3):
        step = np.zeros(3)
        step[j] = eps
        numeric = (compute_cost_logistic(X, y, w + step, b, lam)
                   - compute_cost_logistic(X, y, w - step, b, lam)) / (2 * eps)
        assert dj_dw[j] == pytest.approx(numeric, abs=1e-6)

    numeric_b = (compute_cost_logistic(X, y, w, b + eps, lam)
                 - compute_cost_logistic(X, y, w, b - eps, lam)) / (2 * eps)
    assert dj_db == pytest.approx(numeric_b, abs=1e-6)


def test_bias_gradient_carries_no_penalty_term():
    """dj_db must be identical whatever lam is."""
    rng = np.random.default_rng(3)
    X, y, w, b = rng.normal(size=(15, 2)), rng.normal(size=15), rng.normal(size=2), 0.6
    gradients = [compute_gradient_linear(X, y, w, b, lam)[1] for lam in (0.0, 1.0, 50.0)]
    assert gradients[0] == pytest.approx(gradients[1]) == pytest.approx(gradients[2])


# ---------------------------------------------------------------- ridge behaviour

def test_gradient_descent_reaches_the_closed_form_solution():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 3.0 + rng.normal(0, 0.2, 60)
    lam = 2.0

    w_gd, b_gd, _ = gradient_descent(X, y, np.zeros(3), 0.0, 0.1, 20_000,
                                     compute_gradient_linear, compute_cost_linear, lam)
    w_exact, b_exact = ridge_normal_equation(X, y, lam)
    assert np.allclose(w_gd, w_exact, atol=1e-6)
    assert b_gd == pytest.approx(b_exact, abs=1e-6)


def test_increasing_lambda_shrinks_the_weights(noisy_curve):
    X, y = noisy_curve
    norms = [np.linalg.norm(ridge_normal_equation(X, y, lam)[0])
             for lam in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)]
    assert np.all(np.diff(norms) < 0)


def test_penalty_makes_an_unsolvable_system_well_conditioned():
    """With more features than examples the lam = 0 system is singular.

    numpy does not raise on it, because rounding leaves the matrix technically
    invertible, so it silently returns a meaningless answer. The condition number is
    what exposes the problem: above about 1e16 a float64 solve has no correct digits
    left. Adding lam puts the system back in a range where the answer means something.
    """
    rng = np.random.default_rng(5)
    X = rng.normal(size=(5, 12))          # 5 rows, 12 columns, so rank is at most 5
    y = rng.normal(size=5)
    X_b = np.hstack([np.ones((5, 1)), X])
    penalty = np.eye(13)
    penalty[0, 0] = 0.0

    condition_unpenalised = np.linalg.cond(X_b.T @ X_b)
    condition_penalised = np.linalg.cond(X_b.T @ X_b + 1.0 * penalty)

    assert condition_unpenalised > 1e16      # no usable precision remains
    assert condition_penalised < 1e3         # comfortably solvable
    w, b = ridge_normal_equation(X, y, lam=1.0)
    assert np.all(np.isfinite(w)) and np.isfinite(b)


def test_regularised_cost_decreases_monotonically(noisy_curve):
    X, y = noisy_curve
    _, _, history = gradient_descent(X, y, np.zeros(X.shape[1]), 0.0, 0.05, 3000,
                                     compute_gradient_linear, compute_cost_linear, lam=1.0)
    assert np.all(np.diff(history["cost"]) <= 1e-12)


# ---------------------------------------------------------------- lasso behaviour

def test_soft_threshold_against_hand_values():
    x = np.array([-5.0, -0.5, 0.0, 0.5, 5.0])
    assert np.allclose(soft_threshold(x, 1.0), [-4.0, 0.0, 0.0, 0.0, 4.0])
    assert np.allclose(soft_threshold(x, 0.0), x)


def test_soft_threshold_never_changes_sign():
    rng = np.random.default_rng(6)
    x = rng.normal(size=50) * 3
    shrunk = soft_threshold(x, 1.0)
    moved = shrunk != 0
    assert np.all(np.sign(shrunk[moved]) == np.sign(x[moved]))
    assert np.all(np.abs(shrunk) <= np.abs(x))


def test_lasso_produces_exact_zeros_where_ridge_does_not(noisy_curve):
    X, y = noisy_curve
    w_lasso, _, _ = lasso_gradient_descent(X, y, np.zeros(X.shape[1]), 0.0,
                                           0.02, 20_000, lam=1.0)
    w_ridge, _ = ridge_normal_equation(X, y, lam=1.0)
    assert np.sum(np.abs(w_lasso) < 1e-12) > 0        # some weights are exactly zero
    assert np.sum(np.abs(w_ridge) < 1e-12) == 0       # none of these are


def test_stronger_lasso_penalty_keeps_fewer_weights(noisy_curve):
    X, y = noisy_curve
    kept = []
    for lam in (0.01, 0.1, 1.0, 10.0):
        w, _, _ = lasso_gradient_descent(X, y, np.zeros(X.shape[1]), 0.0, 0.02, 20_000, lam)
        kept.append(int(np.sum(np.abs(w) > 1e-12)))
    assert np.all(np.diff(kept) <= 0)
    assert kept[0] > kept[-1]


def test_lasso_objective_decreases(noisy_curve):
    X, y = noisy_curve
    _, _, history = lasso_gradient_descent(X, y, np.zeros(X.shape[1]), 0.0,
                                           0.02, 5000, lam=1.0)
    assert np.all(np.diff(history["cost"]) <= 1e-10)


# ---------------------------------------------------------------- regularised logistic

def test_penalty_bounds_the_weights_on_perfectly_separable_data():
    """Without a penalty the weight norm grows forever, as lesson 02 exercise 4 showed."""
    rng = np.random.default_rng(7)
    X = np.vstack([rng.normal([-3.0, -3.0], 0.5, size=(50, 2)),
                   rng.normal([3.0, 3.0], 0.5, size=(50, 2))])
    y = np.concatenate([np.zeros(50), np.ones(50)])

    def norm_after(num_iters, lam):
        w, _, _ = gradient_descent(X, y, np.zeros(2), 0.0, 0.5, num_iters,
                                   compute_gradient_logistic, compute_cost_logistic, lam)
        return np.linalg.norm(w)

    assert norm_after(50_000, lam=0.0) > norm_after(10_000, lam=0.0) + 0.1   # still growing
    assert norm_after(50_000, lam=1.0) == pytest.approx(norm_after(10_000, lam=1.0), abs=1e-6)


def test_regularised_logistic_still_classifies_separable_data():
    rng = np.random.default_rng(8)
    X = np.vstack([rng.normal([-3.0, -3.0], 0.5, size=(40, 2)),
                   rng.normal([3.0, 3.0], 0.5, size=(40, 2))])
    y = np.concatenate([np.zeros(40), np.ones(40)])
    model = RegularizedLogisticRegression(lam=1.0, alpha=0.5, num_iters=3000).fit(X, y)
    assert model.score(X, y) == 1.0


# ---------------------------------------------------------------- model classes

def test_ridge_class_matches_the_functions(noisy_curve):
    X, y = noisy_curve
    model = RidgeRegression(lam=1.0, alpha=0.05, num_iters=5000, normalize=False).fit(X, y)
    w, b, _ = gradient_descent(X, y, np.zeros(X.shape[1]), 0.0, 0.05, 5000,
                               compute_gradient_linear, compute_cost_linear, lam=1.0)
    assert np.allclose(model.w, w)
    assert model.b == pytest.approx(b)


def test_lasso_class_uses_the_l1_objective(noisy_curve):
    X, y = noisy_curve
    model = LassoRegression(lam=1.0, alpha=0.02, num_iters=5000, normalize=False).fit(X, y)
    assert model.history["cost"][-1] == pytest.approx(
        compute_cost_lasso(X, y, model.w, model.b, lam=1.0)
    )


def test_normalization_statistics_come_from_the_training_data(noisy_curve):
    X, y = noisy_curve
    model = RidgeRegression(lam=1.0, alpha=0.05, num_iters=500, normalize=True).fit(X, y)
    assert np.allclose(model.mu, X.mean(axis=0))
    assert np.allclose(model.sigma, X.std(axis=0))
