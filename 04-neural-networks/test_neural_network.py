"""Tests for lesson 04. Run with:  python3 -m pytest 04-neural-networks -q"""

import numpy as np
import pytest

from neural_network import (
    NeuralNetwork,
    gradient_check,
    identity,
    one_hot,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    softmax,
    tanh,
    tanh_derivative,
    zscore_normalize,
)


@pytest.fixture
def xor():
    """The smallest problem no linear model can solve."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    return X, y


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([-2.0, -2.0], 1.0, size=(40, 2)),
                   rng.normal([2.0, 2.0], 1.0, size=(40, 2))])
    y = np.concatenate([np.zeros(40), np.ones(40)])
    return X, y


# ---------------------------------------------------------------- activations

def test_relu_and_its_derivative():
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert np.allclose(relu(z), [0.0, 0.0, 0.0, 0.5, 2.0])
    assert np.allclose(relu_derivative(z), [0.0, 0.0, 0.0, 1.0, 1.0])


def test_activation_derivatives_match_central_differences():
    rng = np.random.default_rng(1)
    z = rng.normal(size=40) * 2
    eps = 1e-6
    for fn, derivative in [(sigmoid, sigmoid_derivative), (tanh, tanh_derivative)]:
        numeric = (fn(z + eps) - fn(z - eps)) / (2 * eps)
        assert np.allclose(derivative(z), numeric, atol=1e-6)


def test_softmax_rows_are_probability_distributions():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(7, 4)) * 3
    P = softmax(Z)
    assert P.shape == Z.shape
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P > 0)


def test_softmax_does_not_overflow_and_ignores_a_constant_shift():
    Z = np.array([[1000.0, 1001.0, 999.0]])
    with np.errstate(over="raise", invalid="raise"):
        P = softmax(Z)
    assert np.all(np.isfinite(P))
    assert P.sum() == pytest.approx(1.0)
    # adding the same constant to every logit in a row must not change the output
    assert np.allclose(softmax(np.array([[1.0, 2.0, 0.0]])), P)


def test_one_hot_encoding():
    encoded = one_hot(np.array([0, 2, 1, 2]), n_classes=3)
    assert encoded.shape == (4, 3)
    assert np.allclose(encoded, [[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])
    assert np.all(encoded.sum(axis=1) == 1)


# ---------------------------------------------------------------- shapes and forward

def test_parameter_shapes_follow_the_layer_sizes():
    net = NeuralNetwork([3, 5, 4, 2], output="softmax")
    assert [W.shape for W in net.W] == [(3, 5), (5, 4), (4, 2)]
    assert [b.shape for b in net.b] == [(5,), (4,), (2,)]


def test_forward_pass_shapes_and_cache():
    net = NeuralNetwork([3, 5, 2], output="softmax")
    X = np.random.default_rng(0).normal(size=(11, 3))
    A, cache = net.forward(X)
    assert A.shape == (11, 2)
    assert [a.shape for a in cache["A"]] == [(11, 3), (11, 5), (11, 2)]
    assert [z.shape for z in cache["Z"]] == [(11, 5), (11, 2)]


def test_weights_are_not_initialised_to_zero():
    """Zero weights make every unit in a layer identical forever."""
    net = NeuralNetwork([4, 6, 1])
    assert not np.allclose(net.W[0], 0.0)
    assert np.allclose(net.b[0], 0.0)      # biases may safely start at zero


def test_he_initialisation_has_roughly_the_intended_scale():
    net = NeuralNetwork([500, 500, 1], hidden_activation="relu", seed=0)
    assert np.std(net.W[0]) == pytest.approx(np.sqrt(2.0 / 500), rel=0.1)


def test_rejects_an_unknown_output_type():
    with pytest.raises(ValueError):
        NeuralNetwork([2, 2, 1], output="not_a_real_output")


# ---------------------------------------------------------------- gradients

@pytest.mark.parametrize("output,n_out", [("sigmoid", 1), ("softmax", 3), ("linear", 2)])
@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid"])
def test_backpropagation_matches_numerical_gradients(output, n_out, activation):
    """The central test of the lesson, run over every output and activation pairing."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 4))
    if output == "softmax":
        Y = one_hot(rng.integers(0, n_out, 12), n_out)
    elif output == "sigmoid":
        Y = (rng.random((12, 1)) > 0.5).astype(float)
    else:
        Y = rng.normal(size=(12, n_out))

    net = NeuralNetwork([4, 5, 4, n_out], hidden_activation=activation,
                        output=output, lam=0.0, seed=1)
    assert gradient_check(net, X, Y) < 1e-7


def test_backpropagation_is_correct_with_the_l2_penalty():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(10, 3))
    Y = (rng.random((10, 1)) > 0.5).astype(float)
    net = NeuralNetwork([3, 6, 1], hidden_activation="tanh", output="sigmoid",
                        lam=2.5, seed=2)
    assert gradient_check(net, X, Y) < 1e-7


def test_bias_gradients_carry_no_penalty():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(8, 3))
    Y = (rng.random((8, 1)) > 0.5).astype(float)

    gradients = []
    for lam in (0.0, 10.0):
        net = NeuralNetwork([3, 4, 1], hidden_activation="tanh", output="sigmoid",
                            lam=lam, seed=7)
        _, cache = net.forward(X)
        gradients.append(net.backward(Y, cache)[1])
    for db_zero, db_penalised in zip(*gradients):
        assert np.allclose(db_zero, db_penalised)


def test_output_gradient_is_prediction_minus_label():
    """The fused output gradient, the same expression as lessons 01 and 02."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(9, 3))
    Y = (rng.random((9, 1)) > 0.5).astype(float)
    net = NeuralNetwork([3, 1], output="sigmoid", seed=0)   # no hidden layer

    A, cache = net.forward(X)
    dW, db = net.backward(Y, cache)
    expected_dW = X.T @ (A - Y) / len(Y)
    assert np.allclose(dW[0], expected_dW)


# ---------------------------------------------------------------- learning

def test_a_network_learns_xor(xor):
    X, y = xor
    net = NeuralNetwork([2, 4, 1], hidden_activation="tanh", output="sigmoid", seed=3)
    net.fit(X, y, alpha=0.5, epochs=3000, batch_size=4)
    assert net.score(X, y) == 1.0


def test_a_network_without_a_hidden_layer_cannot_learn_xor(xor):
    """No hidden layer means this is exactly logistic regression, and XOR is not linearly
    separable, so 75 percent is the ceiling."""
    X, y = xor
    net = NeuralNetwork([2, 1], output="sigmoid", seed=3)
    net.fit(X, y, alpha=0.5, epochs=4000, batch_size=4)
    assert net.score(X, y) <= 0.75


def test_training_reduces_the_cost(blobs):
    X, y = blobs
    net = NeuralNetwork([2, 8, 1], hidden_activation="relu", output="sigmoid", seed=0)
    before = net.compute_cost(X, y.reshape(-1, 1))
    net.fit(X, y, alpha=0.2, epochs=150, batch_size=16)
    assert net.history["cost"][-1] < before
    assert net.score(X, y) > 0.95


def test_multiclass_classification_with_softmax():
    rng = np.random.default_rng(8)
    centres = [[-3.0, 0.0], [3.0, 0.0], [0.0, 3.0]]
    X = np.vstack([rng.normal(c, 0.6, size=(50, 2)) for c in centres])
    y = np.repeat([0, 1, 2], 50)

    net = NeuralNetwork([2, 12, 3], hidden_activation="relu", output="softmax", seed=1)
    net.fit(X, one_hot(y, 3), alpha=0.3, epochs=300, batch_size=32)
    assert net.score(X, y) > 0.95
    assert np.allclose(net.predict_proba(X).sum(axis=1), 1.0)


def test_linear_output_fits_a_regression_target():
    rng = np.random.default_rng(9)
    x = rng.uniform(-2, 2, 200)
    X = x.reshape(-1, 1)
    y = np.sin(2 * x).reshape(-1, 1)

    net = NeuralNetwork([1, 24, 24, 1], hidden_activation="tanh", output="linear", seed=2)
    net.fit(X, y, alpha=0.05, epochs=800, batch_size=32)
    assert net.score(X, y.ravel()) > 0.95


def test_penalty_keeps_the_weights_smaller(blobs):
    X, y = blobs
    norms = []
    for lam in (0.0, 5.0):
        net = NeuralNetwork([2, 8, 1], hidden_activation="tanh", output="sigmoid",
                            lam=lam, seed=4)
        net.fit(X, y, alpha=0.2, epochs=300, batch_size=16)
        norms.append(sum(float(np.sum(W ** 2)) for W in net.W))
    assert norms[1] < norms[0]


def test_fit_accepts_a_one_dimensional_label_array(blobs):
    X, y = blobs
    net = NeuralNetwork([2, 5, 1], output="sigmoid", seed=0).fit(X, y, epochs=20)
    assert net.predict(X).shape == (len(y),)


def test_zscore_normalize_reuses_training_statistics():
    rng = np.random.default_rng(10)
    X = rng.normal(5, 3, size=(30, 2))
    Xn, mu, sigma = zscore_normalize(X)
    assert np.allclose(Xn.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(zscore_normalize(X, mu, sigma)[0], Xn)
