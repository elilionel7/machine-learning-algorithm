"""Neural networks from scratch with NumPy: forward pass and backpropagation.

A neural network is a stack of the building block from lessons 01 and 02. Each layer
computes a linear score and then applies a nonlinearity:

    Z = A_prev @ W + b          the same linear step as every previous lesson
    A = g(Z)                    an activation function applied entrywise

Stacking these without the nonlinearity would be pointless, because a composition of
linear maps is itself linear. The activation is what buys the extra expressive power.

Shape convention, unchanged from lessons 01 to 03: one row per example.

    A^[0] = X            (m, n_0)      the input
    W^[l]                (n_{l-1}, n_l)
    b^[l]                (n_l,)
    Z^[l], A^[l]         (m, n_l)

Backpropagation is the chain rule applied layer by layer, from the output backwards:

    dZ^[l]      = dA^[l] * g'(Z^[l])                 (m, n_l)
    dW^[l]      = A^[l-1].T @ dZ^[l] / m             (n_{l-1}, n_l)
    db^[l]      = sum of dZ^[l] over examples / m    (n_l,)
    dA^[l-1]    = dZ^[l] @ W^[l].T                   (m, n_{l-1})

The name refers to dA flowing right to left through the network, each layer converting the
gradient of the cost with respect to its output into the gradient with respect to its
input, and picking up its own parameter gradients on the way.
"""

import numpy as np


# ----------------------------------------------------------------------------------
# activation functions, each paired with its derivative in terms of z
# ----------------------------------------------------------------------------------

def sigmoid(z):
    """Map any real number into (0, 1), computed so large inputs cannot overflow."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    """The rectified linear unit, max(0, z). The default hidden activation."""
    return np.maximum(0.0, z)


def relu_derivative(z):
    """1 where z is positive, 0 elsewhere.

    Strictly there is no derivative at exactly zero. Any value in [0, 1] is a valid
    subgradient there and the convention is to use 0, which is what z > 0 returns.
    """
    return (z > 0).astype(float)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2


def identity(z):
    return z


def identity_derivative(z):
    return np.ones_like(z)


def softmax(z):
    """Turn each row of logits into a probability distribution over the columns.

    Subtracting the row maximum before exponentiating changes nothing mathematically,
    because the same factor cancels between numerator and denominator, but it guarantees
    the largest exponent is exp(0) = 1 so nothing overflows.
    """
    z = np.asarray(z, dtype=float)
    shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "identity": (identity, identity_derivative),
}


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------

def one_hot(y, n_classes=None):
    """Turn integer labels of shape (m,) into a (m, n_classes) indicator matrix."""
    y = np.asarray(y).astype(int).ravel()
    n_classes = int(y.max()) + 1 if n_classes is None else n_classes
    encoded = np.zeros((len(y), n_classes))
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def zscore_normalize(X, mu=None, sigma=None):
    """Scale each feature to zero mean and unit standard deviation."""
    if mu is None:
        mu = X.mean(axis=0)
    if sigma is None:
        sigma = X.std(axis=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma


# ----------------------------------------------------------------------------------
# the network
# ----------------------------------------------------------------------------------

class NeuralNetwork:
    """A fully connected feedforward network trained by mini-batch gradient descent.

    Parameters
    ----------
    layer_sizes : list of int
        Widths from input to output, for example [2, 8, 8, 1].
    hidden_activation : "relu", "tanh" or "sigmoid"
        Applied to every layer except the output.
    output : "sigmoid", "softmax" or "linear"
        Picks the output activation and the matching loss. Binary cross-entropy for
        sigmoid, categorical cross-entropy for softmax, mean squared error for linear.
    lam : float
        L2 penalty on the weights, exactly as in lesson 03. Biases are never penalised.
    seed : int
        Controls the random initialisation.
    """

    def __init__(self, layer_sizes, hidden_activation="relu", output="sigmoid",
                 lam=0.0, seed=0):
        if output not in ("sigmoid", "softmax", "linear"):
            raise ValueError("output must be 'sigmoid', 'softmax' or 'linear'")
        self.layer_sizes = list(layer_sizes)
        self.hidden_activation = hidden_activation
        self.output = output
        self.lam = lam
        self.seed = seed
        self.history = None
        self._initialise_parameters()

    # -------------------------------------------------------------- initialisation

    def _initialise_parameters(self):
        """He initialisation for relu, Xavier otherwise.

        Weights must not start at zero. Every unit in a layer would then compute the same
        thing and receive the same gradient forever, so the layer could never become
        anything more than a single unit. This is the symmetry breaking problem, and it is
        the reason neural networks need random initialisation while linear and logistic
        regression are perfectly happy starting from zero.

        The scale matters too. Too small and the signal dies as it passes through layers,
        too large and it explodes. Both variants below keep the variance of the activations
        roughly stable from layer to layer.
        """
        rng = np.random.default_rng(self.seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            if self.hidden_activation == "relu":
                scale = np.sqrt(2.0 / n_in)      # He
            else:
                scale = np.sqrt(1.0 / n_in)      # Xavier
            self.W.append(rng.normal(0.0, scale, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    # -------------------------------------------------------------- forward

    def forward(self, X):
        """Run the network forward, returning the output and the cache backprop needs.

        The cache holds every Z and A because the backward pass needs them. This is why
        training uses more memory than prediction does.
        """
        activation, _ = ACTIVATIONS[self.hidden_activation]
        A = np.asarray(X, dtype=float)
        cache = {"A": [A], "Z": []}

        n_layers = len(self.W)
        for layer in range(n_layers):
            Z = A @ self.W[layer] + self.b[layer]
            if layer == n_layers - 1:
                if self.output == "sigmoid":
                    A = sigmoid(Z)
                elif self.output == "softmax":
                    A = softmax(Z)
                else:
                    A = Z
            else:
                A = activation(Z)
            cache["Z"].append(Z)
            cache["A"].append(A)
        return A, cache

    # -------------------------------------------------------------- cost

    def compute_cost(self, X, Y):
        """Data loss plus the L2 penalty. Y must already be in output shape (m, n_out)."""
        Y = np.asarray(Y, dtype=float)
        _, cache = self.forward(X)
        Z_out = cache["Z"][-1]
        m = Y.shape[0]

        if self.output == "sigmoid":
            # computed from the logits, the stable form from lesson 02
            data_loss = float(np.mean(np.sum(np.logaddexp(0.0, Z_out) - Y * Z_out, axis=1)))
        elif self.output == "softmax":
            # log-sum-exp, the multi class version of the same trick
            shifted = Z_out - Z_out.max(axis=1, keepdims=True)
            log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            data_loss = float(-np.sum(Y * log_probs) / m)
        else:
            data_loss = float(np.sum((Z_out - Y) ** 2) / (2 * m))

        penalty = self.lam * sum(float(np.sum(W ** 2)) for W in self.W) / (2 * m)
        return data_loss + penalty

    # -------------------------------------------------------------- backward

    def backward(self, Y, cache):
        """Backpropagation. Returns lists of dW and db matching self.W and self.b.

        The output layer gradient is fused with the loss. For sigmoid with binary
        cross-entropy, and for softmax with categorical cross-entropy, the algebra
        collapses to the same expression seen in lessons 01 and 02:

            dZ_out = (A_out - Y) / m

        which is prediction minus label, again. Computing it directly rather than
        multiplying dA by the activation derivative avoids the numerical trouble that
        would come from dividing by probabilities near zero.
        """
        Y = np.asarray(Y, dtype=float)
        m = Y.shape[0]
        _, activation_derivative = ACTIVATIONS[self.hidden_activation]
        n_layers = len(self.W)

        dW = [None] * n_layers
        db = [None] * n_layers

        A_out = cache["A"][-1]
        dZ = (A_out - Y) / m          # identical for all three output choices

        for layer in reversed(range(n_layers)):
            A_prev = cache["A"][layer]
            dW[layer] = A_prev.T @ dZ + self.lam * self.W[layer] / m
            db[layer] = dZ.sum(axis=0)
            if layer > 0:
                dA_prev = dZ @ self.W[layer].T
                dZ = dA_prev * activation_derivative(cache["Z"][layer - 1])
        return dW, db

    # -------------------------------------------------------------- training

    def fit(self, X, Y, alpha=0.1, epochs=200, batch_size=32, seed=0, verbose=False):
        """Mini-batch gradient descent, the optimiser from the lesson 01 solutions."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        rng = np.random.default_rng(seed)
        m = X.shape[0]
        self.history = {"epoch": [], "cost": []}

        for epoch in range(epochs):
            order = rng.permutation(m)
            for start in range(0, m, batch_size):
                idx = order[start:start + batch_size]
                _, cache = self.forward(X[idx])
                dW, db = self.backward(Y[idx], cache)
                for layer in range(len(self.W)):
                    self.W[layer] -= alpha * dW[layer]
                    self.b[layer] -= alpha * db[layer]

            cost = self.compute_cost(X, Y)
            self.history["epoch"].append(epoch)
            self.history["cost"].append(cost)
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:>5}  cost {cost:.6f}")
        return self

    # -------------------------------------------------------------- prediction

    def predict_proba(self, X):
        return self.forward(X)[0]

    def predict(self, X, threshold=0.5):
        A = self.predict_proba(X)
        if self.output == "softmax":
            return A.argmax(axis=1)
        if self.output == "sigmoid":
            return (A >= threshold).astype(int).ravel()
        return A

    def score(self, X, y):
        """Accuracy for the classifiers, R squared for the linear output."""
        y = np.asarray(y)
        if self.output == "linear":
            predictions = self.predict(X).ravel()
            residual = np.sum((y.ravel() - predictions) ** 2)
            total = np.sum((y.ravel() - y.mean()) ** 2)
            return float(1 - residual / total)
        return float(np.mean(self.predict(X) == y.ravel()))


# ----------------------------------------------------------------------------------
# gradient checking
# ----------------------------------------------------------------------------------

def gradient_check(network, X, Y, eps=1e-6):
    """Compare backpropagation against central differences over every parameter.

    Returns the relative error

        ||analytic - numeric|| / (||analytic|| + ||numeric||)

    Below about 1e-7 the gradients agree. Above 1e-3 there is a real bug. This is the
    single most valuable test to have when implementing backpropagation, because a wrong
    gradient still trains, just badly, and looks like a tuning problem rather than a bug.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    _, cache = network.forward(X)
    dW, db = network.backward(Y, cache)

    analytic, numeric = [], []
    for params, grads in ((network.W, dW), (network.b, db)):
        for layer, (param, grad) in enumerate(zip(params, grads)):
            flat = param.ravel()
            for index in range(flat.size):
                original = flat[index]
                flat[index] = original + eps
                cost_up = network.compute_cost(X, Y)
                flat[index] = original - eps
                cost_down = network.compute_cost(X, Y)
                flat[index] = original
                numeric.append((cost_up - cost_down) / (2 * eps))
                analytic.append(grad.ravel()[index])

    analytic, numeric = np.array(analytic), np.array(numeric)
    denominator = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    if denominator == 0:
        return 0.0
    return float(np.linalg.norm(analytic - numeric) / denominator)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # XOR, the smallest problem that no linear model can solve
    X_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_xor = np.array([0, 1, 1, 0])

    net = NeuralNetwork([2, 4, 1], hidden_activation="tanh", output="sigmoid", seed=3)
    net.fit(X_xor, y_xor, alpha=0.5, epochs=4000, batch_size=4)
    print("XOR")
    for point, probability in zip(X_xor, net.predict_proba(X_xor).ravel()):
        print(f"  {point} -> {probability:.4f}")
    print(f"  accuracy {net.score(X_xor, y_xor):.3f}, final cost {net.history['cost'][-1]:.6f}")

    # two interleaving spirals, a harder two dimensional problem
    def make_spirals(n_per_class, noise=0.06, seed=0):
        r = np.random.default_rng(seed)
        t = np.sqrt(r.uniform(0.05, 1, n_per_class)) * 2.6 * np.pi
        points, labels = [], []
        for k in (0, 1):
            angle = t + k * np.pi
            xs = np.column_stack([t * np.cos(angle), t * np.sin(angle)]) / 9.0
            points.append(xs + r.normal(0, noise, xs.shape))
            labels.append(np.full(n_per_class, k))
        return np.vstack(points), np.concatenate(labels)

    X_spiral, y_spiral = make_spirals(300, seed=1)
    deep = NeuralNetwork([2, 32, 32, 1], hidden_activation="relu", output="sigmoid",
                         lam=0.01, seed=1)
    deep.fit(X_spiral, y_spiral, alpha=0.3, epochs=400, batch_size=32)
    print(f"\nspirals: accuracy {deep.score(X_spiral, y_spiral):.4f}, "
          f"final cost {deep.history['cost'][-1]:.4f}")

    small = NeuralNetwork([2, 6, 1], hidden_activation="relu", output="sigmoid", seed=1)
    print(f"gradient check, relative error: {gradient_check(small, X_spiral[:20], y_spiral[:20]):.3e}")
