# Lesson 04 — Neural Networks

Stack the linear-then-nonlinear block from lessons 01 and 02 into layers, and compute the
gradient through the whole stack with backpropagation.

Work through [`04_neural_networks.ipynb`](04_neural_networks.ipynb) first. It derives
everything and plots it. [`neural_network.py`](neural_network.py) is the clean version of
the same code.

## The formulas

**Forward pass**, with $A^{[0]} = X$:

$$Z^{[l]} = A^{[l-1]}W^{[l]} + b^{[l]} \qquad A^{[l]} = g^{[l]}\big(Z^{[l]}\big) \qquad l = 1 \ldots L$$

**Backward pass**, working from layer $L$ down to layer 1, where $\odot$ is entrywise
multiplication:

$$
\begin{aligned}
dZ^{[l]} &= dA^{[l]} \odot g'\big(Z^{[l]}\big) && (m, n_l) \\
dW^{[l]} &= \tfrac{1}{m}\big(A^{[l-1]}\big)^\top dZ^{[l]} && (n_{l-1}, n_l) \\
db^{[l]} &= \tfrac{1}{m}\textstyle\sum_i dZ^{[l]}_i && (n_l,) \\
dA^{[l-1]} &= dZ^{[l]}\big(W^{[l]}\big)^\top && (m, n_{l-1})
\end{aligned}
$$

**Output layer**, where the loss and the activation fuse. Identical for sigmoid with binary
cross-entropy, softmax with categorical cross-entropy, and linear with squared error:

$$dZ^{[L]} = \frac{1}{m}\big(A^{[L]} - Y\big)$$

**Initialisation**, chosen to keep the activation variance stable across layers:

$$\text{He (relu)}: \sigma = \sqrt{\frac{2}{n_{l-1}}} \qquad \text{Xavier (tanh, sigmoid)}: \sigma = \sqrt{\frac{1}{n_{l-1}}}$$

**Softmax**, computed after subtracting the row maximum so nothing overflows:

$$\operatorname{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

## Shapes

One row per example, the same convention as every previous lesson.

| symbol | shape |
|---|---|
| $A^{[0]} = X$ | $(m, n_0)$ |
| $W^{[l]}$ | $(n_{l-1}, n_l)$ |
| $b^{[l]}$ | $(n_l,)$ |
| $Z^{[l]}, A^{[l]}$ | $(m, n_l)$ |

## API

```python
from neural_network import NeuralNetwork, gradient_check, one_hot

net = NeuralNetwork([2, 16, 16, 1], hidden_activation="relu",
                    output="sigmoid", lam=0.01, seed=0)
net.fit(X, y, alpha=0.3, epochs=400, batch_size=32)
net.predict_proba(X); net.predict(X); net.score(X, y)

gradient_check(net, X, Y)      # relative error, below 1e-7 means the gradient is right
```

`output` is `"sigmoid"` for binary classification, `"softmax"` for multiple classes, or
`"linear"` for regression. Each selects the matching loss automatically.

## Things worth remembering

- A stack of linear layers collapses algebraically into a single linear layer. The
  activation is the only reason depth buys anything.
- The hidden layer does not learn a curved boundary. It learns a change of coordinates in
  which a straight boundary works, and the notebook plots exactly that for XOR.
- Zero initialisation is fatal here, though it was fine in lessons 01 to 03. Every unit in a
  layer would compute the same thing and receive the same gradient forever.
- The cost is no longer convex. Half the random seeds fail to solve XOR with two hidden
  units, so the starting point now matters. Extra width makes the optimisation easier
  independently of what the model can represent.
- The sigmoid derivative peaks at 0.25, so backpropagation shrinks the gradient once per
  layer. In the notebook's six layer network the first layer's gradient is about $10^5$
  times smaller than the last. Use relu or tanh in hidden layers.
- Fuse the loss with the output activation. The $\sigma(1-\sigma)$ factor cancels and you
  get prediction minus label, the same expression as in lessons 01, 02 and 03.
- Run `gradient_check` once on a small network before trusting anything. A wrong gradient
  still trains, just worse, and looks exactly like a tuning problem.
- Depth is more parameter efficient than width. In the notebook, two layers of 16 units
  match one layer of 256 using about a third of the parameters.
- Everything from lesson 03 still applies. The L2 penalty attaches to the weight matrices
  the same way, biases still exempt.

## Run

```bash
python neural_network.py     # XOR, spirals, and a gradient check
python -m pytest -q          # 30 tests, gradient checked over every activation and output
```
