# Lesson 01 — Linear Regression

Predict a continuous value as a straight line (hyperplane) through the data, fitted by
minimising squared error with gradient descent.

Work through [`01_linear_regression.ipynb`](01_linear_regression.ipynb) first — it derives
everything and plots it. [`linear_regression.py`](linear_regression.py) is the clean
version of the same code. [`01_exercise_solutions.ipynb`](01_exercise_solutions.ipynb) works
the five end-of-lesson exercises: the two classic gradient-descent bugs, mean absolute
error and outliers, stopping criteria, polynomial features, and SGD.

## The four formulas

**Model** — what it predicts:

$$f_{w,b}(x) = w \cdot x + b = \sum_{j=1}^{n} w_j x_j + b$$

**Cost** — how wrong it is (mean squared error, halved for convenience):

$$J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)^2$$

**Gradient** — which way is downhill:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)x_j^{(i)}
\qquad
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)$$

**Gradient descent** — repeat until converged, updating both *simultaneously*:

$$w := w - \alpha\frac{\partial J}{\partial w} \qquad b := b - \alpha\frac{\partial J}{\partial b}$$

Vectorised with $e = Xw + b - y$:  $J = \frac{1}{2m}e^\top e$, $\nabla_w J = \frac{1}{m}X^\top e$.

## API

```python
from linear_regression import LinearRegression, compute_cost, compute_gradient, gradient_descent

model = LinearRegression(alpha=0.1, num_iters=2000, normalize=True).fit(X, y)
model.predict(X_new)
model.score(X, y)          # R^2
model.history["cost"]      # learning curve
```

## Things worth remembering

- The $\frac{1}{2}$ in the cost exists only so the 2 from differentiating the square cancels.
- Update **all** parameters from the gradient at the *old* values. Using a just-updated
  `w` to compute `b`'s gradient is a real and quiet bug.
- The learning curve must decrease on every iteration. If it doesn't: $\alpha$ too large,
  or the gradient is wrong.
- Always check a hand-derived gradient against a central difference
  $\frac{J(\theta+\epsilon)-J(\theta-\epsilon)}{2\epsilon}$ before trusting it.
- Scale features (z-score) when they have different ranges — it is the difference between
  $\alpha = 10^{-7}$ and $\alpha = 0.1$.
- Compute $\mu, \sigma$ on training data only, then reuse them on test data.
- MSE is convex here, so there is exactly one minimum — no local optima to worry about.
- A gradient check validates the **gradient**, not the **optimiser**. Bugs in the update
  loop sail straight past it (exercise 1).
- The loss is a modelling choice: squared error treats outliers as signal, absolute error
  treats them as noise (exercise 2).

## Run

```bash
python linear_regression.py       # demo: gradient descent vs the closed form
python -m pytest -q               # 9 tests, including the numerical gradient check
```

## Next

Lesson 02, logistic regression: same cost → gradient → descent machinery, with a sigmoid
output and cross-entropy loss.
