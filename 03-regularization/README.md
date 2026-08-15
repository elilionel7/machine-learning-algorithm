# Lesson 03 — Regularisation

Fitting the training data as well as possible is the wrong goal. This lesson measures the
gap between training error and test error, then closes it by penalising large weights.

Work through [`03_regularization.ipynb`](03_regularization.ipynb) first. It derives
everything and plots it. [`regularization.py`](regularization.py) is the clean version of
the same code. [`03_exercise_solutions.ipynb`](03_exercise_solutions.ipynb) works the five
end-of-lesson exercises: ridge as data augmentation, early stopping as regularisation, the
one standard error rule, lasso on irrelevant features, and learning curves.

## The formulas

**Ridge**, or L2, which shrinks every weight:

$$J(w,b) = \frac{1}{2m}\left[\sum_{i=1}^{m}\big(f_{w,b}(x^{(i)}) - y^{(i)}\big)^2 + \lambda\sum_{j=1}^{n}w_j^2\right]$$

$$\frac{\partial J}{\partial w_j} = \frac{1}{m}\left[\sum_{i=1}^{m}\big(f_{w,b}(x^{(i)}) - y^{(i)}\big)x_j^{(i)} + \lambda w_j\right]
\qquad
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}\big(f_{w,b}(x^{(i)}) - y^{(i)}\big)$$

Substituted into the update, the penalty becomes a shrink factor applied before every step,
which is why L2 is also called **weight decay**:

$$w_j := w_j\left(1 - \frac{\alpha\lambda}{m}\right) - \frac{\alpha}{m}\sum_{i=1}^{m}\big(f_{w,b}(x^{(i)}) - y^{(i)}\big)x_j^{(i)}$$

**Closed form**, where $P$ is the identity with entry $(0,0)$ zeroed so the bias escapes:

$$\theta = \big(X_b^\top X_b + \lambda P\big)^{-1}X_b^\top y$$

**Lasso**, or L1, which zeroes some weights outright:

$$J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}\big(f_{w,b}(x^{(i)}) - y^{(i)}\big)^2 + \frac{\lambda}{m}\sum_{j=1}^{n}\lvert w_j \rvert$$

solved by proximal gradient descent, where each step is followed by soft thresholding:

$$\operatorname{soft}(x, t) = \operatorname{sign}(x)\max\big(\lvert x \rvert - t, 0\big)$$

**Bias and variance decomposition**, which the notebook confirms numerically:

$$\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{\big(\mathbb{E}[\hat{f}(x)] - f(x)\big)^2}_{\text{bias}^2} + \underbrace{\operatorname{Var}\big[\hat{f}(x)\big]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}}$$

## API

```python
from regularization import RidgeRegression, LassoRegression, RegularizedLogisticRegression

ridge = RidgeRegression(lam=1.0, alpha=0.01, num_iters=50_000).fit(X, y)
lasso = LassoRegression(lam=1.0, alpha=0.02, num_iters=50_000).fit(X, y)
clf = RegularizedLogisticRegression(lam=1.0, alpha=0.5, num_iters=3000).fit(X, y)

ridge.predict(X_new); ridge.score(X, y); ridge.mse(X, y)
```

Also available: `polynomial_features`, `train_test_split`, `k_fold_indices`,
`ridge_normal_equation`, `soft_threshold`, `zscore_normalize`.

## Things worth remembering

- Training error falls monotonically with model complexity, so it can never tell you a
  model is too complex. Only held out data can.
- The mechanical signature of overfitting is an exploding weight norm. In the notebook it
  runs from 0.73 at degree 1 to 42,290 at degree 15.
- Increasing $\lambda$ and decreasing model complexity do the same job. Regularisation
  turns a choice between integers into a continuous dial.
- The bias $b$ is never penalised. It sets the level of the predictions, not the shape,
  and shrinking it towards zero would bias every prediction for no gain.
- Gradient descent diverges when $\alpha \geq 2m/\lambda$. Raising $\lambda$ without
  lowering $\alpha$ is a reliable way to get `nan`.
- Scaling is mandatory. The penalty treats all weights alike, so unscaled features make
  the same $\lambda$ mean different things.
- The derivative of $w^2$ vanishes at zero, so ridge weights approach zero without
  arriving. The derivative of $\lvert w \rvert$ is $\pm 1$ everywhere, so lasso pushes all
  the way to zero and pins them there.
- Adding $\lambda$ to the diagonal also fixes conditioning. In the notebook it takes the
  condition number from $1.3 \times 10^{18}$, where float64 has no correct digits left, to
  $83$.
- $\lambda$ cannot be chosen on the training set, which always prefers zero, nor on the
  test set, which would stop being honest. Use a validation split or k-fold cross
  validation.
- Small held out sets give noisy scores. A measured test error can fall below the noise
  variance even though the noise variance is the floor in expectation.
- Ridge is exactly ordinary least squares on data augmented with $n$ rows of
  $\sqrt{\lambda}I$ and zero targets, which is one identity that explains both the
  conditioning fix and the Bayesian reading (exercise 1).
- Gradient descent grows $\|w\|$ from zero, so the iteration count is itself a weight
  budget. Early stopping regularises whether you meant it to or not (exercise 2).
- Learning curves say whether more data will help. A closed gap means high bias and more
  data will not help; an open gap means high variance and it will (exercise 5).

## Run

```bash
python regularization.py     # ridge lambda sweep and a lasso against ridge comparison
python -m pytest -q          # 24 tests, including gradient checks with the penalty
```

## Next

Lesson 04, neural networks: stack these building blocks into layers and differentiate
through all of them with backpropagation.
