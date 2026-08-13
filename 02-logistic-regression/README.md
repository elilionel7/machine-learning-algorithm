# Lesson 02 — Logistic Regression

Binary classification. The model outputs a probability that an example belongs to the
positive class, and a threshold turns that probability into a predicted label.

Work through [`02_logistic_regression.ipynb`](02_logistic_regression.ipynb) first. It
derives everything and plots it. [`logistic_regression.py`](logistic_regression.py) is the
clean version of the same code. [`02_exercise_solutions.ipynb`](02_exercise_solutions.ipynb)
works the five end-of-lesson exercises: reading weights as odds ratios, making squared
error stall on purpose, class imbalance, perfect separation, and mini-batch training.

## The formulas

**Sigmoid**, which maps any real number into the open interval $(0, 1)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}} \qquad \sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)$$

**Model**, whose output is a probability rather than a class:

$$z = w \cdot x + b \qquad f_{w,b}(x) = \sigma(z) = P(y = 1 \mid x)$$

**Cost**, binary cross-entropy, also called log loss:

$$J(w,b) = \frac{1}{m}\sum_{i=1}^{m}\left[-y^{(i)}\log\big(f_{w,b}(x^{(i)})\big) - \big(1 - y^{(i)}\big)\log\big(1 - f_{w,b}(x^{(i)})\big)\right]$$

**Gradient**, which has the identical form to lesson 01:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m}\big(f_{w,b}(x^{(i)}) - y^{(i)}\big)x_j^{(i)}
\qquad
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}\big(f_{w,b}(x^{(i)}) - y^{(i)}\big)$$

**Decision boundary**, the set of points scoring exactly $0.5$:

$$w \cdot x + b = 0$$

which is a line in two dimensions and a hyperplane in general.

## API

```python
from logistic_regression import LogisticRegression, sigmoid, compute_cost, compute_gradient

model = LogisticRegression(alpha=0.5, num_iters=3000, normalize=True).fit(X, y)
model.predict_proba(X_new)          # probabilities, shape (m,)
model.predict(X_new, threshold=0.7) # labels 0 or 1, shape (m,)
model.score(X, y)                   # accuracy
model.history["cost"]               # learning curve
```

Also available: `accuracy`, `confusion_matrix`, `precision_recall_f1`.

## Things worth remembering

- The model returns a **probability**, and the threshold is a separate decision you make
  afterwards. You can move it without refitting.
- Read $z$ as the **log-odds**. Adding one unit to feature $j$ multiplies the odds by
  $e^{w_j}$.
- Squared error fails here for two reasons. It is not convex once composed with the
  sigmoid, and its gradient vanishes exactly when the model is confidently wrong, because
  the chain rule brings down a factor of $\sigma'(z)$.
- Cross-entropy is the negative log-likelihood of a Bernoulli model. It is derived, not
  invented.
- The $\sigma'$ factor cancels in the cross-entropy gradient, which is why the result is
  the same expression as linear regression.
- Compute the sigmoid piecewise so no exponential overflows, and compute the cost from the
  logits using `np.logaddexp(0, z) - y * z` so no logarithm sees a zero.
- Accuracy is misleading under class imbalance. Read the confusion matrix, then choose
  precision or recall based on which mistake is more costly.
- The boundary is linear in whatever features you supply. Adding squared features gives an
  elliptical boundary in the original space.
- Changing the threshold slides the boundary parallel to itself, to where
  $w \cdot x + b = \log\frac{t}{1-t}$. It never rotates it (exercise 3).
- On perfectly separable data there is no finite optimum. The weights grow without bound
  and buy confidence rather than correctness, which is what regularisation fixes
  (exercise 4).

## Run

```bash
python logistic_regression.py     # demo on two clusters, prints metrics
python -m pytest -q               # 15 tests, including the numerical gradient check
```

## Next

Lesson 03, regularisation: what happens when a model is flexible enough to fit noise, and
how a penalty on large weights controls it.
