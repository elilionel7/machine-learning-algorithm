# Machine Learning Algorithms From Scratch

Learning the fundamentals of machine learning by implementing every algorithm myself,
with NumPy arrays and nothing else. No scikit-learn, no autograd — the point is to write
the maths out and see it work.

Each lesson is a folder containing:

| file | purpose |
|---|---|
| `NN_<topic>.ipynb` | the lesson: equations in LaTeX, derivations, plots, step-by-step build |
| `NN_exercise_solutions.ipynb` | worked answers to the end-of-lesson exercises |
| `<topic>.py` | the clean reference implementation, importable by later lessons |
| `test_<topic>.py` | tests, including numerical gradient checks |
| `README.md` | one-page summary of the formulas |

## Lessons

| # | topic | key ideas |
|---|---|---|
| [01](01-linear-regression/) | Linear regression | model, mean squared error, gradient descent, learning rate, feature scaling, normal equation |
| [02](02-logistic-regression/) | Logistic regression | sigmoid, log-odds, cross-entropy, decision boundary, precision and recall, numerical stability |
| 03 | Regularisation | overfitting, ridge and lasso, bias and variance |
| 04 | Neural networks | forward pass, backpropagation |

(03 onward are planned, not written yet.)

## Setup

```bash
pip install -r requirements.txt
nbstripout --install --attributes .gitattributes
```

The second line matters, and is needed once per clone. It registers a git filter that
strips notebook outputs on the way *into* a commit while leaving your local copy
untouched — so diffs stay readable instead of being thousands of lines of base64 PNG.
Without it, notebooks get committed with their outputs.

## Running things

```bash
# the reference implementation's built-in demo
python 01-linear-regression/linear_regression.py
python 02-logistic-regression/logistic_regression.py

# the tests for one lesson
python -m pytest 01-linear-regression -q

# every test in the repo
python -m pytest -q

# the notebook
jupyter notebook 01-linear-regression/01_linear_regression.ipynb
```

Notebooks are committed **without** their outputs (see Setup above), so run a notebook
top-to-bottom to see its plots.

## Conventions used throughout

- `X` is the design matrix with shape `(m, n)` — **one row per example, one column per feature**
- `y` has shape `(m,)`, `w` has shape `(n,)`, `b` is a scalar
- superscript $(i)$ indexes examples, subscript $j$ indexes features
- every hand-derived gradient gets a numerical (finite-difference) check in the tests
