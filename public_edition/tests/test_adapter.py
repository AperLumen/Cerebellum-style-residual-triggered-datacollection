"""Unit tests for cerebellum_cancel.adapter heads and helpers.

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/
"""

import numpy as np
import pytest

# macOS Accelerate BLAS emits spurious FP warnings from matmul worker
# threads (numpy issue); the computed values are unaffected.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*encountered in matmul:RuntimeWarning")

from cerebellum_cancel.adapter import (LogisticHead, MLPHead,
                                       average_precision)


def test_average_precision_perfect_ranking():
    y = np.array([0, 1, 0, 1, 0])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    assert average_precision(y, s) == 1.0


def test_average_precision_worst_ranking():
    y = np.array([1, 0])
    s = np.array([0.1, 0.9])
    assert average_precision(y, s) == 0.5


def test_logistic_head_separates_linearly():
    rng = np.random.default_rng(0)
    x0 = rng.normal(0, 1, (200, 8))
    x1 = rng.normal(3, 1, (200, 8))
    X = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(200), np.ones(200)])
    head = LogisticHead(8).fit(X, y)
    assert (head.scores(X[200:]) > head.scores(X[:200])).mean() > 0.95


def test_mlp_head_separates_nonlinearly():
    """XOR is not linearly separable: the MLP must beat a linear reference."""
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (800, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.float32)
    mlp = MLPHead(2, hidden=16, seed=0).fit(X, y, iters=800)
    acc = ((mlp.scores(X) > 0) == (y > 0.5)).mean()
    lin = LogisticHead(2, seed=0).fit(X, y, iters=300, lr=0.1)
    acc_lin = ((lin.scores(X) > 0) == (y > 0.5)).mean()
    assert acc > 0.9
    assert acc_lin < 0.75      # linear reference must actually struggle
    assert acc > acc_lin + 0.1


def test_mlp_head_deterministic():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (100, 4))
    y = (X[:, 0] > 0).astype(np.float32)
    a = MLPHead(4, seed=3).fit(X, y).scores(X)
    b = MLPHead(4, seed=3).fit(X, y).scores(X)
    assert np.allclose(a, b)
