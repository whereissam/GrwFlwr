"""Logistic regression on plain NumPy.

Deliberately not scikit-learn or torch: the model has to train in well under a
second so a federated round is instant on stage, and the weights have to be two
small arrays that are obvious to serialize and easy to show as "this is all that
crosses the wire".
"""

from __future__ import annotations

import numpy as np

from .data import N_FEATURES


def init_params() -> list[np.ndarray]:
    """Return zeroed [weights, bias]. Zeros keep every run reproducible."""
    return [np.zeros(N_FEATURES, dtype=np.float64), np.zeros(1, dtype=np.float64)]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Branch on sign so exp() never overflows on large-magnitude logits.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def predict_proba(params: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    """Probability that irrigating now is safe."""
    w, b = params
    return _sigmoid(x @ w + b[0])


def train(
    params: list[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
) -> tuple[list[np.ndarray], float]:
    """Full-batch gradient descent. Returns (new params, final loss)."""
    w, b = (p.copy() for p in params)
    n = max(1, len(x))
    loss = 0.0

    for _ in range(epochs):
        p = _sigmoid(x @ w + b[0])
        error = p - y
        w -= lr * (x.T @ error) / n
        b -= lr * error.mean()
        loss = log_loss(y, p)

    return [w, b], loss


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def accuracy(params: list[np.ndarray], x: np.ndarray, y: np.ndarray) -> float:
    return float(((predict_proba(params, x) >= 0.5).astype(np.float64) == y).mean())
