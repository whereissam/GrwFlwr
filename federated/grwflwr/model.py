"""Multinomial logistic regression on plain NumPy.

Three classes (Low / Medium / High irrigation need), trained by full-batch
gradient descent so a federated round finishes instantly on stage. The whole
model is a 23x3 weight matrix and three biases -- small enough to print, which
makes "this is all that crosses the wire" a claim the audience can check.

Class weighting matters here. High is about 2% of rows, so an unweighted fit
learns to never predict it. Weights are inverse-frequency, computed per client
from its own label counts.
"""

from __future__ import annotations

import numpy as np

from .data import NUM_CLASSES, N_FEATURES


def init_params() -> list[np.ndarray]:
    """Zeroed [weights, bias]. Zeros keep every run reproducible."""
    return [
        np.zeros((N_FEATURES, NUM_CLASSES), dtype=np.float64),
        np.zeros(NUM_CLASSES, dtype=np.float64),
    ]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)  # shift for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict_proba(params: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    w, b = params
    return _softmax(x @ w + b)


def predict(params: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    return predict_proba(params, x).argmax(axis=1)


def class_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights so the rare High class is not ignored."""
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0  # a class this farm never saw contributes nothing
    weights = len(y) / (NUM_CLASSES * counts)
    return weights


def train(
    params: list[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
) -> tuple[list[np.ndarray], float]:
    """Full-batch weighted gradient descent. Returns (new params, final loss)."""
    w, b = (p.copy() for p in params)
    n = max(1, len(x))

    onehot = np.zeros((len(y), NUM_CLASSES))
    onehot[np.arange(len(y)), y] = 1.0
    sample_w = class_weights(y)[y][:, None]

    loss = 0.0
    for _ in range(epochs):
        p = _softmax(x @ w + b)
        error = (p - onehot) * sample_w
        w -= lr * (x.T @ error) / n
        b -= lr * error.mean(axis=0)
        loss = cross_entropy(y, p)

    return [w, b], loss


def cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], eps, 1.0))))


def accuracy(params: list[np.ndarray], x: np.ndarray, y: np.ndarray) -> float:
    """Reported for completeness only -- always predicting Low scores ~75%."""
    return float((predict(params, x) == y).mean())


def per_class_recall(params: list[np.ndarray], x: np.ndarray, y: np.ndarray) -> list[float]:
    """Fraction of each true class the model actually catches."""
    pred = predict(params, x)
    out = []
    for c in range(NUM_CLASSES):
        mask = y == c
        out.append(float((pred[mask] == c).mean()) if mask.any() else float("nan"))
    return out


def macro_f1(params: list[np.ndarray], x: np.ndarray, y: np.ndarray) -> float:
    """Unweighted mean F1 across classes: rare classes count as much as common."""
    pred = predict(params, x)
    scores = []
    for c in range(NUM_CLASSES):
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        if tp == 0.0:
            scores.append(0.0)
            continue
        precision, recall = tp / (tp + fp), tp / (tp + fn)
        scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))
