"""Flower ClientApp: trains the irrigation-need model on one farm's rows.

The farm's CSV is read here and nowhere else. What leaves this process is a
23x3 weight matrix, three biases, and a row count.
"""

from __future__ import annotations

import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .data import NUM_CLASSES, farm_data
from .model import cross_entropy, macro_f1, per_class_recall, predict_proba, train

app = ClientApp()


def _partition(context: Context) -> int:
    return int(context.node_config.get("partition-id", 0))


def _load_params(message: Message) -> list[np.ndarray]:
    arrays = message.content["arrays"].to_numpy_ndarrays()
    return [arrays[0].astype(np.float64), arrays[1].astype(np.float64)]


@app.train()
def train_fn(message: Message, context: Context) -> Message:
    """Improve the global model on local rows, return weights only."""
    params = _load_params(message)
    cfg = message.content["config"]

    x_train, y_train, _, _ = farm_data(_partition(context))
    params, loss = train(params, x_train, y_train,
                         int(cfg["local-epochs"]), float(cfg["learning-rate"]))

    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    return Message(
        RecordDict({
            "arrays": ArrayRecord(params),
            "metrics": MetricRecord({
                "num-examples": len(x_train),
                "train_loss": loss,
                "train_macro_f1": macro_f1(params, x_train, y_train),
                # Reported so the server can show how few High rows any one
                # farm actually holds -- the core argument for federating.
                "high_examples": int(counts[2]),
            }),
        }),
        reply_to=message,
    )


@app.evaluate()
def evaluate_fn(message: Message, context: Context) -> Message:
    """Score the incoming global model on this farm's held-out field."""
    params = _load_params(message)
    _, _, x_test, y_test = farm_data(_partition(context))
    recalls = per_class_recall(params, x_test, y_test)

    return Message(
        RecordDict({"metrics": MetricRecord({
            "num-examples": len(x_test),
            "eval_loss": cross_entropy(y_test, predict_proba(params, x_test)),
            "eval_macro_f1": macro_f1(params, x_test, y_test),
            "eval_high_recall": 0.0 if np.isnan(recalls[2]) else recalls[2],
        })}),
        reply_to=message,
    )
