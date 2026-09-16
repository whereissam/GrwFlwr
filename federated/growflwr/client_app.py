"""Flower ClientApp: trains the irrigation-safety model on one farm's data.

Raw telemetry is read here and nowhere else. What leaves this process is the
weight vector and a row count -- see `server_app.py` for the audit that proves it.
"""

from __future__ import annotations

import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .data import farm_data
from .model import accuracy, log_loss, predict_proba, train

app = ClientApp()


def _partition(context: Context) -> int:
    return int(context.node_config.get("partition-id", 0))


def _load_params(message: Message) -> list[np.ndarray]:
    arrays = message.content["arrays"].to_numpy_ndarrays()
    return [arrays[0].astype(np.float64), arrays[1].astype(np.float64)]


@app.train()
def train_fn(message: Message, context: Context) -> Message:
    """Take the global model, improve it on local rows, return weights only."""
    params = _load_params(message)
    cfg = message.content["config"]
    epochs = int(cfg["local-epochs"])
    lr = float(cfg["learning-rate"])

    x_train, y_train, _, _ = farm_data(_partition(context))
    params, loss = train(params, x_train, y_train, epochs, lr)

    metrics = MetricRecord(
        {
            "num-examples": len(x_train),
            "train_loss": loss,
            "train_accuracy": accuracy(params, x_train, y_train),
        }
    )
    return Message(
        RecordDict({"arrays": ArrayRecord(params), "metrics": metrics}),
        reply_to=message,
    )


@app.evaluate()
def evaluate_fn(message: Message, context: Context) -> Message:
    """Score the incoming global model on this farm's held-out local rows."""
    params = _load_params(message)
    _, _, x_test, y_test = farm_data(_partition(context))

    metrics = MetricRecord(
        {
            "num-examples": len(x_test),
            "eval_loss": log_loss(y_test, predict_proba(params, x_test)),
            "eval_accuracy": accuracy(params, x_test, y_test),
        }
    )
    return Message(RecordDict({"metrics": metrics}), reply_to=message)
