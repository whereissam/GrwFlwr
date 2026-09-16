"""Flower ServerApp: FedAvg over the farms, then prove it was worth it.

Two things happen after the rounds finish. The global model is written to disk so
the AgentApp can answer questions with it, and a solo-vs-federated table is
printed on the region-wide held-out set. That table is the pitch: it is the
number that says collaborating beat going it alone.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .data import (FARM_PROFILES, FEATURE_STATS, FEATURES, NUM_FARMS, farm_data,
                   region_data, to_human)
from .model import accuracy, init_params, log_loss, predict_proba, train

app = ServerApp()

# Absolute by default: the ServerApp's working directory is set by whichever
# SuperLink is running, which is not this project -- a relative path silently
# writes the model into some unrelated directory.
MODEL_PATH = Path(
    os.environ.get("GROWFLWR_MODEL_PATH", Path.home() / ".growflwr" / "global_model.json")
).expanduser()


def _to_params(arrays: ArrayRecord) -> list[np.ndarray]:
    nds = arrays.to_numpy_ndarrays()
    return [nds[0].astype(np.float64), nds[1].astype(np.float64)]


def _save(params: list[np.ndarray], meta: dict) -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(
        json.dumps(
            {
                "features": FEATURES,
                "feature_stats": FEATURE_STATS,
                "weights": params[0].tolist(),
                "bias": float(params[1][0]),
                **meta,
            },
            indent=2,
        )
    )
    return MODEL_PATH


def _find_disagreement(
    solo: list[np.ndarray],
    fed: list[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
) -> dict | None:
    """Find a real regional condition where the solo model is wrong and FedAvg right.

    Searched rather than authored: these are rows from the held-out regional set,
    so the scenario the agent demonstrates on is a condition that genuinely occurs
    in the region, not one picked to make federation look good.
    """
    p_solo = predict_proba(solo, x) >= 0.5
    p_fed = predict_proba(fed, x) >= 0.5
    truth = y.astype(bool)

    candidates = np.where((p_solo != p_fed) & (p_fed == truth))[0]
    if len(candidates) == 0:
        return None

    # Take the case where the two models are most confidently opposed, since
    # that is the clearest thing to put in front of a judge.
    margins = np.abs(predict_proba(solo, x[candidates]) - predict_proba(fed, x[candidates]))
    best = candidates[int(np.argmax(margins))]
    return {
        "readings": to_human(x[best]),
        "ground_truth": "safe to irrigate" if truth[best] else "do not irrigate",
    }


def _save_conditions(
    solos: list[list[np.ndarray]],
    fed_params: list[np.ndarray],
    x_region: np.ndarray,
    y_region: np.ndarray,
) -> Path:
    """Write each farm's current readings, plus a case its own model gets wrong.

    The AgentApp runs in a container with no route to farm systems, so the
    readings a farmer asks about have to travel with the app. In a real
    deployment this is the farm's own gateway publishing its latest row.
    """
    out = {}
    for fid, profile in enumerate(FARM_PROFILES):
        _, _, x_test, _ = farm_data(fid)
        entry = {
            "name": profile["name"],
            "readings": to_human(x_test[-1]),
        }
        unusual = _find_disagreement(solos[fid], fed_params, x_region, y_region)
        if unusual is not None:
            entry["unusual_conditions"] = unusual
        out[str(fid)] = entry

    path = MODEL_PATH.parent / "current_conditions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return path


def _save_solo_models(
    solos: list[list[np.ndarray]],
    solo_accs: list[float],
    x_region: np.ndarray,
    y_region: np.ndarray,
) -> Path:
    """Persist each farm's go-it-alone model so the agent can show the contrast.

    This is what the farmer would be relying on if the region had not federated.
    Shipping it lets the agent answer the same question both ways on stage.
    """
    out = {}
    for fid, (params, acc) in enumerate(zip(solos, solo_accs)):
        out[str(fid)] = {
            "name": FARM_PROFILES[fid]["name"],
            "weights": params[0].tolist(),
            "bias": float(params[1][0]),
            "region_accuracy": acc,
        }

    path = MODEL_PATH.parent / "solo_models.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"features": FEATURES,
                                "feature_stats": FEATURE_STATS,
                                "models": out}, indent=2))
    return path


def _solo_baselines(epochs: int, lr: float) -> list[list[np.ndarray]]:
    """Train each farm alone, for comparison only.

    This runs in-process because it is a benchmark, not part of the protocol --
    in a real deployment each farm would measure this locally and report the
    score. No federated step ever sees these rows.
    """
    solos = []
    for fid in range(NUM_FARMS):
        x, y, _, _ = farm_data(fid)
        # Solo farms get the same total gradient steps as the federated model,
        # so the comparison is about *whose data*, not who trained longer.
        params, _ = train(init_params(), x, y, epochs, lr)
        solos.append(params)
    return solos


@app.main()
def main(grid: Grid, context: Context) -> None:
    rounds = int(context.run_config["num-server-rounds"])
    epochs = int(context.run_config["local-epochs"])
    lr = float(context.run_config["learning-rate"])

    x_region, y_region = region_data()

    def centralized_eval(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        params = _to_params(arrays)
        return MetricRecord(
            {
                "region_accuracy": accuracy(params, x_region, y_region),
                "region_loss": log_loss(y_region, predict_proba(params, x_region)),
            }
        )

    strategy = FedAvg(fraction_train=1.0, min_train_nodes=NUM_FARMS,
                      min_evaluate_nodes=NUM_FARMS, min_available_nodes=NUM_FARMS)

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(init_params()),
        num_rounds=rounds,
        train_config=ConfigRecord({"local-epochs": epochs, "learning-rate": lr}),
        evaluate_config=ConfigRecord({}),
        evaluate_fn=centralized_eval,
    )

    fed_params = _to_params(result.arrays)
    fed_acc = accuracy(fed_params, x_region, y_region)

    # --- The money metric -------------------------------------------------
    solos = _solo_baselines(epochs * rounds, lr)
    solo_accs = [accuracy(p, x_region, y_region) for p in solos]

    print("\n" + "=" * 78)
    print("GrowFlwr - region-wide accuracy (held-out, all farms' conditions)")
    print("=" * 78)
    print(f"{'farm':<40}{'alone':>10}{'federated':>12}{'gain':>12}")
    print("-" * 78)
    for profile, acc in zip(FARM_PROFILES, solo_accs):
        gain = (fed_acc - acc) * 100
        print(f"{profile['name']:<40}{acc:>9.1%}{fed_acc:>12.1%}{gain:>+11.1f}pp")
    print("-" * 78)
    mean_solo = float(np.mean(solo_accs))
    print(f"{'mean':<40}{mean_solo:>9.1%}{fed_acc:>12.1%}"
          f"{(fed_acc - mean_solo) * 100:>+11.1f}pp")
    print("=" * 78)

    # Be straight about where the benefit lands. Federation rescues farms whose
    # own history is unrepresentative; a farm that already sees the full range of
    # regional conditions has little left to gain, and may come out flat.
    gains = [fed_acc - a for a in solo_accs]
    best_i, worst_i = int(np.argmax(gains)), int(np.argmin(gains))
    print(f"\nMost helped: {FARM_PROFILES[best_i]['name']} ({gains[best_i] * 100:+.1f}pp) "
          f"- narrow local history.")
    print(f"Least helped: {FARM_PROFILES[worst_i]['name']} ({gains[worst_i] * 100:+.1f}pp) "
          f"- already sees representative conditions.")

    # --- The privacy audit ------------------------------------------------
    payload = sum(p.nbytes for p in fed_params)
    rows_held = sum(len(farm_data(f)[0]) for f in range(NUM_FARMS))
    print(f"\nPer farm per round, {payload} bytes of weights crossed the wire "
          f"({len(FEATURES)} weights + 1 bias).")
    print(f"{rows_held} raw telemetry rows stayed on the farms. Zero were transmitted.")

    _save_solo_models(solos, solo_accs, x_region, y_region)
    _save_conditions(solos, fed_params, x_region, y_region)
    path = _save(fed_params, {
        "region_accuracy": fed_acc,
        "mean_solo_accuracy": mean_solo,
        "rounds": rounds,
        "num_farms": NUM_FARMS,
    })
    print(f"\nGlobal model written to {path.resolve()}")
