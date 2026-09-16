"""The team's federated-learning irrigation model.

Produced by `federated/` (FedAvg across four farms) and shipped inside the FAB:
the AgentApp runs in a remote container with no route to the machine that
trained it. Scoring happens in-process, so a farmer's readings never leave.

The encoding here must match `federated/growflwr/data.py` exactly -- same
column order, vocabularies, normalization constants and interaction terms. All
of that is read from the model JSON rather than duplicated as constants here,
and the feature count is checked, so the two cannot drift apart silently.
"""

import json
import math
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data"

_MAX_APPLICATION_MM = 30.0
_LITRES_PER_MM_HA = 10_000.0


class ModelUnavailable(RuntimeError):
    """The model artifacts are not bundled in this build."""


def _load(name: str) -> dict:
    path = _DATA / name
    if not path.exists():
        raise ModelUnavailable(
            f"{name} missing. Run the federated training in ../federated, then "
            f"./scripts/sync-model.sh, before building the FAB."
        )
    return json.loads(path.read_text())


def _encode(spec: dict, field: dict, weather: dict) -> list[float]:
    """Mirror of the training-time encoder, driven by the model's own spec."""
    numeric, categorical = spec["numeric"], spec["categorical"]
    stats = spec["feature_stats"]

    vec: list[float] = []
    for name in numeric:
        raw = weather[name] if name in weather else field[name]
        mean, std = stats[name]
        vec.append((float(raw) - mean) / std)

    blocks: dict[str, int] = {}
    for col, values in categorical.items():
        blocks[col] = len(vec)
        onehot = [0.0] * len(values)
        if field.get(col) in values:
            onehot[values.index(field[col])] = 1.0
        vec.extend(onehot)

    moisture = vec[numeric.index("soil_moisture_pct_nfk")]
    for col in ("crop_type", "growth_stage"):
        start, size = blocks[col], len(categorical[col])
        vec.extend(moisture * v for v in vec[start:start + size])

    expected = len(spec["feature_names"])
    if len(vec) != expected:
        raise ModelUnavailable(
            f"Encoded {len(vec)} features but the model expects {expected}. The "
            f"agent and the training code have drifted apart -- re-run "
            f"scripts/sync-model.sh."
        )
    return vec


def _softmax(z: list[float]) -> list[float]:
    top = max(z)
    e = [math.exp(v - top) for v in z]
    total = sum(e)
    return [v / total for v in e]


def _score(spec: dict, weights, bias, field: dict, weather: dict) -> list[float]:
    x = _encode(spec, field, weather)
    logits = [
        bias[c] + sum(x[i] * weights[i][c] for i in range(len(x)))
        for c in range(len(bias))
    ]
    return _softmax(logits)


def predict_irrigation_need(field: dict, weather: dict) -> dict:
    """Classify one field as Low / Medium / High irrigation need."""
    model = _load("global_model.json")
    probs = _score(model, model["weights"], model["bias"], field, weather)
    idx = max(range(len(probs)), key=probs.__getitem__)
    names = model["class_names"]

    return {
        "source": f"federated model, FedAvg over {model['num_farms']} farms",
        "need": names[str(idx)],
        "need_class": idx,
        "confidence_pct": f"{probs[idx]:.0%}",
        "probabilities": {names[str(i)]: round(p, 3) for i, p in enumerate(probs)},
        "region_macro_f1": round(model["region_macro_f1"], 3),
        "region_high_recall_pct": f"{model['region_high_recall']:.0%}",
    }


def compare_against_solo(farm_id: str, field: dict, weather: dict) -> dict:
    """Score the same field with this farm's go-it-alone model.

    The counterfactual: what this farmer would have been told if the region had
    never federated. Identical inputs, so any difference is down to whose data
    trained the model.
    """
    solo_file = _load("solo_models.json")
    index = str(int(farm_id.split("_")[-1]) - 1)
    if index not in solo_file["models"]:
        raise ModelUnavailable(f"No solo model for {farm_id}.")

    solo = solo_file["models"][index]
    probs = _score(solo_file, solo["weights"], solo["bias"], field, weather)
    idx = max(range(len(probs)), key=probs.__getitem__)
    names = solo_file["class_names"]
    federated = predict_irrigation_need(field, weather)

    return {
        "this_farm_alone": {
            "need": names[str(idx)],
            "confidence_pct": f"{probs[idx]:.0%}",
            "macro_f1": round(solo["macro_f1"], 3),
        },
        "federated": {
            "need": federated["need"],
            "confidence_pct": federated["confidence_pct"],
            "macro_f1": federated["region_macro_f1"],
        },
        "models_disagree": names[str(idx)] != federated["need"],
    }


def water_plan(field: dict, weather: dict, need_class: int) -> dict:
    """Turn a class into litres for this specific field.

    Depth replaces what the crop has used since the last irrigation --
    evapotranspiration times days elapsed, less what the on-farm rain gauge
    caught -- capped at a practical single application. This is a stated
    agronomic rule, not a model output, and is labelled as such.
    """
    if need_class == 0:
        return {"action": "no irrigation needed", "litres": 0, "depth_mm": 0.0,
                "basis": "the model classifies need as Low"}

    days = float(field["days_since_last_irrigation"])
    et0 = float(weather.get("et0_mm", 0.0))
    rain = float(field.get("onfarm_rain_gauge_mm", 0.0))

    depth = max(0.0, min(et0 * days - rain, _MAX_APPLICATION_MM))
    if need_class == 1:
        depth *= 0.6  # Medium: a holding dose, not a full refill

    area = float(field["field_area_ha"])
    return {
        "action": "irrigate" if depth > 0 else "no irrigation needed",
        "depth_mm": round(depth, 1),
        "field_area_ha": area,
        "litres": round(depth * area * _LITRES_PER_MM_HA),
        # Cubic metres too: a 42 ha field needs millions of litres, which is a
        # number nobody can read at a glance.
        "cubic_metres": round(depth * area * _LITRES_PER_MM_HA / 1000),
        "basis": (f"{et0} mm/day evapotranspiration over {days:.0f} days since last "
                  f"irrigation, less {rain} mm measured rain, capped at "
                  f"{_MAX_APPLICATION_MM:.0f} mm"),
        "water_source": field.get("water_source"),
        "irrigation_type": field.get("irrigation_type"),
    }
