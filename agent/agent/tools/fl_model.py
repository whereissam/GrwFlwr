"""The team's federated-learning irrigation model.

Replaces the earlier heuristic placeholder. The model is produced by
`federated/` (FedAvg across four farms) and ships inside the FAB as JSON: the
AgentApp runs in a remote container with no route to the laptop that trained it.

Scoring happens in-process. There is no inference endpoint to call, so none of
the egress restrictions apply here, and a farmer's readings never leave the app.
"""

import json
import math
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data"


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


def _score(weights: list[float], bias: float, stats: dict, features: list[str],
           readings: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Return (probability, per-feature contribution) for one set of readings."""
    missing = [f for f in features if f not in readings]
    if missing:
        raise ModelUnavailable(f"Readings missing required fields: {missing}")

    logit = float(bias)
    contributions = {}
    for name, weight in zip(features, weights):
        mean, std = stats[name]
        term = weight * ((float(readings[name]) - mean) / std)
        contributions[name] = round(term, 3)
        logit += term
    return 1.0 / (1.0 + math.exp(-logit)), contributions


def predict_irrigation_safety(readings: dict[str, float]) -> dict:
    """Score readings with the federated global model."""
    model = _load("global_model.json")
    probability, contributions = _score(
        model["weights"], model["bias"], model["feature_stats"],
        model["features"], readings,
    )
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)

    return {
        "source": f"federated model, FedAvg over {model['num_farms']} farms",
        "decision": "safe to irrigate" if probability >= 0.5 else "do not irrigate",
        "confidence_pct": f"{probability:.0%}",
        "probability_safe": round(probability, 3),
        "region_accuracy_pct": f"{model['region_accuracy']:.1%}",
        "top_drivers": [
            {
                "feature": name,
                "reading": readings[name],
                "effect": "supports irrigating" if term > 0 else "argues against",
            }
            for name, term in ranked[:3]
        ],
    }


def compare_against_solo(farm_id: str, readings: dict[str, float]) -> dict:
    """Score the same readings with this farm's go-it-alone model.

    The counterfactual: what this farmer would have been told if the region had
    never federated. Identical inputs, so any difference is down to whose data
    trained the model.
    """
    solo_file = _load("solo_models.json")
    index = str(int(farm_id.split("-")[-1]) - 1)
    if index not in solo_file["models"]:
        raise ModelUnavailable(f"No solo model for {farm_id}.")

    solo = solo_file["models"][index]
    probability, _ = _score(
        solo["weights"], solo["bias"], solo_file["feature_stats"],
        solo_file["features"], readings,
    )
    decision = "safe to irrigate" if probability >= 0.5 else "do not irrigate"
    federated = predict_irrigation_safety(readings)

    return {
        "this_farm_alone": {
            "decision": decision,
            "confidence_pct": f"{probability:.0%}",
            "region_accuracy_pct": f"{solo['region_accuracy']:.1%}",
        },
        "federated": {
            "decision": federated["decision"],
            "confidence_pct": federated["confidence_pct"],
            "region_accuracy_pct": federated["region_accuracy_pct"],
        },
        "models_disagree": decision != federated["decision"],
    }


def water_budget(readings: dict[str, float], horizon_days: float = 2.0) -> dict:
    """Litres per hectare needed over the horizon, after forecast rain.

    Crop water use is reference evapotranspiration scaled by the crop
    coefficient; rain already forecast is subtracted, because irrigating on top
    of it is the waste this system exists to prevent.
    """
    demand_mm = readings["et0_mm_day"] * readings["crop_coefficient"] * horizon_days
    deficit_mm = max(0.0, demand_mm - readings["rain_forecast_48h_mm"])
    return {
        "horizon_days": horizon_days,
        "crop_demand_mm": round(demand_mm, 1),
        "rain_expected_mm": round(readings["rain_forecast_48h_mm"], 1),
        "net_deficit_mm": round(deficit_mm, 1),
        "litres_per_hectare": round(deficit_mm * 10_000),
    }


def recommendation(prediction: dict, budget: dict) -> dict:
    """Reconcile "is it safe" with "is it needed" into one instruction.

    These are different questions and they can disagree: the model can rate
    irrigating safe while the forecast rain already covers crop demand, which
    would otherwise produce the useless advice "irrigate, 0 litres". Need is
    checked first, because water not needed is water not spent.
    """
    if "error" in prediction or "error" in budget:
        return {"action": "unavailable",
                "reason": "a required figure could not be computed"}

    if budget["net_deficit_mm"] <= 0.0:
        return {
            "action": "no need to irrigate",
            "reason": (f"forecast rain of {budget['rain_expected_mm']} mm already "
                       f"covers crop demand of {budget['crop_demand_mm']} mm"),
            "litres_per_hectare": 0,
        }

    if prediction["probability_safe"] < 0.5:
        return {
            "action": "do not irrigate",
            "reason": (f"the federated model rates irrigating safe at only "
                       f"{prediction['confidence_pct']}"),
            "litres_per_hectare": 0,
        }

    return {
        "action": "irrigate",
        "reason": (f"crop demand exceeds forecast rain by "
                   f"{budget['net_deficit_mm']} mm and the federated model rates "
                   f"this safe at {prediction['confidence_pct']}"),
        "litres_per_hectare": budget["litres_per_hectare"],
    }
