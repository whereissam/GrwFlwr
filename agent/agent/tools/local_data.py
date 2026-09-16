"""Locally recorded farm data (soil sensors, crop, irrigation history).

These rows are the farm's own. They are used to score the federated model
in-process and are never transmitted -- the point of federating was that this
table stays on the farm.
"""

import json
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data"
FARMS_PATH = _DATA / "farms.json"
CONDITIONS_PATH = _DATA / "current_conditions.json"


def _conditions() -> dict:
    return json.loads(CONDITIONS_PATH.read_text())


def list_farms() -> dict:
    """Every farm and field this app knows about."""
    return {
        farm_id: [f["field_id"] for f in farm["fields"]]
        for farm_id, farm in _conditions().items()
    }


def get_local_farm_data(farm_id: str, field_id: str | None = None) -> dict:
    """Latest recorded state for one field, or the farm's first field."""
    conditions = _conditions()
    farm = conditions.get(farm_id)
    if farm is None:
        return {"error": f"No local data for farm_id={farm_id!r}. "
                         f"Known farms: {sorted(conditions)}"}

    fields = farm["fields"]
    if field_id is None:
        chosen = fields[0]
    else:
        matches = [f for f in fields if f["field_id"] == field_id]
        if not matches:
            return {"error": f"No field {field_id!r} on {farm_id}. "
                             f"Known fields: {[f['field_id'] for f in fields]}"}
        chosen = matches[0]

    return {"farm_id": farm_id, "field": chosen}


def driest_field(farm_id: str) -> dict:
    """The field with the lowest soil moisture — what a farmer asks about first."""
    conditions = _conditions()
    farm = conditions.get(farm_id)
    if farm is None:
        return {"error": f"No local data for farm_id={farm_id!r}"}
    chosen = min(farm["fields"], key=lambda f: float(f["soil_moisture_pct_nfk"]))
    return {"farm_id": farm_id, "field": chosen}


def weather_features(forecast: dict, field: dict) -> tuple[dict, str]:
    """The two regional weather values the model needs: et0_mm and rain_mm.

    Taken from the live forecast when it is available. The fallback is the
    field's own rain gauge reading and a seasonal-average ET0, and the caller is
    told which was used so the answer can say so rather than implying freshness.
    """
    et0 = _first(forecast.get("reference_evapotranspiration_mm"))
    rain = _first(forecast.get("precipitation_sum_mm"))

    if et0 is not None and rain is not None:
        return {"et0_mm": round(et0, 2), "rain_mm": round(rain, 2)}, \
               "live forecast (open-meteo)"

    # 3.78 mm/day is the season mean from the regional weather file; using it is
    # better than refusing to answer, but it is not a forecast.
    return {"et0_mm": 3.78,
            "rain_mm": float(field.get("onfarm_rain_gauge_mm", 0.0))}, \
           "seasonal average - live forecast unavailable"


def _first(seq):
    return seq[0] if isinstance(seq, list) and seq else None
