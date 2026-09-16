"""Locally recorded farm data (soil sensors, crop, irrigation history).

These readings are the farm's own. They are used to score the federated model
in-process and are never transmitted -- the whole point of federating was that
this table stays on the farm.
"""

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "farms.json"


def get_local_farm_data(farm_id: str) -> dict:
    """Look up locally recorded data for a farm by id."""
    farms = json.loads(DATA_PATH.read_text())
    farm = farms.get(farm_id)
    if farm is None:
        return {"error": f"No local data for farm_id={farm_id!r}. "
                         f"Known farms: {sorted(farms)}"}
    return farm


def list_farms() -> dict:
    """Every farm this app knows about, for the operator."""
    return {
        fid: {"name": f["name"], "crop": f["crop"]}
        for fid, f in json.loads(DATA_PATH.read_text()).items()
    }


def build_model_readings(farm: dict, weather: dict) -> tuple[dict, str]:
    """Assemble the seven features the federated model needs.

    Four come from the farm's own sensors, three from the weather forecast.
    Returns the readings and a note on where the weather half came from, so the
    answer can say plainly whether it used a live forecast or stored values.
    """
    readings = dict(farm["farm_measured"])

    rain = _first(weather.get("precipitation_sum_mm"))
    rain_day2 = _second(weather.get("precipitation_sum_mm"))
    et0 = _first(weather.get("reference_evapotranspiration_mm"))
    temp = _first(weather.get("temperature_max_c"))

    if None not in (rain, et0, temp):
        readings["rain_forecast_48h_mm"] = round(rain + (rain_day2 or 0.0), 2)
        readings["et0_mm_day"] = round(et0, 2)
        readings["temp_c"] = round(temp, 2)
        return readings, "live forecast (open-meteo)"

    readings.update(farm["weather_fallback"])
    return readings, "stored values - live forecast unavailable"


def _first(seq):
    return seq[0] if isinstance(seq, list) and seq else None


def _second(seq):
    return seq[1] if isinstance(seq, list) and len(seq) > 1 else None
