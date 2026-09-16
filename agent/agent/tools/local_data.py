"""Locally recorded farm data (soil sensors, crop, irrigation history)."""

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "farms.json"


def get_local_farm_data(farm_id: str) -> dict:
    """Look up locally recorded data for a farm by id."""
    farms = json.loads(DATA_PATH.read_text())
    farm = farms.get(farm_id)
    if farm is None:
        return {"error": f"No local data for farm_id={farm_id!r}"}
    return farm
