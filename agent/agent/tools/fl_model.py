"""Client for the team's federated-learning watering model.

TODO(fl-team): set FL_MODEL_URL to the real SuperGrid inference endpoint once
the federated model is deployed. Until then this falls back to a simple
heuristic so the agent can be demoed end-to-end without the model.

The endpoint must accept a GET request with query parameters and return a
JSON object: direct outbound HTTP from AgentApp code is blocked by the
SuperGrid run sandbox's egress policy, so predictions are fetched through the
platform's `web_fetch` connector, which only supports GET-style fetches (no
arbitrary POST bodies).
"""

import json
import os
from urllib.parse import urlencode

FL_MODEL_URL = os.environ.get("FL_MODEL_URL")


def build_fl_model_url(
    farm_id: str,
    soil_moisture_pct: float | None = None,
    forecast_precipitation_mm: float | None = None,
) -> str | None:
    """Build the FL model's prediction URL, or None if FL_MODEL_URL is unset."""
    if not FL_MODEL_URL:
        return None
    params: dict[str, str] = {"farm_id": farm_id}
    if soil_moisture_pct is not None:
        params["soil_moisture_pct"] = str(soil_moisture_pct)
    if forecast_precipitation_mm is not None:
        params["forecast_precipitation_mm"] = str(forecast_precipitation_mm)
    return f"{FL_MODEL_URL}?{urlencode(params)}"


def parse_fl_model_response(response_text: str) -> dict:
    """Parse the FL model's raw JSON response."""
    return json.loads(response_text)


def heuristic_irrigation_prediction(
    soil_moisture_pct: float | None = None,
    forecast_precipitation_mm: float | None = None,
) -> dict:
    """Fallback prediction used until the real FL model is reachable."""
    moisture = 20.0 if soil_moisture_pct is None else soil_moisture_pct
    rain = 0.0 if forecast_precipitation_mm is None else forecast_precipitation_mm
    score = max(0.0, min(1.0, (30.0 - moisture) / 30.0 - rain / 50.0))
    return {
        "source": "heuristic-fallback (FL_MODEL_URL not set)",
        "irrigation_need_score": round(score, 2),
        "recommended_irrigation_mm": round(score * 15, 1),
    }
