"""Weather forecast via the free Open-Meteo API.

Direct outbound HTTP from AgentApp code is blocked by the SuperGrid run
sandbox's egress policy (only the platform's own connectors, like
`web_fetch`, may reach the public internet). So this module only builds the
URL and parses the response; the actual fetch is done by the caller through
`agent.connectors.call({"name": "web_fetch", ...})` and the raw response
text is passed in.
"""

import json

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def build_weather_forecast_url(latitude: float, longitude: float, forecast_days: int = 3) -> str:
    """Build the Open-Meteo forecast URL for a location."""
    days = max(1, min(forecast_days, 16))
    query = (
        f"latitude={latitude}&longitude={longitude}&"
        "daily=precipitation_sum,precipitation_probability_max,"
        "temperature_2m_max,et0_fao_evapotranspiration&"
        f"forecast_days={days}&timezone=auto"
    )
    return f"{OPEN_METEO_URL}?{query}"


def parse_weather_forecast(response_text: str) -> dict:
    """Parse a raw Open-Meteo JSON response into a compact summary."""
    daily = json.loads(response_text).get("daily", {})
    return {
        "source": "open-meteo.com",
        "days": daily.get("time", []),
        "precipitation_sum_mm": daily.get("precipitation_sum", []),
        "precipitation_probability_max_pct": daily.get("precipitation_probability_max", []),
        "temperature_max_c": daily.get("temperature_2m_max", []),
        "reference_evapotranspiration_mm": daily.get("et0_fao_evapotranspiration", []),
    }
