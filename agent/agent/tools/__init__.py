"""Data sources the irrigation agent uses: weather, local farm data, FL model."""

from .fl_model import (
    ModelUnavailable,
    compare_against_solo,
    predict_irrigation_safety,
    recommendation,
    water_budget,
)
from .local_data import build_model_readings, get_local_farm_data, list_farms
from .weather import build_weather_forecast_url, parse_weather_forecast

__all__ = [
    "ModelUnavailable",
    "build_model_readings",
    "build_weather_forecast_url",
    "compare_against_solo",
    "get_local_farm_data",
    "list_farms",
    "parse_weather_forecast",
    "predict_irrigation_safety",
    "recommendation",
    "water_budget",
]
