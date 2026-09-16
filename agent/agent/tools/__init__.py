"""Data sources the watering agent uses: weather, local farm data, FL model."""

from .fl_model import (
    ModelUnavailable,
    compare_against_solo,
    predict_irrigation_need,
    water_plan,
)
from .local_data import (
    driest_field,
    get_local_farm_data,
    list_farms,
    weather_features,
)
from .weather import build_weather_forecast_url, parse_weather_forecast

__all__ = [
    "ModelUnavailable",
    "build_weather_forecast_url",
    "compare_against_solo",
    "driest_field",
    "get_local_farm_data",
    "list_farms",
    "parse_weather_forecast",
    "predict_irrigation_need",
    "water_plan",
    "weather_features",
]
