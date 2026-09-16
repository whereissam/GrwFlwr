"""A Flower AgentApp that advises farmers on irrigation decisions.

Combines a free weather forecast, the team's federated-learning irrigation
model, and locally recorded farm data, then lets the model reason over all
three to answer the farmer's question.
"""

import json
import os

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from openai import OpenAI

from .tools import (
    build_fl_model_url,
    build_weather_forecast_url,
    get_local_farm_data,
    heuristic_irrigation_prediction,
    parse_fl_model_response,
    parse_weather_forecast,
)

MODEL = "openai/gpt-5.6-sol"

SYSTEM_PROMPT = """\
You are an irrigation advisor for a smallholder farm. You are given the \
latest weather forecast, the federated-learning model's irrigation \
prediction, and locally recorded farm data as JSON. Use all three to give a \
clear, actionable recommendation (irrigate now / wait / how much water) and \
briefly explain why, citing the data you used. Keep the answer concise and \
farmer-friendly. If a data source's JSON contains an "error" field, note \
that it was unavailable and reason from whatever data you do have.\
"""

app = AgentApp()


def _safe_call(label: str, func) -> dict:
    """Run a data lookup and never let it take down the whole run."""
    try:
        return func()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, don't crash the run
        print(f"{label} lookup failed: {exc!r}")
        return {"error": f"{label} unavailable: {exc}"}


def _web_fetch(agent: AgentSession, call_id: str, url: str) -> str:
    """Fetch a URL's raw response text through the platform's web_fetch connector.

    Direct outbound HTTP from AgentApp code is blocked by the SuperGrid run
    sandbox's egress policy; web_fetch is the sanctioned path to the public
    internet, so all external lookups go through it instead of `requests`.
    """
    result = agent.connectors.call(
        {"name": "web_fetch", "call_id": call_id, "arguments": {"url": url}}
    )
    output = json.loads(result["output"])
    return output["content"]


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Answer a farmer's irrigation question using weather, FL, and local data."""
    question = context.run_config.get("agent.input")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("agent.input must be a non-empty string")

    farm_id = context.run_config.get("agent.farm_id", "farm-001")
    latitude = context.run_config.get("agent.latitude", 52.52)
    longitude = context.run_config.get("agent.longitude", 13.405)

    # Gather context up front rather than via model-driven tool calls: these
    # are deterministic lookups, so there's no need for the model to decide
    # whether/when to call them, and streaming tool-call events currently
    # crash the Flower runtime's event handling (server-side bug, not ours).
    # Each lookup is wrapped so one failing source degrades gracefully
    # instead of taking down the whole run.
    weather = _safe_call(
        "weather",
        lambda: parse_weather_forecast(
            _web_fetch(agent, "weather-fetch", build_weather_forecast_url(latitude, longitude))
        ),
    )
    farm_data = _safe_call("local farm data", lambda: get_local_farm_data(farm_id))

    fl_model_url = build_fl_model_url(
        farm_id,
        soil_moisture_pct=farm_data.get("soil_moisture_pct"),
        forecast_precipitation_mm=next(iter(weather.get("precipitation_sum_mm", [])), None),
    )
    if fl_model_url is not None:
        fl_prediction = _safe_call(
            "FL model",
            lambda: parse_fl_model_response(_web_fetch(agent, "fl-model-fetch", fl_model_url)),
        )
    else:
        fl_prediction = heuristic_irrigation_prediction(
            soil_moisture_pct=farm_data.get("soil_moisture_pct"),
            forecast_precipitation_mm=next(iter(weather.get("precipitation_sum_mm", [])), None),
        )

    client = OpenAI(
        base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
        api_key=os.environ["FLWR_RUNTIME_API_KEY"],
        max_retries=0,
    )

    context_json = json.dumps(
        {
            "weather_forecast": weather,
            "local_farm_data": farm_data,
            "fl_model_prediction": fl_prediction,
        },
        indent=2,
    )
    input_items = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Farm ID: {farm_id}\n"
                f"Location: {latitude}, {longitude}\n\n"
                f"Data:\n{context_json}\n\n"
                f"Farmer question: {question.strip()}"
            ),
        },
    ]

    stream = client.responses.create(model=MODEL, input=input_items, stream=True)

    output_text = []
    for event in stream:
        agent.events.emit(event.to_dict())
        if event.type in {"error", "response.failed"}:
            raise RuntimeError(f"Model response failed: {event}")
        if event.type == "response.output_text.delta":
            output_text.append(event.delta)

    print("".join(output_text))
