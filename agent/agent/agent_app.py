"""GrowFlwr: a Flower AgentApp that advises farmers on irrigation decisions.

Combines a live weather forecast, the team's federated-learning irrigation
model, and locally recorded field data, then lets the model explain the result.

Two things are deliberate. Context is gathered up front rather than through
model-driven tool calls: the lookups are deterministic, and streaming tool-call
events currently crash the Flower runtime's event handling. And every number in
the answer is computed before the model is called -- the model explains figures,
it does not produce them.
"""

import json
import os

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from openai import OpenAI

from .tools import (
    build_weather_forecast_url,
    compare_against_solo,
    driest_field,
    get_local_farm_data,
    parse_weather_forecast,
    predict_irrigation_need,
    water_plan,
    weather_features,
)

MODEL = "openai/gpt-5.6-sol"

# Each farm's approximate location, for the weather lookup. The dataset carries
# no coordinates; these place the region on the Canal d'Urgell, Lleida.
FARM_LOCATION = {
    "farmer_1": (41.62, 0.62),
    "farmer_2": (41.47, 0.86),
}

SYSTEM_PROMPT = """\
You are GrowFlwr, an irrigation advisor for farms sharing one water region.

You are given a DATA block: one field's latest sensor readings, a weather
forecast, a Low/Medium/High need classification from a model trained
federatively across every farm in the region, and a water plan. Those numbers
are already computed and are the only quantitative facts you may state.

Rules:
- Lead with the need class and what to do, in one sentence a busy farmer can act
  on. Give the volume in cubic metres, mentioning litres only if it helps.
- Justify it from the readings given, naming actual values.
- Use the pre-formatted percentage strings exactly as given.
- The water plan is an agronomic rule, not a model prediction. Do not describe
  the volume as something the model predicted.
- If `weather_source` says the forecast was unavailable, say so plainly rather
  than implying the numbers are fresh.
- If the models disagree, say what this farm's own model would have advised and
  why the federated one is more trustworthy here: it learned from conditions
  this farm rarely sees. If they agree, do not mention the comparison.
- Never invent a reading. Five sentences at most, no bullet lists.
"""

app = AgentApp()


def _safe_call(label: str, func) -> dict:
    """Run a lookup and never let it take down the whole run."""
    try:
        return func()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, don't crash
        print(f"{label} lookup failed: {exc!r}")
        return {"error": f"{label} unavailable: {exc}"}


def _web_fetch(agent: AgentSession, call_id: str, url: str) -> str:
    """Fetch a URL through the platform's web_fetch connector.

    Direct outbound HTTP from AgentApp code is blocked by the run sandbox's
    egress proxy; web_fetch is the sanctioned path to the public internet.
    """
    result = agent.connectors.call(
        {"name": "web_fetch", "call_id": call_id, "arguments": {"url": url}}
    )
    return json.loads(result["output"])["content"]


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Answer one irrigation question for one field."""
    question = context.run_config.get("agent.input")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("agent.input must be a non-empty string")

    farm_id = str(context.run_config.get("agent.farm_id", "farmer_1"))
    field_id = str(context.run_config.get("agent.field_id", "")).strip()

    # No field named: answer about the driest one, which is what a farmer
    # walking the property would ask about first.
    located = (get_local_farm_data(farm_id, field_id) if field_id
               else driest_field(farm_id))
    if "error" in located:
        print(f"Cannot answer: {located['error']}")
        raise ValueError(located["error"])
    field = located["field"]

    lat, lon = FARM_LOCATION.get(farm_id, (41.62, 0.62))
    forecast = _safe_call(
        "weather",
        lambda: parse_weather_forecast(
            _web_fetch(agent, "weather-fetch", build_weather_forecast_url(lat, lon))
        ),
    )
    weather, weather_source = weather_features(
        forecast if "error" not in forecast else {}, field
    )

    prediction = _safe_call("FL model", lambda: predict_irrigation_need(field, weather))
    counterfactual = _safe_call(
        "solo comparison", lambda: compare_against_solo(farm_id, field, weather)
    )
    plan = _safe_call(
        "water plan",
        lambda: water_plan(field, weather, prediction.get("need_class", 0)),
    )

    _print_header(farm_id, field, weather_source, prediction, counterfactual, plan,
                  question)

    data = {
        "farm_id": farm_id,
        "field": field,
        "weather_forecast": forecast,
        "weather_source": weather_source,
        "weather_used_by_model": weather,
        "fl_model_prediction": prediction,
        "counterfactual": counterfactual,
        "water_plan": plan,
    }

    client = OpenAI(
        base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
        api_key=os.environ["FLWR_RUNTIME_API_KEY"],
        max_retries=0,
    )
    stream = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"DATA:\n{json.dumps(data, indent=2)}\n\n"
                                        f"Farmer question: {question.strip()}"},
        ],
        stream=True,
    )

    output_text = []
    for event in stream:
        agent.events.emit(event.to_dict())
        if event.type in {"error", "response.failed"}:
            raise RuntimeError(f"Model response failed: {event}")
        if event.type == "response.output_text.delta":
            output_text.append(event.delta)

    print("".join(output_text))


def _print_header(farm_id, field, weather_source, prediction, counterfactual, plan,
                  question):
    """Show the decision both ways before the model speaks, for the operator."""
    print(f"--- {farm_id} / {field['field_id']} "
          f"({field['crop_type']}, {field['growth_stage']}) ---")
    print(f"soil moisture {field['soil_moisture_pct_nfk']}% nFK, "
          f"{field['days_since_last_irrigation']}d since irrigation, "
          f"{field['field_area_ha']} ha")
    print(f"weather: {weather_source}")

    if "error" not in counterfactual:
        alone, fed = counterfactual["this_farm_alone"], counterfactual["federated"]
        print(f"{'this farm alone:':<20}{alone['need']:<8}"
              f"({alone['confidence_pct']:>4}, macro-F1 {alone['macro_f1']})")
        print(f"{'federated:':<20}{fed['need']:<8}"
              f"({fed['confidence_pct']:>4}, macro-F1 {fed['macro_f1']})")
        if counterfactual["models_disagree"]:
            print(">>> THEY DISAGREE")

    if "error" not in plan and plan.get("cubic_metres"):
        print(f"{'>>> PLAN':<20}{plan['action']} - {plan['depth_mm']} mm = "
              f"{plan['cubic_metres']:,} m3")
    elif "error" not in plan:
        print(f"{'>>> PLAN':<20}{plan['action']}")
    print(f"\nquestion: {question.strip()}\n")
