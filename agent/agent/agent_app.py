"""GrowFlwr: a Flower AgentApp that advises farmers on irrigation decisions.

Combines a live weather forecast, the team's federated-learning irrigation
model, and locally recorded farm data, then lets the model explain the result.

Two things are deliberate. Context is gathered up front rather than through
model-driven tool calls: the lookups are deterministic, and streaming tool-call
events currently crash the Flower runtime's event handling. And every number in
the answer is computed here before the model is called -- the model's job is to
explain figures, not to produce them.
"""

import json
import os

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from openai import OpenAI

from .tools import (
    ModelUnavailable,
    build_model_readings,
    build_weather_forecast_url,
    compare_against_solo,
    get_local_farm_data,
    parse_weather_forecast,
    predict_irrigation_safety,
    water_budget,
)

MODEL = "openai/gpt-5.6-sol"

SYSTEM_PROMPT = """\
You are GrowFlwr, an irrigation advisor for farms sharing one water region.

You are given a DATA block: the farm's own sensor readings, a live weather
forecast, a safety prediction from a model trained federatively across every
farm in the region, and a water budget. Those numbers are already computed and
are the only quantitative facts you may state.

Rules:
- Lead with the decision and the confidence, in one sentence a busy farmer can
  act on.
- Justify it from the drivers given, naming the actual readings.
- State the water budget in litres per hectare.
- Use the pre-formatted percentage strings exactly as given.
- If a data source contains an "error" field, or the weather source says values
  are stored rather than live, say so plainly instead of implying it was fresh.
- If the models disagree, say what this farm's own model would have advised and
  why the federated one is more trustworthy here: it learned from conditions
  this farm rarely sees. If they agree, do not mention the comparison.
- Never invent a reading or a forecast. Five sentences at most, no bullet lists.
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
    """Answer one irrigation question for one farm."""
    question = context.run_config.get("agent.input")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("agent.input must be a non-empty string")
    farm_id = str(context.run_config.get("agent.farm_id", "farm-001"))
    scenario = str(context.run_config.get("agent.scenario", "today"))

    farm = get_local_farm_data(farm_id)
    if "error" in farm:
        # A farm we have no data for is reported as such. It is never worth
        # letting the model guess at a soil reading.
        print(f"Cannot answer: {farm['error']}")
        raise ValueError(farm["error"])

    if scenario == "unusual":
        # A real regional condition this farm's own model gets wrong, found in
        # held-out data during training rather than authored for the demo.
        unusual = farm.get("unusual_conditions")
        if not unusual:
            raise ValueError(f"No unusual-conditions case recorded for {farm_id}.")
        readings, weather_source = unusual["readings"], "recorded scenario"
        weather = {"note": "scenario uses recorded conditions, not a live forecast"}
        known_answer = unusual["ground_truth"]
    else:
        weather = _safe_call(
            "weather",
            lambda: parse_weather_forecast(
                _web_fetch(agent, "weather-fetch",
                           build_weather_forecast_url(farm["latitude"], farm["longitude"]))
            ),
        )
        readings, weather_source = build_model_readings(farm, weather)
        known_answer = None

    prediction = _safe_call("FL model", lambda: predict_irrigation_safety(readings))
    counterfactual = _safe_call("solo comparison", lambda: compare_against_solo(farm_id, readings))
    budget = _safe_call("water budget", lambda: water_budget(readings))

    _print_header(farm, scenario, weather_source, counterfactual, known_answer, question)

    data = {
        "farm": {k: farm[k] for k in ("name", "crop", "soil_type", "irrigation_system")},
        "scenario": scenario,
        "weather_forecast": weather,
        "weather_source": weather_source,
        "model_readings": readings,
        "fl_model_prediction": prediction,
        "counterfactual": counterfactual,
        "water_budget": budget,
    }
    if known_answer:
        data["known_correct_answer"] = known_answer

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


def _print_header(farm, scenario, weather_source, counterfactual, known_answer, question):
    """Show the decision both ways before the model speaks, for the operator."""
    print(f"--- {farm['name']}  [{scenario}] ---")
    print(f"weather: {weather_source}")
    if "error" not in counterfactual:
        alone, fed = counterfactual["this_farm_alone"], counterfactual["federated"]
        print(f"{'this farm alone:':<20}{alone['decision']:<18}"
              f"({alone['confidence_pct']:>4} confident, {alone['region_accuracy_pct']} accuracy)")
        print(f"{'federated:':<20}{fed['decision']:<18}"
              f"({fed['confidence_pct']:>4} confident, {fed['region_accuracy_pct']} accuracy)")
        if counterfactual["models_disagree"]:
            print(f"{'>>> THEY DISAGREE':<20}"
                  + (f"correct answer: {known_answer}" if known_answer else ""))
    print(f"\nquestion: {question.strip()}\n")
