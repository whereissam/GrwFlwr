"""GrwFlwr: a Flower AgentApp that advises farmers on watering decisions.

Combines a live weather forecast, the team's federated-learning irrigation
model, and locally recorded field data, then lets the model explain the result.

Two things are deliberate. Context is gathered up front rather than through
model-driven tool calls: the lookups are deterministic, and streaming tool-call
events currently crash the Flower runtime's event handling. And every number in
the answer is computed before the model is called -- the model explains figures,
it does not produce them.
"""

import datetime
import json
import os
from zoneinfo import ZoneInfo

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

MODEL = "flower-endeavor-v1.0"

# Each farm's approximate location, for the weather lookup. The dataset carries
# no coordinates; these place the region on the Canal d'Urgell, Lleida.
FARM_LOCATION = {
    "farmer_1": (41.62, 0.62),
    "farmer_2": (41.47, 0.86),
}

SYSTEM_PROMPT = """\
You are GrwFlwr, a watering advisor for farms sharing one water region.

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


def _next_daily_run_at(hour: int, minute: int, tz: str) -> str:
    """Return the next occurrence of hour:minute in tz as an ISO 8601 timestamp."""
    now = datetime.datetime.now(ZoneInfo(tz))
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += datetime.timedelta(days=1)
    return candidate.isoformat()


def _setup_daily_automation(agent: AgentSession, context: Context) -> None:
    """One-off setup: register a daily recurring run via Flower's automation connector.

    Triggered by `agent.action="schedule"`. This isn't model-driven (see
    `_web_fetch`'s docstring on why: streaming tool-call events currently
    crash the runtime), so it's a direct connector call instead of something
    the model decides to do.
    """
    start_at = _next_daily_run_at(
        hour=int(context.run_config.get("agent.schedule_hour", 16)),
        minute=0,
        tz=context.run_config.get("agent.timezone", "Europe/Berlin"),
    )
    result = agent.connectors.call(
        {
            "name": "start_automation",
            "call_id": "daily-irrigation-schedule",
            "arguments": {
                "input": "Should I water today?",
                "start_at": start_at,
                "fixed_interval": 86400,
            },
        }
    )
    print(f"Daily watering check scheduled starting {start_at}: {result}")


def _maybe_schedule_from_question(
    agent: AgentSession, client: OpenAI, question: str, default_start_at: str
) -> dict | None:
    """Let the model decide, from the farmer's own words, whether to schedule
    a recurring daily check — and if so, register it for real.

    Uses a single non-streaming tool-call turn rather than the app's usual
    streamed calls: per `_web_fetch`'s docstring, streaming tool-call *events*
    are what crashes the Flower runtime server-side, so this sidesteps that
    by never streaming a turn where a tool call is on the table.
    """
    tools = agent.connectors.tools(["start_automation"])
    allowed_tool_names = {tool["name"] for tool in tools if isinstance(tool.get("name"), str)}
    if "start_automation" not in allowed_tool_names:
        return None

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "If, and only if, the farmer is explicitly asking to set "
                    "up a recurring/automated watering check (e.g. 'do this "
                    "every day at 4pm'), call start_automation with "
                    'input="Should I water today?", fixed_interval=86400 '
                    f"for daily recurrence, and start_at={default_start_at!r} "
                    "unless the farmer clearly asked for a different time (in "
                    "which case compute the next matching ISO 8601 timestamp "
                    "with a timezone offset). Otherwise, do not call any tool "
                    "— most questions are not scheduling requests."
                ),
            },
            {"role": "user", "content": question},
        ],
        tools=tools,
        tool_choice="auto",
        stream=False,
    )
    for item in response.output:
        if getattr(item, "type", None) == "function_call" and item.name == "start_automation":
            if item.name not in allowed_tool_names:
                raise RuntimeError(f"Tool {item.name!r} was not exposed")
            arguments = json.loads(item.arguments)
            result = agent.connectors.call(
                {
                    "name": "start_automation",
                    "call_id": item.call_id,
                    "arguments": arguments,
                }
            )
            print(f"Automation scheduled via chat request: {arguments} -> {result}")
            return arguments
    return None


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
    """Answer one watering question for one field, or set up a daily check."""
    if context.run_config.get("agent.action") == "schedule":
        _setup_daily_automation(agent, context)
        return

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

    # If the farmer asked for a recurring check in their own words, register it.
    # Wrapped because a scheduling failure must not cost them today's answer.
    try:
        _maybe_schedule_from_question(
            agent, client, question,
            _next_daily_run_at(
                hour=int(context.run_config.get("agent.schedule_hour", 16)),
                minute=0,
                tz=str(context.run_config.get("agent.timezone", "Europe/Berlin")),
            ),
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, don't crash the run
        print(f"automation scheduling check failed: {exc!r}")

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
