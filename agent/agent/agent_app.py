"""A Flower AgentApp that advises farmers on irrigation decisions.

Combines a free weather forecast, the team's federated-learning irrigation
model, and locally recorded farm data, then lets the model reason over all
three to answer the farmer's question.
"""

import datetime
import json
import os
from zoneinfo import ZoneInfo

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

MODEL = "flower-endeavor-v1.0"

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
                "input": "Should I irrigate today?",
                "start_at": start_at,
                "fixed_interval": 86400,
            },
        }
    )
    print(f"Daily irrigation check scheduled starting {start_at}: {result}")


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
                    "up a recurring/automated irrigation check (e.g. 'do this "
                    "every day at 4pm'), call start_automation with "
                    'input="Should I irrigate today?", fixed_interval=86400 '
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
    if context.run_config.get("agent.action") == "schedule":
        _setup_daily_automation(agent, context)
        return

    question = context.run_config.get("agent.input")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("agent.input must be a non-empty string")

    farm_id = context.run_config.get("agent.farm_id", "farm-001")
    latitude = context.run_config.get("agent.latitude", 52.52)
    longitude = context.run_config.get("agent.longitude", 13.405)

    client = OpenAI(
        base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
        api_key=os.environ["FLWR_RUNTIME_API_KEY"],
        max_retries=0,
    )

    try:
        scheduled = _maybe_schedule_from_question(
            agent,
            client,
            question,
            _next_daily_run_at(
                hour=int(context.run_config.get("agent.schedule_hour", 16)),
                minute=0,
                tz=context.run_config.get("agent.timezone", "Europe/Berlin"),
            ),
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, don't crash the run
        print(f"automation scheduling check failed: {exc!r}")
        scheduled = None

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

    recommended_irrigation_mm = fl_prediction.get("recommended_irrigation_mm") or 0
    should_irrigate = recommended_irrigation_mm > 0
    if should_irrigate:
        print(f"PUMP: ON — irrigating {recommended_irrigation_mm:.1f}mm")
    else:
        print("PUMP: OFF — no irrigation needed")

    context_json = json.dumps(
        {
            "weather_forecast": weather,
            "local_farm_data": farm_data,
            "fl_model_prediction": fl_prediction,
            "pump_action": (
                f"ON — irrigating {recommended_irrigation_mm:.1f}mm"
                if should_irrigate
                else "OFF — no irrigation needed"
            ),
            "automation_just_scheduled": scheduled,
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
                f"Farmer question: {question.strip()}\n\n"
                "The pump has already been activated/deactivated per pump_action "
                "above (this is a demo, not a real pump) — end your answer with "
                "a line stating that pump status verbatim. If "
                "automation_just_scheduled is not null, also confirm in plain "
                "language that a recurring daily check was just set up, citing "
                "its start_at and interval. If it is null, that only means this "
                "particular message didn't create one just now — you have no "
                "way to see whether an automation from an earlier message is "
                "still active, so if asked about existing/upcoming schedules, "
                "say so plainly and point to the SuperGrid dashboard's "
                "federation → Latest activity → Automations tab instead of "
                "guessing either way."
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
