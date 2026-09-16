---
tags: [agentapp]
dataset: []
framework: []
---

# Flower AgentApp — Irrigation Advisor

An `AgentApp` that answers a farmer's question (e.g. "Should I irrigate
today?") using data gathered from:

- **Weather** — live forecast (precipitation, temperature, evapotranspiration)
  from the free [Open-Meteo](https://open-meteo.com) API, no key required.
- **FL model** — the team's federated-learning irrigation prediction
  ([agent/tools/fl_model.py](agent/tools/fl_model.py)). Until the model is
  deployed on SuperGrid, set `FL_MODEL_URL` to its endpoint; without it, a
  heuristic fallback is used so the agent still runs end-to-end. **The
  endpoint must accept a GET request** with `farm_id`, `soil_moisture_pct`,
  and `forecast_precipitation_mm` query params and return a JSON object (see
  below for why POST isn't an option).
- **Local data** — locally recorded farm data such as soil moisture and crop
  ([agent/data/farms.json](agent/data/farms.json)).

These are fetched up front in Python and passed to the model as context in a
single streamed `responses.create` call (see `agent/agent_app.py`), rather
than via the Responses API's native tool-calling — streaming tool-call events
currently crash the Flower runtime's event handling server-side, so this
sidesteps that until it's fixed upstream.

**Important:** direct outbound HTTP (e.g. `requests.get`) from AgentApp code
is blocked by the SuperGrid run sandbox's egress policy — only the
platform's own connectors can reach the public internet. So all external
fetches go through `agent.connectors.call({"name": "web_fetch", ...})`
(see `_web_fetch` in `agent/agent_app.py`) instead of calling `requests`
directly. `web_fetch` only supports GET-style fetches (no arbitrary request
body), which is why the FL model endpoint's contract is GET + query params
rather than POST + JSON body.

Flower Runtime supplies the SDK base URL and task token, so the AgentApp does
not need provider credentials.

## Build

Install the project and build its Flower App Bundle (FAB):

```shell
uv sync
uv run flwr build
```

## Customize and run

Edit `agent/agent_app.py` to change the model or add your agent logic. Then log
in and run the app on SuperGrid:

```shell
uv run flwr login supergrid
uv run flwr run . supergrid --stream
```

Override the default input, farm id, or location for a run with:

```shell
uv run flwr run . supergrid \
  --run-config 'agent.input="Should I irrigate today?" agent.farm_id="farm-001" agent.latitude=52.52 agent.longitude=13.405' \
  --stream
```

## Pump automation

The agent decides whether to irrigate from the FL model's
`recommended_irrigation_mm` and logs `PUMP: ON/OFF` (a demo stand-in — no
real hardware is driven); the model's reply also states that pump status.

To run this check automatically every day, register Flower's native
recurring-run automation once with a dedicated `agent.action="schedule"`
run (see `_setup_daily_automation` in `agent/agent_app.py`, which calls the
`start_automation` connector directly rather than via the model, for the
same reason described above):

```shell
uv run flwr run . supergrid \
  --run-config 'agent.action="schedule" agent.schedule_hour=16 agent.timezone="Europe/Berlin"' \
  --stream
```

This registers a run series that repeats daily at `agent.schedule_hour` in
`agent.timezone`, each time re-running the normal irrigation check (weather +
FL model + local data) with `agent.input="Should I irrigate today?"`. Manage
or stop it from the SuperGrid federation's **Latest activity → Automations**
tab (there's no CLI command for listing/stopping automations yet).

## Learn more

See the [Flower Agent documentation](https://flower.ai/docs/agent/) for more
tutorials and guides.
