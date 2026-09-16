# GrowFlwr — Build TODO

**GrowFlwr: federated irrigation advice without sharing farm data.**

Farms keep their telemetry local. Only model weights reach the regional authority.
A Flower Agent turns the resulting global model — plus live weather and the
farmer's own readings — into a plain-language answer to *"can I irrigate safely?"*

Architecture: [`diagrams/architecture.svg`](diagrams/architecture.svg)
(source: [`diagrams/architecture.mmd`](diagrams/architecture.mmd))

---

## Status snapshot

| Half | State |
|---|---|
| Agent (LLM loop, streaming, event emit) | stock template works, no domain logic |
| Federated (ClientApp, ServerApp, FedAvg) | not started |
| The seam (agent queries global model) | not started |
| Demo + pitch | not started |

Verified working today: `uv sync`, `uv run flwr build` → `flwrlabs.agent.0-2-0.fab`,
`agent.agent_app` imports and constructs its `AgentApp`.

---

## P0 — Repo hygiene

- [ ] **SPIKE FIRST (30 min, blocks P2):** can the AgentApp make outbound HTTPS calls
      to a third-party host? Put a bare `httpx.get("https://api.open-meteo.com/...")`
      in `main()` and run it on SuperGrid. If the sandbox blocks egress, the live-weather
      tool is dead and you fall back to a bundled forecast fixture — better to learn
      that now than at hour 10.

- [ ] Decide project layout: move the Flower project from `agent/` to repo root,
      **or** document that every `flwr` command runs from inside `agent/`.
      Right now `flwr run .` fails at repo root and a judge cloning this will hit that first.
- [ ] Root `README.md`: one-paragraph pitch, the architecture SVG, quickstart.
- [ ] Rename package `agent` → `growflwr` (pyproject `name`, `packages`,
      `[tool.flwr.app.components]`, imports).
- [ ] Set `description` in `pyproject.toml` to the real tagline.

## P1 — Federated half (this is what wins a Flower prize)

- [ ] Synthetic dataset generator: 4 farms × N days of
      `soil_moisture, rainfall_mm, temp_c, et0, crop_stage, water_used_l` →
      label `safe_to_irrigate`. Give each farm a *different* distribution
      (sandy vs clay soil, upstream vs downstream) so FedAvg visibly beats solo training.
- [ ] `ClientApp`: load that farm's partition, train the local model, return weights.
- [ ] `ServerApp` + `FedAvg` strategy, 5–10 rounds.
- [ ] Model: start with logistic regression / small MLP. Resist anything bigger —
      it has to train in seconds on stage.
- [ ] Persist the final global model where the agent can load it.
- [ ] **Money metric:** accuracy of `solo farm model` vs `federated global model`,
      per farm. Print it. This single number is the pitch.

## P2 — Agent half

- [ ] Replace the one-shot `agent.input` prompt with a real question-answering flow.
- [ ] Tool: `get_weather(lat, lon)` — live forecast. Pick a no-key API
      (open-meteo) so the demo can't die on a missing credential.
- [ ] Tool: `get_local_readings(farm_id)` — this farm's current soil/usage row.
- [ ] Tool: `predict_irrigation_safety(features)` — loads the global model, returns
      probability + the features that drove it.
- [ ] System prompt: answer with a recommendation, a confidence, and a water budget
      in litres. Always cite which tool gave which number.
- [ ] Refuse to answer from the LLM's own guesswork when a tool call fails —
      say the data is unavailable instead of hallucinating a soil reading.

## P3 — The seam (the actual demo moment)

- [ ] End-to-end script: run FL rounds → global model lands → ask the agent a
      question → it answers using that model. One command, start to finish.
- [ ] Show the counterfactual: same question against a solo-trained model gives a
      worse/riskier answer. Side by side.
- [ ] Prove the privacy claim on screen: log exactly what crosses the wire
      (tensor shapes, byte counts) and show no raw rows are in it.

## P4 — Pitch

- [ ] 3-minute demo script, timed and rehearsed.
- [ ] Slide: the problem (drought, farms won't share data, regulator needs regional view).
- [ ] Slide: the architecture diagram.
- [ ] Slide: the accuracy table from P1.
- [ ] Have a recorded fallback video. Conference wifi will betray you.

---

## Open decisions

- [ ] **Where does the global model live at inference time?** Bundled into the FAB,
      or fetched from the authority? Bundled is faster to build and demos fine.
- [ ] **Is the authority a real Flower SuperLink deployment or simulated locally?**
      Simulation is the safe hackathon answer; say so honestly on stage.
- [ ] **Differential privacy on the updates?** Strong pitch material, real time cost.
      Only if P1–P3 are done.

## Known risks

- Scope. The federated half and the agent half are each a full project. If time
  runs short, **cut P2 tools down to weather-only** and keep the FL story intact —
  a thin agent over a real federated model beats a rich agent over a fake one.
- `flwr` version pinning: project targets `1.35.0`. Don't upgrade mid-hackathon.
- The demo needs `FLWR_RUNTIME_BASE_URL` / `FLWR_RUNTIME_API_KEY`. These are injected
  by the runtime itself (`flwr/supercore/task_process/agent/run_agentapp.py:296-301`),
  so no provider credentials are needed — but it also means the agent only works
  *inside* a Flower run. There is no `python agent_app.py` path for quick local iteration.
  Budget for slow edit-run cycles, or stub the client behind an interface you can fake.
