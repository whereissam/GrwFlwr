# GrowFlwr — Build TODO

**GrowFlwr: federated irrigation advice without sharing farm data.**

Farms keep their telemetry local. Only model weights reach the regional authority.
A Flower Agent turns the resulting global model — plus each farm's own readings —
into a plain-language answer to *"can I irrigate safely?"*

Architecture: [`diagrams/architecture.svg`](diagrams/architecture.svg)
(source: [`diagrams/architecture.mmd`](diagrams/architecture.mmd))

---

## Status

| Half | State |
|---|---|
| Federated training (`federated/`) | **working** — FedAvg over 4 farms, 8 rounds, ~23s |
| Agent (`agent/`) | **working on SuperGrid** — live weather + federated model |
| The seam | **working** — model ships in the FAB, agent scores against it |
| Counterfactual demo | **working** — solo vs federated, `scenario="unusual"` |
| Demo script + pitch | not started |

Branch `growflwr-merged` combines Julian's agent skeleton (live weather via the
`web_fetch` connector, per-source graceful degradation) with the federated model
and four-farm data. Live open-meteo readings now feed the model's rain, ET₀ and
temperature features; the farm's own sensors supply the other four and never
leave the app.

Current numbers (one seed, region-wide held-out set):

```
farm                                     alone   federated      gain
Farm 1 - sandy upland, canal head        93.2%       97.3%    +4.1pp
Farm 2 - clay lowland, canal head        97.1%       97.3%    +0.2pp
Farm 3 - loam midslope, canal tail       96.8%       97.3%    +0.5pp
Farm 4 - terraced orchard, canal tail    94.2%       97.3%    +3.1pp
mean                                     95.3%       97.3%    +2.0pp
```

---

## Why there are two Flower projects

Not a style choice. `control_handlers.py:2156` derives the app type from the FAB:

```python
return TaskType.AGENT_APP if "agentapp" in components else TaskType.SERVER_APP
```

A FAB is **either** an AgentApp **or** a ServerApp/ClientApp pair, and simulation
is explicitly skipped for agentapp bundles (`control_handlers.py:669`). Declaring
all three in one project was tried: `flwr run` started `flwr-agentapp` and the FL
code never executed. There is no CLI flag to pick a component.

So: `agent/` is the AgentApp (built on the Flower AI example, edited in place),
`federated/` is the FL pair. They communicate through a JSON artifact.

---

## Done

- [x] **P0 egress spike — and its correction.** Direct outbound HTTP from the
      AgentApp is blocked by an allowlisting proxy
      (`proxy.frontend.platform.prod:8080`): `api.open-meteo.com` and
      `example.com` fail with `Tunnel connection failed: 403 Forbidden`, while
      `pypi.org` and `raw.githubusercontent.com` succeed. DNS resolves for
      blocked hosts, so it presents as flaky networking rather than policy.
      **This does not mean the internet is unreachable.** The platform provides
      connectors — `web_fetch`, `web_search`, `browser_use`
      (`connector/registry.py:44`) — and `agent.connectors.call({"name":
      "web_fetch", ...})` is the sanctioned path out. Live weather works.
- [x] **Model-driven tool calls are not usable yet.** Streaming tool-call events
      crash the runtime's event handling (server-side). Context is gathered up
      front instead; `agent.tools(names)` exists for when that is fixed.
- [x] **No model API key needed on SuperGrid** — the runtime injects
      `FLWR_RUNTIME_BASE_URL` / `FLWR_RUNTIME_API_KEY`
      (`run_agentapp.py:296-301`). A **local** SuperLink is different: it needs
      `FLWR_MODEL_API_KEY` (a Flower AI key, default endpoint
      `https://api.flower.ai/v1/responses`), or `FLWR_MODEL_API_ENDPOINT`
      pointed at any Open-Responses-compatible service.
- [x] Synthetic per-farm data where each farm sits a chosen signed distance from
      the regional rule, so its local history is genuinely unrepresentative.
- [x] NumPy logistic regression; weights are 64 bytes, easy to show on screen.
- [x] `ClientApp` / `ServerApp` / FedAvg, running in local simulation.
- [x] Solo-vs-federated table on a region-wide held-out set, plus a byte-level
      audit of what crossed the wire.
- [x] Model artifact carries its own unit mapping (`feature_stats`), so the agent
      speaks in soil-moisture percent rather than standardized units.
- [x] Agent tools: farm readings, safety prediction with ranked drivers, water
      budget in litres/hectare.
- [x] Agent refuses to answer when a fact is missing instead of inventing one
      (verified: `farm-id=9` fails loudly).

## P2 — Agent, remaining

- [x] Percentages pre-formatted in `tools/fl_model.py`.
- [x] Live weather through the `web_fetch` connector, feeding three of the
      model's seven features.
- [x] Graceful degradation per source — a failed weather lookup falls back to
      stored values and the answer says so rather than implying it was fresh.
- [ ] Let a farmer ask a hypothetical ("what if I wait three days?") by
      overriding readings, not just scoring today's row.
- [ ] Only farm-001's coordinates have been exercised against live weather.
      Check the other three return sane ET₀ — a bad forecast silently produces a
      confident wrong answer.
- [ ] Decide whether to serve the model from `raw.githubusercontent.com` instead
      of bundling. Bundled is safer on stage.

## P3 — The seam

- [x] `scripts/sync-model.sh` copies training artifacts into the agent bundle.
- [x] Counterfactual: every answer is scored by both this farm's solo model and
      the federated one, and the header prints both. Disagreement cases are
      searched from held-out data during training, not authored.
- [ ] One command that runs training then asks a question, start to finish.
- [ ] Merge `growflwr-merged` back — agree with Julian which branch is trunk
      before either of you builds further on top of it.

## P4 — Pitch

- [ ] 3-minute demo script, timed and rehearsed.
- [ ] Slides: problem → architecture diagram → accuracy table → privacy audit.
- [ ] Recorded fallback video. Conference wifi will betray you.

---

## Known risks

- **The gain is not uniform, and the pitch must not pretend otherwise.**
  Federation rescues farms with narrow local history (+4.1pp) and does almost
  nothing for a farm that already sees representative conditions (+0.2pp). A
  sweep over spread, regime extremity and history length found **no config where
  all four farms gain reliably** — the best was 9/10 seeds with the top farm
  gaining +0.4pp, inside the noise. Re-running with a different seed will move
  the small numbers and may turn one slightly negative. Say this out loud before
  a judge finds it; it is a real property of FL, not a defect.
- **The demo depends on SuperGrid.** The agent cannot run without a runtime, and
  there is no `python agent_app.py` path. Every iteration is a full remote run.
- **Artifacts must be re-synced after retraining**, or the agent answers with a
  stale model. `scripts/sync-model.sh` is the only thing preventing this.
- `flwr` is pinned at 1.37.0 locally; projects target 1.35.0. Do not upgrade
  mid-hackathon.
- **Live weather adds a failure mode the bundled version did not have.** If
  open-meteo is slow or the connector errors mid-demo, the run falls back to
  stored values. That is handled, but the answer will say "stored values" on
  stage — rehearse what you say when it happens.
- **Two branches now diverge.** `agent-julian` and `growflwr-merged` both touch
  `agent/agent_app.py` heavily. Settle trunk early; a merge conflict at hour 10
  is worse than a five-minute conversation now.
