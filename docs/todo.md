# GrwFlwr — Build TODO

**Federated irrigation advice without sharing farm data.**

Farms in one water region will not publish their telemetry, and each has only
ever grown its own crop. GrwFlwr trains one model across them without moving a
row, then puts a Flower Agent in front of it.

Architecture: [`diagrams/architecture.svg`](diagrams/architecture.svg) ·
Data asks: [`DATA_REQUEST.md`](DATA_REQUEST.md) · Deck: [`../slides/grwflwr.html`](../slides/grwflwr.html)

---

## Status

| Piece | State |
|---|---|
| `federated/` — FedAvg, 3-class, 2 farms, 30 rounds | **working**, ~40s in simulation |
| `agent/` — AgentApp on SuperGrid | **working**, live weather + federated model |
| Counterfactual (solo vs federated) | **working**, printed on every answer |
| Daily scheduling (Julian) | **merged**, `agent.action="schedule"` |
| `scripts/demo.sh` end to end | **working** |
| Root README, 5-slide deck | **written** |
| Demo script, rehearsal, fallback video | **not started** |

Headline numbers — train 2025, test 2026, 973 held-out rows
(Low 645 · Medium 284 · High 44):

```
model                         macro-F1    vs federated
solo: farmer_1                   0.596          −0.140
solo: farmer_2                   0.666          −0.071
FEDERATED (FedAvg)               0.736
```

Cross-crop, the strongest result: farmer_1 has only ever grown potato. Scored on
farmer_2's maize it manages **0.480** against the federated **0.664** (+38%).

744 bytes of weights per farm per round. 973 rows stayed on the farms.

---

## Things that are true and must not be misstated on stage

- **The dataset is generated, not measured.** Every figure describes a model
  recovering `generate_data.py`'s rules. Real error would be larger.
- **High recall is 68% — 30 of 44.** Centralized pooling catches slightly more
  (70%). We are not ahead on the class that matters most.
- **Federated 0.736 vs centralized 0.722.** Close enough that the honest claim is
  "keeping the data home costs essentially nothing", not "federation wins".
- **farmer_2's own model still edges federated on its own maize** (0.678 vs
  0.664). Federation's win is on ground you have not farmed.
- **Two clients is a very small federation.** An earlier four-farm split had one
  farm *losing* by joining. Whether everyone gains depends on specialisation.
- **The water volume is an agronomic rule, not a model output** — ET₀ × days
  since irrigation, less measured rain, capped at 30 mm. The model classifies
  need; it does not compute litres.
- **"+18% accuracy" is not ours.** Our gains are +23% / +11% macro-F1 regionally
  and +38% cross-crop. We report macro-F1, not accuracy, because always
  predicting Low scores 66.3%.

---

## Settled, with the evidence

- [x] **Two Flower projects, not one.** `control_handlers.py:2156` derives app
      type from the FAB: `AGENT_APP if "agentapp" in components else SERVER_APP`,
      and simulation is skipped for agent bundles (`:669`). Tried it; `flwr run`
      started `flwr-agentapp` and the FL code never executed.
- [x] **Direct outbound HTTP is blocked; connectors are not.** The egress proxy
      403s everything but `pypi.org`, `raw.githubusercontent.com`,
      `api.flower.ai`. `agent.connectors.call({"name": "web_fetch"...})` is the
      sanctioned way out, and is how live weather works.
- [x] **Tool calling works, but not while streaming.** Streaming a turn with
      tools on the table crashes the runtime; a single non-streaming
      `responses.create(tools=..., tool_choice="auto")` is fine. That is how
      scheduling-from-chat works. Irrigation advice still uses deterministic
      grounding so the model cannot invent a figure.
- [x] **No model API key on SuperGrid** — the runtime injects
      `FLWR_RUNTIME_BASE_URL` / `FLWR_RUNTIME_API_KEY` (`run_agentapp.py:296`). A
      *local* SuperLink needs `FLWR_MODEL_API_KEY` instead.
- [x] **A FAB carries no `.csv`** — `common/constant.py:71` allows only
      py/toml/md/yaml/yml/json/jsonl/LICENSE. `scripts/sync-data.sh` converts to
      JSONL.
- [x] **num-supernodes must be passed per run.** `flwr` migrates
      `[tool.flwr.federations]` into `~/.flwr/config.toml` and then ignores
      pyproject, so a value there goes stale. More SuperNodes than partitions
      does not fail the run — the extras raise every round while FedAvg quietly
      completes. Use `--federation-config 'num-supernodes=2'`.
- [x] **Class weighting is the load-bearing choice.** Turning it off barely moves
      macro-F1 (0.709 → 0.693) but collapses High recall from 68% to **23%**.
- [x] **The model was undertrained at 10 rounds.** 30 rounds moves train/test
      together, 0.751/0.709 → 0.775/0.736. Still underfitting.
- [x] **Local epochs barely matter** between 1 and 100; only one-shot averaging
      is bad (0.633). The objective is convex, so client drift is mild — this
      will not hold if the model is ever swapped for a network.
- [x] **Non-IID does not hurt here** — specialised 0.709 vs IID-shuffled 0.695.
- [x] **Interaction terms matter** — the label is a moisture threshold depending
      on crop and stage, which an additive model cannot express.
      macro-F1 0.704 → 0.732 on the previous data.

---

## P4 — Pitch (the only thing left a judge sees)

- [ ] 3-minute demo script, timed and rehearsed against `scripts/demo.sh`.
- [ ] **Recorded fallback video.** The agent cannot run without SuperGrid, and
      the deck's live-weather path can fall back mid-demo.
- [ ] Decide who says the "generated data" caveat and when. Best said early, by
      us.
- [ ] **Check the presenting browser has auto dark mode off.** Chrome's rewrites
      turned the deck's orange fills near-black during testing;
      `color-scheme: light` did not stop it. Export a PDF as insurance.

## Smaller things, if time allows

- [ ] The agent crashes on a missing sensor reading rather than degrading. The
      FL side imputes; the agent does not.
- [ ] Only farmer_1 and farmer_2's default fields have been exercised against
      live weather. A bad ET₀ silently produces a confident wrong volume.
- [ ] `scripts/demo.sh` still loops farms 1–4; it should read the farm list from
      the dataset rather than hardcoding.
- [ ] Julian's branch and this one have diverged once already. Agree which is
      trunk before either side adds more.

## Known risks

- **Live weather can fail mid-demo** and fall back to a seasonal average, which
  the answer will say out loud. Rehearse that sentence.
- **Artifacts must be re-synced after retraining** (`scripts/sync-model.sh`), or
  the agent answers from a stale model. The feature-count check catches a shape
  change, not a stale value.
- **Adding a crop or soil to `schema.json` invalidates every trained model** —
  the feature vector changes length. Retrain and re-sync.
- `flwr` pinned at 1.37.0 locally; projects target 1.35.0. Do not upgrade
  mid-hackathon.
