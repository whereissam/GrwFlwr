# GrwFlwr — Build TODO

**Federated irrigation advice without sharing farm data.**

Farms in one water region each hold telemetry they will not publish. None has
seen enough conditions alone to know when irrigation is genuinely needed.
GrwFlwr trains one model across all of them without moving a row, then puts a
Flower Agent in front of it.

Architecture: [`diagrams/architecture.svg`](diagrams/architecture.svg) ·
Data asks: [`DATA_REQUEST.md`](DATA_REQUEST.md)

---

## Status

| Piece | State |
|---|---|
| `federated/` — FedAvg, 3-class, 4 farms | **working**, ~25s in local simulation |
| `agent/` — AgentApp on SuperGrid | **working**, live weather + federated model |
| Counterfactual (solo vs federated) | **working**, printed on every answer |
| `scripts/demo.sh` end to end | **working** |
| Root README | **written** |
| Slides / demo script / fallback video | **not started** |

Current numbers, pooled held-out fields (666 rows: Low 502, Medium 151, High 13):

```
model                         macro-F1    vs federated  High rows held
solo: farm_1                     0.619          -0.113              10
solo: farm_2                     0.713          -0.019               9
solo: farm_3                     0.654          -0.078               7
solo: farm_4                     0.752          +0.020              15
FEDERATED (FedAvg)               0.732                              41
```

744 bytes of weights per farm per round. 2,126 rows stayed on the farms.

---

## Things that are true and must not be misstated on stage

- **The dataset is generated, not measured.** Every figure describes how well a
  model recovers `data/generate_data.py`'s rules, not how it would advise a real
  farmer. Real error would very likely be larger.
- **Federation is not free for everyone.** farm_4 holds the most data and gives
  up 0.020 macro-F1 by joining. Three farms gain, one pays. That trade is the
  argument for the *regional authority* owning the model rather than the largest
  farm — a better story than a fake uniform win.
- **High recall is 77% of 13 rows.** That is 10 out of 13. Do not present it as a
  precise number.
- **Federated (0.732) currently beats centralized pooling (0.680)**, because
  per-client class weighting balances the rare High class better than one pooled
  fit. Genuine and worth saying: federating costs nothing against centralizing.
- **The water volume is an agronomic rule, not a model output.** ET₀ × days since
  irrigation, less measured rain, capped at 30 mm. The model classifies need; it
  does not compute litres.

---

## Settled, with the evidence

- [x] **Two Flower projects, not one.** `control_handlers.py:2156` derives app
      type from the FAB: `AGENT_APP if "agentapp" in components else SERVER_APP`,
      and simulation is skipped for agent bundles (`:669`). Declaring all three
      components was tried — `flwr run` started `flwr-agentapp` and the FL code
      never executed. No CLI flag overrides it.
- [x] **Direct outbound HTTP is blocked; connectors are not.** The egress proxy
      allowlists `pypi.org`, `raw.githubusercontent.com`, `api.flower.ai` and
      403s everything else. But `agent.connectors.call({"name": "web_fetch"...})`
      reaches the public internet — that is how live weather works. DNS resolves
      for blocked hosts, so direct calls fail looking like flaky networking.
- [x] **No model API key on SuperGrid** — the runtime injects
      `FLWR_RUNTIME_BASE_URL` / `FLWR_RUNTIME_API_KEY` (`run_agentapp.py:296`). A
      *local* SuperLink is different: it needs `FLWR_MODEL_API_KEY`, or
      `FLWR_MODEL_API_ENDPOINT` pointed at an Open-Responses-compatible service.
- [x] **A FAB carries no `.csv`** — `common/constant.py:71` allows only
      py/toml/md/yaml/yml/json/jsonl/LICENSE. CSVs are dropped silently.
      `scripts/sync-data.sh` converts to JSONL.
- [x] **Model-driven tool calls are unusable** — streaming tool-call events crash
      the runtime's event handling. Context is gathered up front instead.
- [x] **Accuracy is a useless metric here** — always predicting Low scores 75.4%.
      macro-F1 and per-class recall throughout.
- [x] **Splitting by date destroys the High class** — left 2 High rows in the
      whole test set. Split holds out each farm's last field instead.
- [x] **Interaction terms matter** — the label is a moisture threshold that
      depends on crop and stage, which an additive model cannot express. Adding
      moisture×crop and moisture×stage: macro-F1 0.704 → 0.732.

---

## P4 — Pitch (the only thing left that a judge sees)

- [ ] Slide deck: problem → architecture → the solo-vs-federated table → the
      privacy audit (744 bytes vs 2,126 rows) → the honest caveats.
- [ ] 3-minute demo script, timed and rehearsed against `scripts/demo.sh`.
- [ ] **Recorded fallback video.** Conference wifi will betray you, and the agent
      cannot run without SuperGrid.
- [ ] Decide who says the "generated data" caveat and when. Best said early and
      by you, not extracted by a judge in Q&A.

## Smaller things, if time allows

- [ ] The agent crashes on a missing sensor reading rather than degrading. Real
      telemetry has gaps; see `DATA_REQUEST.md` ⑤.
- [ ] Only farm_1 and farm_2 have been exercised against live weather. A bad ET₀
      silently produces a confident wrong volume.
- [ ] `scenario="unusual"` (the searched disagreement case) was lost in the
      rewrite to three classes. The counterfactual still prints, but there is no
      curated case where the models disagree — worth restoring for the demo.
- [ ] Merge `grwflwr-merged` and `agent-julian`. Currently a clean fast-forward;
      that stops being true the moment either side adds a commit.

## Known risks

- **Live weather adds a failure mode bundling did not have.** If open-meteo is
  slow mid-demo the run falls back to a seasonal average and says so on stage.
  Rehearse that sentence.
- **Artifacts must be re-synced after retraining** (`scripts/sync-model.sh`), or
  the agent answers from a stale model. The feature-count check in
  `tools/fl_model.py` catches a shape change but not a silent value change.
- `flwr` pinned at 1.37.0 locally; projects target 1.35.0. Do not upgrade
  mid-hackathon.
