# GrwFlwr - your agentic watering advisor

This agentic app helps you, a smallholder farmer, whether to water today,
combining live weather, locally recorded farm data, and a federated-learning
model - reasoned over by **Flower's Endeavor model**.

## How it works

For each question, the agent gathers three inputs and hands them to the
model as context:

- **Weather** - live forecast from [Open-Meteo](https://open-meteo.com).
- **Local farm data** - soil moisture, crop, etc., from
  [agent/data/farms.json](agent/data/farms.json).
- **Global model prediction** - the team's federated-learning watering
  model ([agent/tools/fl_model.py](agent/tools/fl_model.py)).

The model (`flower-endeavor-v1.0`) reasons over all three and returns a water now / wait / how much water recommendation. It can e.g. create a daily or hourly automation to check if it should water automatically.

## Installation and setup

```shell
uv sync
uv run flwr build
uv run flwr login supergrid
uv run flwr run . supergrid --stream
```

Override the default input, farm id, or location:

```shell
uv run flwr run . supergrid \
  --run-config 'agent.input="Should I water today?" agent.farm_id="farm-001" agent.latitude=52.52 agent.longitude=13.405' \
  --stream
```

## License

MIT — see [LICENSE](LICENSE).
