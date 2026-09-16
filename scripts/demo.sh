#!/usr/bin/env bash
# GrwFlwr end-to-end demo: train federatively, then ask the agent.
#
#   ./scripts/demo.sh            train + ask every farm
#   ./scripts/demo.sh --ask-only skip training, use the current model
#   ./scripts/demo.sh --farm 1   just one farm, both scenarios
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
ask_only=false
only_farm=""

while [ $# -gt 0 ]; do
  case "$1" in
    --ask-only) ask_only=true; shift ;;
    --farm) only_farm="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

rule() { printf '\n%s\n' "$(printf '=%.0s' {1..78})"; }

if [ "$ask_only" = false ]; then
  rule; echo "1/3  Federated training - four farms, raw data stays local"; rule
  # num-supernodes is pinned on the command line, not in pyproject.toml: the
  # flwr config migration comments out [tool.flwr.federations] and copies the
  # value into ~/.flwr/config.toml, where it then goes stale. Spawning more
  # SuperNodes than the dataset has partitions does not fail the run -- FedAvg
  # completes on the valid clients while the rest throw every round.
  (cd "$root/federated" && uv run flwr run . local-sim \
     --federation-config 'num-supernodes=2' --stream)

  rule; echo "2/3  Bundling the global model into the AgentApp"; rule
  "$root/scripts/sync-model.sh"
fi

rule; echo "3/3  Asking the agent"; rule

farms=(1 2 3 4)
[ -n "$only_farm" ] && farms=("$only_farm")

for n in "${farms[@]}"; do
  farm_id="farm-00${n}"

  echo
  echo "### ${farm_id} - today's conditions, live forecast"
  (cd "$root/agent" && uv run flwr run . supergrid \
    -c "agent.farm_id=\"${farm_id}\" agent.scenario=\"today\" agent.input=\"Should I irrigate today?\"" \
    --stream 2>&1 | grep -vE '^\[92mINFO|^\[93mWARNING|Installed:|uv sync took|Successfully|Using SuperLink|Building|Built|Uninstalled|Installed 1')

  echo
  echo "### ${farm_id} - a condition this farm rarely sees"
  (cd "$root/agent" && uv run flwr run . supergrid \
    -c "agent.farm_id=\"${farm_id}\" agent.scenario=\"unusual\" agent.input=\"Conditions look strange this week, should I irrigate?\"" \
    --stream 2>&1 | grep -vE '^\[92mINFO|^\[93mWARNING|Installed:|uv sync took|Successfully|Using SuperLink|Building|Built|Uninstalled|Installed 1')
done

rule; echo "Done."; rule
