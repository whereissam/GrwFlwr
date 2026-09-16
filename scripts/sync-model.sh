#!/usr/bin/env bash
# Copy the artifacts produced by the federated run into the AgentApp bundle.
# The agent runs in a remote container and cannot read ~/.growflwr, so the
# global model has to ship inside the FAB.
set -euo pipefail
src="${GROWFLWR_ARTIFACT_DIR:-$HOME/.growflwr}"
dst="$(dirname "$0")/../agent/agent/data"
for f in global_model.json current_conditions.json solo_models.json; do
  [ -f "$src/$f" ] || { echo "missing $src/$f - run the federated training first" >&2; exit 1; }
  cp "$src/$f" "$dst/$f"
  echo "synced $f"
done
