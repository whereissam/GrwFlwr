#!/usr/bin/env bash
# Copy the shared dataset into the federated app so it ships inside the FAB.
#
# The CSVs are converted to JSONL on the way in: a FAB only carries .py, .toml,
# .md, .yaml, .yml, .json, .jsonl and LICENSE (flwr/common/constant.py:71), so a
# .csv is silently dropped from the bundle. JSONL keeps one row per line.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dst="$root/federated/growflwr/dataset"
mkdir -p "$dst"

python3 - "$root/data" "$dst" << 'PY'
import csv, json, pathlib, sys

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
written = 0
for name in ["farm_1", "farm_2", "farm_3", "farm_4", "regional_weather"]:
    rows = list(csv.DictReader((src / f"{name}.csv").open()))
    with (dst / f"{name}.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"  {name}.jsonl  {len(rows)} rows")
    written += 1

(dst / "schema.json").write_text((src / "schema.json").read_text())
print(f"synced {written} datasets + schema.json into federated/growflwr/dataset/")
PY
