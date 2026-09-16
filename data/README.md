# On-farm irrigation need

Synthetic daily field observations for a Flower federated model plus an agent
that can use extra operational context.

Spec: [`DATA_SPEC.md`](DATA_SPEC.md). Do not rename columns or retune the
thresholds in that file; change `generate_data.py` to match it.

One row is one field on one morning, before irrigation that day.

## Generate

```shell
python3 data/generate_data.py --farms 2 --years 2025,2026 --seed 20260916 --out data
```

Defaults: 2 specialised farmers in the same folder, seasons 2025 and 2026
(each Apr 15–Sep 30). Farmer 1 is potato on sand; farmer 2 is maize on loam.
High-need days come from drought / high ET0, not from moving label thresholds.
About 2% of `soil_moisture_pct_nfk` cells are empty (dropout); some sensors
stick on the last reading for a few days.

Writes to `--out`:

| Path | What it is |
| --- | --- |
| `partitions.csv` | Both clients in one file. Split on `partition`: `1` and `2`. |
| `regional_weather.csv` | Shared regional `et0_mm` and `rain_mm` |
| `schema.json` | Column list, `label_mapping`, partition profiles |

## Column roles

Published columns in `partitions.csv` (simulation still uses the full spec internally):

**Identifier:** `partition` — `1` or `2`

**Features:** `soil_moisture_pct_nfk`, `previous_irrigation_mm`, `onfarm_rain_gauge_mm`, `soil_type`, `irrigation_type`

**Agent only:** `field_area_ha`

**Label:** `irrigation_need` — `0` Low, `1` Medium, `2` High

Dropped from the export: `date`, `growth_stage`, `crop_type`, `days_since_last_irrigation`, `water_source`, plus `field_id`, `crop_variety_maturity`, `days_after_planting`.
