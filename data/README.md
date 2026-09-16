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
| `partition_1_farmer_1.csv` | Flower client / partition 1 |
| `partition_2_farmer_2.csv` | Flower client / partition 2 |
| `centralized_baseline.csv` | Pooled table for a centralized baseline only |
| `regional_weather.csv` | Shared regional `et0_mm` and `rain_mm` |
| `schema.json` | Column list, `label_mapping`, partition profiles |

## Column roles

**Identifiers:** `farm_id`, `field_id`, `date`

**Features:** `soil_moisture_pct_nfk`, `growth_stage`, `crop_type`, `days_since_last_irrigation`, `previous_irrigation_mm`, `onfarm_rain_gauge_mm`, `soil_type`, `crop_variety_maturity`, `days_after_planting`, `irrigation_type`

**Agent only:** `water_source`, `field_area_ha`

**Label:** `irrigation_need` — `0` Low, `1` Medium, `2` High (see `label_mapping` in `schema.json`)
