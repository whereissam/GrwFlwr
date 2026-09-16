# Irrigation dataset spec

Do not rename columns. Do not change the numeric thresholds, Kc values, stage
lengths, soil TAW, or maturity modifiers in this file. `generate_data.py` must
follow them exactly.

One row = one field on one morning, before irrigation that day.

## Season and clients

- Season: `2026-04-15` to `2026-09-30` inclusive
- Default: 4 farms, 3–5 fields each
- Federated files: `farm_1.csv` … `farm_N.csv`
- Centralized baseline (pooled, not a Flower client): `centralized_baseline.csv`
- Shared weather: `regional_weather.csv` with `date`, `et0_mm`, `rain_mm`

## Columns (do not rename)

| Column | Role | Type | Notes |
| --- | --- | --- | --- |
| `farm_id` | identifier | string | `farm_1` … `farm_N`. Flower client key. |
| `field_id` | identifier | string | Globally unique. |
| `date` | identifier | date | `YYYY-MM-DD`. Morning observation. |
| `soil_moisture_pct_nfk` | feature | float | Sensor reading, clipped to 0–100. |
| `growth_stage` | feature | string | `initial`, `development`, `sensitive`, `late` |
| `crop_type` | feature | string | `potato`, `maize`, `sugar_beet` |
| `days_since_last_irrigation` | feature | int | If never irrigated: equal to `days_after_planting`. No sentinel. |
| `previous_irrigation_mm` | feature | float | 0 if never irrigated this season. |
| `onfarm_rain_gauge_mm` | feature | float | Same value for every field on a farm on a given day. |
| `soil_type` | feature | string | `sand`, `loamy_sand`, `loam` |
| `crop_variety_maturity` | feature | string | `early`, `medium`, `late` |
| `days_after_planting` | feature | int | |
| `irrigation_type` | feature | string | `hose_reel`, `pivot`, `drip` |
| `water_source` | agent_only | string | Not a federated training feature. |
| `field_area_ha` | agent_only | float | Not a federated training feature. |
| `irrigation_need` | label | int | 0 / 1 / 2. See label mapping. |

## Label mapping

Store `irrigation_need` as integers:

| Value | Name |
| --- | --- |
| 0 | Low |
| 1 | Medium |
| 2 | High |

Write this mapping to `schema.json` as `label_mapping`.

Compute the label from **true** soil moisture (% nFK), not from the sensor
reading that is written to the CSV.

Let `T_low` and `T_med` be the crop-and-stage thresholds below. After applying
the potato maturity adjustment:

- if `true_moisture >= T_low` → `0` (Low)
- else if `true_moisture >= T_med` → `1` (Medium)
- else → `2` (High)

## Crop-and-stage thresholds (% nFK)

`T_low` is the wetter boundary (Low vs Medium). `T_med` is the drier boundary
(Medium vs High).

| crop | stage | T_low | T_med |
| --- | --- | --- | --- |
| potato | initial | 50 | 35 |
| potato | development | 55 | 40 |
| potato | sensitive | 65 | 50 |
| potato | late | 45 | 30 |
| maize | initial | 40 | 25 |
| maize | development | 50 | 35 |
| maize | sensitive | 60 | 45 |
| maize | late | 40 | 25 |
| sugar_beet | initial | 45 | 30 |
| sugar_beet | development | 50 | 35 |
| sugar_beet | sensitive | 55 | 40 |
| sugar_beet | late | 40 | 25 |

## `crop_variety_maturity` effects

Stage lengths for medium varieties are in days:

| crop | initial | development | sensitive | late |
| --- | --- | --- | --- | --- |
| potato | 25 | 30 | 45 | 30 |
| maize | 30 | 40 | 50 | 30 |
| sugar_beet | 25 | 35 | 50 | 50 |

Length scale by maturity (round to nearest day, minimum 1):

| maturity | length scale |
| --- | --- |
| early | 0.85 |
| medium | 1.00 |
| late | 1.15 |

Potato threshold adjustment (add to both `T_low` and `T_med`):

| maturity | Δ % nFK |
| --- | --- |
| early | +5 |
| medium | 0 |
| late | −5 |

Maize and sugar beet thresholds do not change with maturity; only stage lengths do.

## Crop coefficients Kc (FAO-56-style)

Daily drying uses `Kc(crop, stage) × ET0 × et_factor`, converted to % nFK with
soil TAW. Do not use a constant dry-down per soil.

| crop | initial | development | sensitive | late |
| --- | --- | --- | --- | --- |
| potato | 0.50 | 0.82 | 1.15 | 0.75 |
| maize | 0.30 | 0.75 | 1.20 | 0.60 |
| sugar_beet | 0.35 | 0.78 | 1.20 | 0.70 |

## Soil TAW (mm, full root zone)

Used only to convert mm of ET / rain / irrigation into % nFK.

| soil_type | taw_mm |
| --- | --- |
| sand | 70 |
| loamy_sand | 100 |
| loam | 140 |

True moisture is clipped to 0–100 after each day's update. The sensor column
is also clipped to 0–100.

## Weather

One regional series is shared by all farms (`regional_weather.csv`).

On-farm rain for farm `f` on date `d`:

```
onfarm_rain_gauge_mm(f, d) = max(0, regional_rain(d) × farm_factor(f) + ε)
```

`farm_factor(f)` is drawn once per farm. `ε ~ N(0, 0.4)` mm is drawn once per
farm-day. Every field on that farm on that day must store the same gauge value.

## Hidden field effects and sensor noise

Per field, drawn once and **not** written to the CSV:

- `et_factor ~ N(1.0, 0.08)`
- sensor bias `~ N(0, 2)`

Each day, the CSV moisture is:

```
soil_moisture_pct_nfk = clip(true_moisture + bias + N(0, 3), 0, 100)
```

## Irrigation events (for the time series only)

Irrigation decisions are not the label. They only update the water balance so
the season is physically plausible:

- always irrigate when the true-moisture class is High
- sometimes irrigate when Medium (more often for drip than hose reel)
- refill toward 85% nFK, capped by system typical depth: drip 18 mm, pivot 30 mm, hose_reel 40 mm
