#!/usr/bin/env python3
"""Generate the irrigation table. Label constants must match data/DATA_SPEC.md."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

# In-year window from DATA_SPEC.md; --years repeats that window.
SEASON_START_MD = (4, 15)
SEASON_END_MD = (9, 30)
DEFAULT_YEARS = (2025, 2026)
DEFAULT_FARMS = 2

CROPS = ("potato", "maize", "sugar_beet")
SOILS = ("sand", "loamy_sand", "loam")
IRRIGATION_TYPES = ("hose_reel", "pivot", "drip")
STAGES = ("initial", "development", "sensitive", "late")
MATURITIES = ("early", "medium", "late")
WATER_SOURCES = (
    "groundwater_well",
    "irrigation_canal",
    "river_pump",
    "farm_reservoir",
)

def client_id(index: int) -> str:
    return f"farmer_{index}"


COMBINED_FILENAME = "partitions.csv"


# First two profiles are the default pair (different crop and soil).
# Spec soils only: sand, loamy_sand, loam.
FARM_PROFILES = (
    {"crops": ("potato",), "soils": ("sand",), "irrigation": ("drip",)},
    {"crops": ("maize",), "soils": ("loam",), "irrigation": ("pivot",)},
    {"crops": ("sugar_beet",), "soils": ("loam",), "irrigation": ("drip",)},
    {"crops": ("potato",), "soils": ("sand",), "irrigation": ("hose_reel",)},
    {"crops": ("maize",), "soils": ("loamy_sand",), "irrigation": ("hose_reel",)},
    {"crops": ("sugar_beet",), "soils": ("loam",), "irrigation": ("pivot",)},
    {"crops": ("potato", "maize"), "soils": ("sand", "loamy_sand"), "irrigation": ("drip", "pivot")},
    {"crops": ("maize", "sugar_beet"), "soils": ("loam",), "irrigation": ("hose_reel", "pivot")},
    {"crops": ("potato", "sugar_beet"), "soils": ("loamy_sand",), "irrigation": ("drip", "hose_reel")},
    {"crops": ("potato",), "soils": ("sand", "loamy_sand"), "irrigation": ("pivot",)},
)

COLUMN_ORDER = [
    "farm_id",
    "field_id",
    "date",
    "soil_moisture_pct_nfk",
    "growth_stage",
    "crop_type",
    "days_since_last_irrigation",
    "previous_irrigation_mm",
    "onfarm_rain_gauge_mm",
    "soil_type",
    "crop_variety_maturity",
    "days_after_planting",
    "irrigation_type",
    "water_source",
    "field_area_ha",
    "irrigation_need",
]

# Published CSV: Flower client id plus irrigation-need columns only.
EXPORT_COLUMNS = [
    "partition",
    "soil_moisture_pct_nfk",
    "previous_irrigation_mm",
    "onfarm_rain_gauge_mm",
    "soil_type",
    "irrigation_type",
    "water_source",
    "field_area_ha",
    "irrigation_need",
]


def partition_id(farm_id: str) -> int:
    return int(farm_id.rsplit("_", 1)[1])


def export_row(row: dict) -> dict:
    return {
        "partition": partition_id(row["farm_id"]),
        "soil_moisture_pct_nfk": row["soil_moisture_pct_nfk"],
        "previous_irrigation_mm": row["previous_irrigation_mm"],
        "onfarm_rain_gauge_mm": row["onfarm_rain_gauge_mm"],
        "soil_type": row["soil_type"],
        "irrigation_type": row["irrigation_type"],
        "water_source": row["water_source"],
        "field_area_ha": row["field_area_ha"],
        "irrigation_need": row["irrigation_need"],
    }

LABEL_MAPPING = {0: "Low", 1: "Medium", 2: "High"}

# Crop-and-stage thresholds (% nFK): (T_low, T_med). See DATA_SPEC.md.
THRESHOLDS = {
    "potato": {
        "initial": (50, 35),
        "development": (55, 40),
        "sensitive": (65, 50),
        "late": (45, 30),
    },
    "maize": {
        "initial": (40, 25),
        "development": (50, 35),
        "sensitive": (60, 45),
        "late": (40, 25),
    },
    "sugar_beet": {
        "initial": (45, 30),
        "development": (50, 35),
        "sensitive": (55, 40),
        "late": (40, 25),
    },
}

STAGE_LENGTHS = {
    "potato": (25, 30, 45, 30),
    "maize": (30, 40, 50, 30),
    "sugar_beet": (25, 35, 50, 50),
}

MATURITY_LENGTH_SCALE = {"early": 0.85, "medium": 1.00, "late": 1.15}
POTATO_THRESHOLD_DELTA = {"early": 5, "medium": 0, "late": -5}

KC = {
    "potato": {
        "initial": 0.50,
        "development": 0.82,
        "sensitive": 1.15,
        "late": 0.75,
    },
    "maize": {
        "initial": 0.30,
        "development": 0.75,
        "sensitive": 1.20,
        "late": 0.60,
    },
    "sugar_beet": {
        "initial": 0.35,
        "development": 0.78,
        "sensitive": 1.20,
        "late": 0.70,
    },
}

TAW_MM = {"sand": 70.0, "loamy_sand": 100.0, "loam": 140.0}
MAX_IRRIGATION_MM = {"drip": 18.0, "pivot": 30.0, "hose_reel": 40.0}
MEDIUM_IRRIGATE_P = {"drip": 0.45, "pivot": 0.25, "hose_reel": 0.15}

SENSOR_BIAS_SD = 2.0
SENSOR_DAILY_SD = 3.0
ET_FACTOR_MEAN = 1.0
ET_FACTOR_SD = 0.08
FARM_RAIN_NOISE_SD = 0.4
DEFAULT_MISSING_RATE = 0.02


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def daterange(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def stage_lengths(crop: str, maturity: str) -> tuple[int, int, int, int]:
    scale = MATURITY_LENGTH_SCALE[maturity]
    lengths = []
    for raw in STAGE_LENGTHS[crop]:
        lengths.append(max(1, int(round(raw * scale))))
    return tuple(lengths)  # type: ignore[return-value]


def growth_stage(crop: str, maturity: str, dap: int) -> str:
    acc = 0
    for stage, length in zip(STAGES, stage_lengths(crop, maturity)):
        acc += length
        if dap < acc:
            return stage
    return "late"


def class_thresholds(crop: str, stage: str, maturity: str) -> tuple[float, float]:
    t_low, t_med = THRESHOLDS[crop][stage]
    if crop == "potato":
        delta = POTATO_THRESHOLD_DELTA[maturity]
        t_low += delta
        t_med += delta
    return t_low, t_med


def irrigation_need(true_moisture: float, crop: str, stage: str, maturity: str) -> int:
    t_low, t_med = class_thresholds(crop, stage, maturity)
    if true_moisture >= t_low:
        return 0
    if true_moisture >= t_med:
        return 1
    return 2


def drought_scales(day: date) -> tuple[float, float]:
    """ET0 scale and wet-probability scale. Does not change Kc or thresholds."""
    md = (day.month, day.day)
    if day.year == 2026 and (6, 8) <= md <= (7, 28):
        return 1.45, 0.18
    if day.year == 2026 and (8, 1) <= md <= (8, 18):
        return 1.25, 0.35
    if day.year == 2025 and (7, 1) <= md <= (7, 12):
        return 1.20, 0.50
    return 1.0, 1.0


def regional_weather(years: list[int], rng: random.Random) -> list[dict]:
    rows = []
    for year in years:
        start = date(year, *SEASON_START_MD)
        end = date(year, *SEASON_END_MD)
        for day in daterange(start, end):
            doy = day.timetuple().tm_yday
            et_scale, wet_scale = drought_scales(day)
            seasonal = 2.2 + 2.4 * math.sin(2 * math.pi * (doy - 105) / 365)
            et0 = clip((seasonal + rng.gauss(0, 0.45)) * et_scale, 0.8, 7.8)
            wet_p = (0.22 + 0.08 * math.cos(2 * math.pi * (doy - 60) / 365)) * wet_scale
            if rng.random() < wet_p:
                rain = rng.expovariate(1 / 6.5)
                if rng.random() < 0.12:
                    rain += rng.uniform(12.0, 28.0)
            else:
                rain = rng.uniform(0.0, 0.4) if rng.random() < 0.08 else 0.0
            rows.append(
                {
                    "date": day.isoformat(),
                    "et0_mm": round(et0, 2),
                    "rain_mm": round(max(0.0, rain), 1),
                }
            )
    return rows


@dataclass
class FieldSpec:
    farm_id: str
    field_id: str
    crop_type: str
    soil_type: str
    crop_variety_maturity: str
    irrigation_type: str
    water_source: str
    field_area_ha: float
    plant_month: int
    plant_day: int
    taw_mm: float
    et_factor: float
    sensor_bias: float
    missing_rate: float
    stuck_p: float


def planting_window(crop: str, year: int) -> tuple[date, date]:
    windows = {
        "potato": ((4, 8), (4, 22)),
        "sugar_beet": ((4, 1), (4, 18)),
        "maize": ((4, 18), (5, 6)),
    }
    start_md, end_md = windows[crop]
    return date(year, *start_md), date(year, *end_md)


def build_fields(n_farms: int, rng: random.Random, missing_rate: float) -> list[FieldSpec]:
    fields: list[FieldSpec] = []
    for farm_i in range(1, n_farms + 1):
        farm_id = client_id(farm_i)
        profile = FARM_PROFILES[(farm_i - 1) % len(FARM_PROFILES)]
        n_fields = rng.randint(3, 4)
        water_source = WATER_SOURCES[(farm_i - 1) % len(WATER_SOURCES)]
        for field_i in range(1, n_fields + 1):
            crop = profile["crops"][(field_i - 1) % len(profile["crops"])]
            soil = profile["soils"][(field_i - 1) % len(profile["soils"])]
            irrigation = profile["irrigation"][(field_i - 1) % len(profile["irrigation"])]
            start, end = planting_window(crop, 2026)
            span = (end - start).days
            planted = start + timedelta(days=rng.randint(0, span))
            fields.append(
                FieldSpec(
                    farm_id=farm_id,
                    field_id=f"{farm_id}_field_{field_i}",
                    crop_type=crop,
                    soil_type=soil,
                    crop_variety_maturity=rng.choice(MATURITIES),
                    irrigation_type=irrigation,
                    water_source=water_source,
                    field_area_ha=round(rng.uniform(3.5, 48.0), 1),
                    plant_month=planted.month,
                    plant_day=planted.day,
                    taw_mm=TAW_MM[soil],
                    et_factor=clip(rng.gauss(ET_FACTOR_MEAN, ET_FACTOR_SD), 0.75, 1.25),
                    sensor_bias=rng.gauss(0.0, SENSOR_BIAS_SD),
                    missing_rate=missing_rate,
                    stuck_p=0.012,
                )
            )
    return fields


def farm_rain_series(
    weather: list[dict], n_farms: int, rng: random.Random
) -> dict[tuple[str, str], float]:
    factors = {}
    for i in range(1, n_farms + 1):
        profile = FARM_PROFILES[(i - 1) % len(FARM_PROFILES)]
        # Sand-specialist farms catch a bit less of the regional rain.
        if profile["soils"][0] == "sand":
            factors[client_id(i)] = rng.uniform(0.62, 0.92)
        elif profile["soils"][0] == "loam":
            factors[client_id(i)] = rng.uniform(0.95, 1.22)
        else:
            factors[client_id(i)] = rng.uniform(0.80, 1.10)
    out: dict[tuple[str, str], float] = {}
    for row in weather:
        for farm_id, factor in factors.items():
            gauge = max(0.0, row["rain_mm"] * factor + rng.gauss(0.0, FARM_RAIN_NOISE_SD))
            out[(farm_id, row["date"])] = round(gauge, 1)
    return out


def maybe_irrigate(
    need: int,
    true_moisture: float,
    spec: FieldSpec,
    rng: random.Random,
) -> float:
    if need == 2:
        irrigate = True
    elif need == 1:
        irrigate = rng.random() < MEDIUM_IRRIGATE_P[spec.irrigation_type]
    else:
        irrigate = False
    if not irrigate:
        return 0.0
    refill_pct = max(0.0, 85.0 - true_moisture)
    mm = refill_pct / 100.0 * spec.taw_mm
    mm = min(mm, MAX_IRRIGATION_MM[spec.irrigation_type])
    if mm < 4.0:
        return 0.0
    return round(mm, 1)


def simulate(
    spec: FieldSpec,
    weather: list[dict],
    farm_rain: dict[tuple[str, str], float],
    rng: random.Random,
) -> list[dict]:
    rows: list[dict] = []
    by_year: dict[int, list[dict]] = {}
    for met in weather:
        by_year.setdefault(date.fromisoformat(met["date"]).year, []).append(met)

    for year, year_weather in by_year.items():
        true_m = clip(rng.uniform(52.0, 78.0), 0.0, 100.0)
        last_irrigation_date: date | None = None
        last_irrigation_mm = 0.0
        last_sensor: float | None = None
        stuck_left = 0
        drift = 0.0
        try:
            planting = date(year, spec.plant_month, spec.plant_day)
        except ValueError:
            planting = date(year, spec.plant_month, 28)

        for met in year_weather:
            day = date.fromisoformat(met["date"])
            dap = (day - planting).days
            rain = farm_rain[(spec.farm_id, met["date"])]
            if dap < 0:
                et_mm = float(met["et0_mm"]) * 0.15 * spec.et_factor
                true_m += 100.0 * (rain - et_mm) / spec.taw_mm
                true_m = clip(true_m, 0.0, 100.0)
                continue

            stage = growth_stage(spec.crop_type, spec.crop_variety_maturity, dap)
            days_since = (
                (day - last_irrigation_date).days if last_irrigation_date else dap
            )
            need = irrigation_need(
                true_m, spec.crop_type, stage, spec.crop_variety_maturity
            )
            drift += rng.gauss(0.0, 0.04)
            live = clip(
                true_m + spec.sensor_bias + drift + rng.gauss(0.0, SENSOR_DAILY_SD),
                0.0,
                100.0,
            )
            if stuck_left > 0 and last_sensor is not None:
                sensor_value: float | None = last_sensor
                stuck_left -= 1
            elif rng.random() < spec.stuck_p:
                sensor_value = last_sensor if last_sensor is not None else live
                stuck_left = rng.randint(3, 8)
            else:
                sensor_value = live
                last_sensor = live

            if rng.random() < spec.missing_rate:
                written: float | str = ""
            else:
                written = round(sensor_value, 1)
                last_sensor = float(written)

            rows.append(
                {
                    "farm_id": spec.farm_id,
                    "field_id": spec.field_id,
                    "date": met["date"],
                    "soil_moisture_pct_nfk": written,
                    "growth_stage": stage,
                    "crop_type": spec.crop_type,
                    "days_since_last_irrigation": days_since,
                    "previous_irrigation_mm": last_irrigation_mm,
                    "onfarm_rain_gauge_mm": rain,
                    "soil_type": spec.soil_type,
                    "crop_variety_maturity": spec.crop_variety_maturity,
                    "days_after_planting": dap,
                    "irrigation_type": spec.irrigation_type,
                    "water_source": spec.water_source,
                    "field_area_ha": spec.field_area_ha,
                    "irrigation_need": need,
                }
            )

            applied = maybe_irrigate(need, true_m, spec, rng)
            et_mm = KC[spec.crop_type][stage] * float(met["et0_mm"]) * spec.et_factor
            true_m += 100.0 * (rain + applied - et_mm) / spec.taw_mm
            true_m = clip(true_m, 0.0, 100.0)
            if applied:
                last_irrigation_mm = applied
                last_irrigation_date = day

    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def schema_document(years: list[int], n_farms: int) -> dict:
    return {
        "dataset": "onfarm_irrigation_need_v3",
        "spec": "DATA_SPEC.md",
        "grain": "one row = one field on one date (morning observation)",
        "season_window": {
            "start_md": list(SEASON_START_MD),
            "end_md": list(SEASON_END_MD),
        },
        "years": years,
        "n_farms_default": n_farms,
        "client_files": [COMBINED_FILENAME],
        "client_id": {"column": "partition", "values": list(range(1, n_farms + 1))},
        "label_mapping": {str(k): v for k, v in LABEL_MAPPING.items()},
        "allowed_values": {
            "soil_type": list(SOILS),
            "irrigation_type": list(IRRIGATION_TYPES),
        },
        "dropped_from_export": [
            "date",
            "growth_stage",
            "crop_type",
            "days_since_last_irrigation",
            "field_id",
            "crop_variety_maturity",
            "days_after_planting",
        ],
        "agent_only_columns": ["water_source", "field_area_ha"],
        "feature_columns": [
            "soil_moisture_pct_nfk",
            "previous_irrigation_mm",
            "onfarm_rain_gauge_mm",
            "soil_type",
            "irrigation_type",
        ],
        "columns": EXPORT_COLUMNS,
        "soil_moisture_pct_nfk": {
            "clip": [0, 100],
            "source": "noisy sensor",
            "missing": "empty cell (gap or dropout)",
        },
        "partition_design": {
            "profiles": [
                {
                    "farm_offset": i + 1,
                    "crops": list(p["crops"]),
                    "soils": list(p["soils"]),
                    "irrigation": list(p["irrigation"]),
                }
                for i, p in enumerate(FARM_PROFILES)
            ]
        },
        "label": {
            "column": "irrigation_need",
            "computed_from": "true_moisture_pct_nfk",
            "thresholds": THRESHOLDS,
            "potato_maturity_delta": POTATO_THRESHOLD_DELTA,
        },
        "kc": KC,
        "stage_lengths_medium": STAGE_LENGTHS,
        "maturity_length_scale": MATURITY_LENGTH_SCALE,
        "taw_mm": TAW_MM,
    }


def validate(rows: list[dict], n_farms: int) -> None:
    if not rows:
        raise SystemExit("no rows generated")
    allowed_need = {0, 1, 2}
    farm_day_rain: dict[tuple[str, str], float] = {}
    for row in rows:
        need = int(row["irrigation_need"])
        if need not in allowed_need:
            raise SystemExit(f"bad label {need}")
        raw = row["soil_moisture_pct_nfk"]
        if raw != "":
            moisture = float(raw)
            if moisture < 0 or moisture > 100:
                raise SystemExit(f"moisture out of range {moisture}")
        if row["crop_type"] not in CROPS:
            raise SystemExit(row["crop_type"])
        if row["soil_type"] not in SOILS:
            raise SystemExit(row["soil_type"])
        if row["irrigation_type"] not in IRRIGATION_TYPES:
            raise SystemExit(row["irrigation_type"])
        if row["growth_stage"] not in STAGES:
            raise SystemExit(row["growth_stage"])
        key = (row["farm_id"], row["date"])
        rain = float(row["onfarm_rain_gauge_mm"])
        if key in farm_day_rain and farm_day_rain[key] != rain:
            raise SystemExit(f"farm-day rain mismatch {key}")
        farm_day_rain[key] = rain
        if int(row["days_since_last_irrigation"]) == 99:
            raise SystemExit("sentinel 99 is not allowed")
        if float(row["previous_irrigation_mm"]) == 0.0:
            if int(row["days_since_last_irrigation"]) != int(row["days_after_planting"]):
                raise SystemExit(
                    "never-irrigated days_since_last_irrigation must equal days_after_planting"
                )
    farms = {row["farm_id"] for row in rows}
    expected = {client_id(i) for i in range(1, n_farms + 1)}
    if farms != expected:
        raise SystemExit(f"farm ids {farms} != {expected}")


def parse_years(raw: str) -> list[int]:
    years = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not years:
        raise SystemExit("--years must list at least one year")
    return years


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate irrigation tables from DATA_SPEC.md")
    parser.add_argument("--farms", type=int, default=DEFAULT_FARMS)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument(
        "--years",
        type=str,
        default=",".join(str(y) for y in DEFAULT_YEARS),
        help="Comma-separated years; each uses the Apr 15–Sep 30 window",
    )
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=DEFAULT_MISSING_RATE,
        help="Per-row probability that soil_moisture_pct_nfk is left empty",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for partitions.csv, schema.json, regional_weather.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.farms < 1:
        raise SystemExit("--farms must be >= 1")
    if not 0.0 <= args.missing_rate < 1.0:
        raise SystemExit("--missing-rate must be in [0, 1)")
    years = parse_years(args.years)
    rng = random.Random(args.seed)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    for pattern in ("farm_*.csv", "partition_*_farmer_*.csv", "centralized_baseline.csv"):
        for old in out.glob(pattern):
            old.unlink()
    combined = out / COMBINED_FILENAME
    if combined.exists():
        combined.unlink()

    weather = regional_weather(years, rng)
    fields = build_fields(args.farms, rng, args.missing_rate)
    farm_rain = farm_rain_series(weather, args.farms, rng)

    all_rows: list[dict] = []
    by_farm: dict[str, list[dict]] = {client_id(i): [] for i in range(1, args.farms + 1)}
    for spec in fields:
        rows = simulate(spec, weather, farm_rain, rng)
        all_rows.extend(rows)
        by_farm[spec.farm_id].extend(rows)

    validate(all_rows, args.farms)

    write_csv(out / "regional_weather.csv", weather, ["date", "et0_mm", "rain_mm"])
    write_csv(out / COMBINED_FILENAME, [export_row(row) for row in all_rows], EXPORT_COLUMNS)
    (out / "schema.json").write_text(
        json.dumps(schema_document(years, args.farms), indent=2) + "\n"
    )

    counts = {name: 0 for name in LABEL_MAPPING.values()}
    for row in all_rows:
        counts[LABEL_MAPPING[int(row["irrigation_need"])]] += 1
    missing = sum(1 for row in all_rows if row["soil_moisture_pct_nfk"] == "")
    crops_by_farm = {
        farm_id: sorted({row["crop_type"] for row in farm_rows})
        for farm_id, farm_rows in by_farm.items()
    }
    high_by_farm = {
        farm_id: sum(int(row["irrigation_need"]) == 2 for row in farm_rows)
        for farm_id, farm_rows in by_farm.items()
    }
    summary = {
        "n_rows": len(all_rows),
        "n_farms": args.farms,
        "n_fields": len({row["field_id"] for row in all_rows}),
        "years": years,
        "label_counts": counts,
        "high_rate": round(counts["High"] / len(all_rows), 4) if all_rows else 0,
        "missing_moisture_rows": missing,
        "crops_by_farm": crops_by_farm,
        "high_rows_by_farm": high_by_farm,
        "seed": args.seed,
        "out_file": COMBINED_FILENAME,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
