#!/usr/bin/env python3
"""Generate the irrigation table. Constants must match data/DATA_SPEC.md."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

SEASON_START = date(2026, 4, 15)
SEASON_END = date(2026, 9, 30)

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


def regional_weather(rng: random.Random) -> list[dict]:
    rows = []
    for day in daterange(SEASON_START, SEASON_END):
        doy = day.timetuple().tm_yday
        seasonal = 2.2 + 2.4 * math.sin(2 * math.pi * (doy - 105) / 365)
        et0 = clip(seasonal + rng.gauss(0, 0.45), 0.8, 6.8)
        wet_p = 0.22 + 0.08 * math.cos(2 * math.pi * (doy - 60) / 365)
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
    planting_date: date
    taw_mm: float
    et_factor: float
    sensor_bias: float


def build_fields(n_farms: int, rng: random.Random) -> list[FieldSpec]:
    fields: list[FieldSpec] = []
    planting_windows = {
        "potato": (date(2026, 4, 8), date(2026, 4, 22)),
        "sugar_beet": (date(2026, 4, 1), date(2026, 4, 18)),
        "maize": (date(2026, 4, 18), date(2026, 5, 6)),
    }
    for farm_i in range(1, n_farms + 1):
        farm_id = f"farm_{farm_i}"
        n_fields = rng.randint(3, 5)
        water_source = WATER_SOURCES[(farm_i - 1) % len(WATER_SOURCES)]
        for field_i in range(1, n_fields + 1):
            crop = CROPS[(farm_i + field_i) % len(CROPS)]
            soil = rng.choice(SOILS)
            start, end = planting_windows[crop]
            span = (end - start).days
            planting = start + timedelta(days=rng.randint(0, span))
            fields.append(
                FieldSpec(
                    farm_id=farm_id,
                    field_id=f"{farm_id}_field_{field_i}",
                    crop_type=crop,
                    soil_type=soil,
                    crop_variety_maturity=rng.choice(MATURITIES),
                    irrigation_type=IRRIGATION_TYPES[(farm_i + field_i) % 3],
                    water_source=water_source,
                    field_area_ha=round(rng.uniform(3.5, 48.0), 1),
                    planting_date=planting,
                    taw_mm=TAW_MM[soil],
                    et_factor=clip(rng.gauss(ET_FACTOR_MEAN, ET_FACTOR_SD), 0.75, 1.25),
                    sensor_bias=rng.gauss(0.0, SENSOR_BIAS_SD),
                )
            )
    return fields


def farm_rain_series(
    weather: list[dict], n_farms: int, rng: random.Random
) -> dict[tuple[str, str], float]:
    factors = {
        f"farm_{i}": rng.uniform(0.82, 1.18) for i in range(1, n_farms + 1)
    }
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
    true_m = clip(rng.uniform(52.0, 78.0), 0.0, 100.0)
    last_irrigation_date: date | None = None
    last_irrigation_mm = 0.0
    rows: list[dict] = []

    for met in weather:
        day = date.fromisoformat(met["date"])
        dap = (day - spec.planting_date).days
        if dap < 0:
            et_mm = float(met["et0_mm"]) * 0.15 * spec.et_factor
            rain = farm_rain[(spec.farm_id, met["date"])]
            true_m += 100.0 * (rain - et_mm) / spec.taw_mm
            true_m = clip(true_m, 0.0, 100.0)
            continue

        stage = growth_stage(spec.crop_type, spec.crop_variety_maturity, dap)
        rain = farm_rain[(spec.farm_id, met["date"])]
        days_since = (
            (day - last_irrigation_date).days if last_irrigation_date else dap
        )
        need = irrigation_need(
            true_m, spec.crop_type, stage, spec.crop_variety_maturity
        )
        sensor = clip(
            true_m + spec.sensor_bias + rng.gauss(0.0, SENSOR_DAILY_SD),
            0.0,
            100.0,
        )
        rows.append(
            {
                "farm_id": spec.farm_id,
                "field_id": spec.field_id,
                "date": met["date"],
                "soil_moisture_pct_nfk": round(sensor, 1),
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


def schema_document() -> dict:
    return {
        "dataset": "onfarm_irrigation_need_v2",
        "spec": "DATA_SPEC.md",
        "grain": "one row = one field on one date (morning observation)",
        "season": {"start": SEASON_START.isoformat(), "end": SEASON_END.isoformat()},
        "label_mapping": {str(k): v for k, v in LABEL_MAPPING.items()},
        "allowed_values": {
            "crop_type": list(CROPS),
            "soil_type": list(SOILS),
            "irrigation_type": list(IRRIGATION_TYPES),
            "growth_stage": list(STAGES),
            "crop_variety_maturity": list(MATURITIES),
        },
        "agent_only_columns": ["water_source", "field_area_ha"],
        "feature_columns": [
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
        ],
        "columns": COLUMN_ORDER,
        "soil_moisture_pct_nfk": {"clip": [0, 100], "source": "noisy sensor"},
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
        moisture = float(row["soil_moisture_pct_nfk"])
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
    expected = {f"farm_{i}" for i in range(1, n_farms + 1)}
    if farms != expected:
        raise SystemExit(f"farm ids {farms} != {expected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate irrigation tables from DATA_SPEC.md")
    parser.add_argument("--farms", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for farm_N.csv, schema.json, regional_weather.csv, centralized_baseline.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.farms < 1:
        raise SystemExit("--farms must be >= 1")
    rng = random.Random(args.seed)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    weather = regional_weather(rng)
    fields = build_fields(args.farms, rng)
    farm_rain = farm_rain_series(weather, args.farms, rng)

    all_rows: list[dict] = []
    by_farm: dict[str, list[dict]] = {f"farm_{i}": [] for i in range(1, args.farms + 1)}
    for spec in fields:
        rows = simulate(spec, weather, farm_rain, rng)
        all_rows.extend(rows)
        by_farm[spec.farm_id].extend(rows)

    validate(all_rows, args.farms)

    write_csv(out / "regional_weather.csv", weather, ["date", "et0_mm", "rain_mm"])
    for farm_id, farm_rows in by_farm.items():
        index = farm_id.split("_", 1)[1]
        write_csv(out / f"farm_{index}.csv", farm_rows, COLUMN_ORDER)
    write_csv(out / "centralized_baseline.csv", all_rows, COLUMN_ORDER)
    (out / "schema.json").write_text(json.dumps(schema_document(), indent=2) + "\n")

    counts = {name: 0 for name in LABEL_MAPPING.values()}
    for row in all_rows:
        counts[LABEL_MAPPING[int(row["irrigation_need"])]] += 1
    summary = {
        "n_rows": len(all_rows),
        "n_farms": args.farms,
        "n_fields": len({row["field_id"] for row in all_rows}),
        "label_counts": counts,
        "seed": args.seed,
        "out": str(out),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
