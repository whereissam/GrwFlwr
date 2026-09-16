"""Synthetic per-farm irrigation telemetry.

The point of this module is to make federation *matter*. Each farm observes a
different slice of the regional feature space (its soil type and its position on
the canal fix what conditions it ever sees), but the underlying agronomic rule
for "is it safe to irrigate right now" is regional and identical everywhere.

A farm training alone fits the rule well on its own slice and extrapolates badly
off it. FedAvg across all four farms recovers the regional rule. That gap is the
demo, and it is a real effect of the data geometry, not a rigged metric.
"""

from __future__ import annotations

import numpy as np

# Feature order is fixed and shared by every module that touches the model.
FEATURES = [
    "soil_moisture_pct",
    "rain_forecast_48h_mm",
    "et0_mm_day",
    "temp_c",
    "crop_coefficient",
    "days_since_irrigation",
    "allocation_headroom_pct",
]
N_FEATURES = len(FEATURES)

# Regional agronomic rule, in standardized feature space. Signs are what an
# agronomist would expect: wet soil and coming rain argue against irrigating,
# high evapotranspiration, heat, a thirsty crop stage and a long dry spell argue
# for it, and you need allocation headroom to be allowed to.
_TRUE_W = np.array([-1.35, -1.10, 0.95, 0.55, 0.85, 0.70, 1.25], dtype=np.float64)
_TRUE_B = -0.15

# Per-farm conditions. Each farm is described by the *direction* of its typical
# conditions (what kind of place it is) and by how far that places it from the
# regional decision boundary. A farm sitting far from the boundary almost always
# sees the same answer, so on its own it cannot learn where the boundary is --
# which is precisely the situation federation is supposed to rescue.
_FARM_DIRECTIONS = [
    ("Farm 1 - sandy upland, canal head",
     [-1.0, -0.6, 0.7, 0.6, 0.2, 0.5, 1.0], 3.2),
    ("Farm 2 - clay lowland, canal head",
     [1.1, 0.5, -0.5, -0.3, 0.2, -0.6, 0.9], -2.6),
    ("Farm 3 - loam midslope, canal tail",
     [-0.7, -0.5, 0.6, 0.4, 0.5, 0.6, -0.9], 2.2),
    ("Farm 4 - terraced orchard, canal tail",
     [0.5, 1.0, -0.4, -0.6, -0.3, -0.5, -0.9], -3.0),
]

# How varied each farm's own conditions are, in standardized units.
_SIGMA = 0.8


def _mu_at_distance(direction: list[float], signed_distance: float) -> np.ndarray:
    """Scale a conditions-direction so the farm sits `signed_distance` off the rule."""
    v = np.array(direction, dtype=np.float64)
    return v * ((signed_distance - _TRUE_B) / float(v @ _TRUE_W))


FARM_PROFILES = [
    {
        "name": name,
        "mu": _mu_at_distance(direction, distance),
        "sigma": np.full(N_FEATURES, _SIGMA),
        "boundary_distance": distance,
    }
    for name, direction, distance in _FARM_DIRECTIONS
]
NUM_FARMS = len(FARM_PROFILES)

# Label noise: real agronomy is not a clean hyperplane.
_LABEL_NOISE = 0.35


def _label(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the regional rule with noise, returning 0/1 labels."""
    logits = x @ _TRUE_W + _TRUE_B
    logits = logits + rng.normal(0.0, _LABEL_NOISE, size=logits.shape)
    return (logits > 0.0).astype(np.float64)


def farm_data(
    partition_id: int, n: int = 600, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, y_train, x_test, y_test) for one farm.

    The test split here is *local*: it comes from the same narrow slice as the
    training data, so a solo model scores well on it. That is exactly why the
    honest comparison lives in `region_data`, not here.
    """
    profile = FARM_PROFILES[partition_id % NUM_FARMS]
    rng = np.random.default_rng(seed + 1000 * partition_id)

    x = clip_physical(rng.normal(profile["mu"], profile["sigma"], size=(n, N_FEATURES)))
    y = _label(x, rng)

    split = int(0.8 * n)
    return x[:split], y[:split], x[split:], y[split:]


def region_data(n: int = 2000, seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
    """Return a region-wide held-out set drawn from every farm's conditions.

    This is the set the regional authority cares about and the set the agent
    will be answering questions against, because a farmer can ask about
    conditions their own farm has not happened to see yet.
    """
    rng = np.random.default_rng(seed)
    per = max(1, n // NUM_FARMS)

    chunks = []
    for profile in FARM_PROFILES:
        chunks.append(rng.normal(profile["mu"], profile["sigma"], size=(per, N_FEATURES)))

    x = clip_physical(np.vstack(chunks))
    rng.shuffle(x)
    return x, _label(x, rng)

# Human-readable units for each feature: (mean, std) used to convert real farm
# readings into the standardized space the model is trained in. The agent shows
# farmers "soil moisture 18%", not "-1.4 standardized", so both sides of the
# system have to agree on this mapping -- it ships inside the model JSON.
FEATURE_STATS = {
    "soil_moisture_pct": (28.0, 9.0),
    "rain_forecast_48h_mm": (6.0, 6.5),
    "et0_mm_day": (4.2, 1.6),
    "temp_c": (24.0, 6.0),
    "crop_coefficient": (0.85, 0.22),
    "days_since_irrigation": (5.0, 3.2),
    "allocation_headroom_pct": (45.0, 22.0),
}


# Physically possible range for each reading. Gaussian sampling happily produces
# 102% allocation headroom or negative rainfall; clamping happens at generation
# so the model trains and scores on the same valid numbers the farmer is shown.
# Clamping at display time instead would mean the shown reading is not the one
# that produced the prediction.
FEATURE_BOUNDS = {
    "soil_moisture_pct": (2.0, 55.0),
    "rain_forecast_48h_mm": (0.0, 60.0),
    "et0_mm_day": (0.3, 11.0),
    "temp_c": (-2.0, 48.0),
    "crop_coefficient": (0.2, 1.35),
    "days_since_irrigation": (0.0, 35.0),
    "allocation_headroom_pct": (0.0, 100.0),
}


def clip_physical(x: np.ndarray) -> np.ndarray:
    """Clamp standardized rows so their real-unit readings stay possible."""
    out = np.array(x, dtype=np.float64, copy=True)
    flat = out.reshape(-1, N_FEATURES)
    for i, name in enumerate(FEATURES):
        mean, std = FEATURE_STATS[name]
        lo, hi = FEATURE_BOUNDS[name]
        np.clip(flat[:, i], (lo - mean) / std, (hi - mean) / std, out=flat[:, i])
    return out.reshape(x.shape)


def to_standard(readings: dict[str, float]) -> np.ndarray:
    """Convert a dict of real-world readings into a standardized feature vector."""
    out = np.empty(N_FEATURES, dtype=np.float64)
    for i, name in enumerate(FEATURES):
        mean, std = FEATURE_STATS[name]
        out[i] = (float(readings[name]) - mean) / std
    return out


# Decimal places per feature: enough that a displayed reading still converts
# back to the value the model actually scored. A crop coefficient has a small
# std, so rounding it to 1dp would move it a quarter of a standard deviation.
_PRECISION = {
    "soil_moisture_pct": 2,
    "rain_forecast_48h_mm": 2,
    "et0_mm_day": 2,
    "temp_c": 2,
    "crop_coefficient": 3,
    "days_since_irrigation": 2,
    "allocation_headroom_pct": 1,
}


def to_human(vector: np.ndarray) -> dict[str, float]:
    """Inverse of `to_standard`, for turning generated rows back into readings."""
    out = {}
    for i, name in enumerate(FEATURES):
        mean, std = FEATURE_STATS[name]
        out[name] = round(float(vector[i]) * std + mean, _PRECISION[name])
    return out
