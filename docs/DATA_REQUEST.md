# Data request

For whoever is extending `data/generate_data.py`.

**Most of the previous round of asks landed.** The farms specialise, there are
two seasons, High-need days come from drought rather than moved thresholds, and
sensors now drop out and stick. All of that made the project's central claim
measurable. Thank you.

**The one remaining ask is farm count: 2 → 8–12.**

Everything below is measured against the current dataset. Reproduction commands
are at the bottom.

---

## What the current dataset supports

| | value |
|---|---|
| Farms (Flower clients) | **2** — farmer_1 potato/sand, farmer_2 maize/loam |
| Rows | 1,946 across 6 fields, seasons 2025 + 2026 |
| Split | train 2025 (973 rows), test 2026 (973 rows) |
| Test label balance | Low 645 · Medium 284 · **High 44** |
| Always-predict-Low accuracy | 66.3% — which is why we report macro-F1 |
| Federated macro-F1 | **0.736** (farmer_1 alone 0.596, farmer_2 alone 0.666) |
| Federated High recall | 68% — 30 of 44 |

---

## What the last round fixed

**Specialisation made the argument provable.** This is the single most useful
change. Because farmer_1 has only ever grown potato, we can measure what a solo
model does on ground it has never farmed:

```
Scored on farmer_2's maize fields (2026, 467 rows):
  farmer_1 alone, never seen maize   macro-F1 0.480
  federated                          macro-F1 0.664   (+38%)
  farmer_2's own model               macro-F1 0.678
```

That is now the centre of the pitch. With the old four-farm data — where every
farm grew every crop — the best we could show was a marginal gain, and one farm
was actually *worse off* for joining.

**Two seasons gave an honest split.** Train 2025, test 2026: "predict a year you
have not seen." The previous single-season data forced a field-level holdout
that left only 13 High rows in the test set; there are now 44.

**Sensor dropout is handled.** ~2% of `soil_moisture_pct_nfk` cells are empty.
They impute to the regional mean, which is zero in standardized space, so the
feature stops contributing rather than the row being dropped.

---

## The remaining ask

### ① More farms: 2 → 8–12

Two clients is a very small federation and the average is correspondingly noisy.
On the previous four-farm dataset, farm count was the only lever that measurably
moved the result:

```
2 farms: macro-F1 0.644
3 farms: macro-F1 0.653
4 farms: macro-F1 0.704
```

**Caveat on that evidence:** those numbers are from the *old* dataset and cannot
be re-measured now, because there are only two partitions to subset. Treat it as
a strong prior, not a current measurement.

`generate_data.py --farms 8` already supports this, and `schema.json`'s
`partition_design` has profiles defined out to at least five farms. Suggested
shape, keeping specialisation intact:

- 8–12 farms, each still committed to one crop and one soil
- at least two farms sharing a crop, so we can separate "learned the crop" from
  "learned that farm"
- a deliberately small farm — one or two fields — to show federation rescuing a
  participant who could never train alone

### ② Slightly more High-need days, if it is cheap

69 across both farms and both seasons, 44 of them in the test year. Workable,
but High recall still moves ~2 percentage points per row. Not urgent, and **not
worth touching the thresholds in `DATA_SPEC.md` for** — more drought periods, as
you did last time, is the right mechanism.

### ③ Nothing else

A third season, more rows per farm, or more fields per farm would not help. The
model saturates well below the data we already have: the train/test macro-F1 gap
is 0.775 / 0.736, so it is underfitting, not short of examples.

---

## What not to change

Per `DATA_SPEC.md`: do not rename columns, and do not alter the numeric
thresholds, Kc values, stage lengths, soil TAW, or maturity modifiers.

Ask ① needs none of that — it is the `--farms` flag plus partition profiles.

## One thing that would break us

The column set and `schema.json`'s `allowed_values` are baked into the model's
feature encoding, and the encoder is checked against the model's own feature
count at inference. **Adding a new crop or soil type to `allowed_values` changes
the feature vector length and invalidates every trained model.** That is fine —
it just means a retrain plus `scripts/sync-model.sh`. Worth a heads-up rather
than a surprise.

---

## A caveat that belongs in the pitch

This dataset is **generated, not measured**. Every accuracy figure describes how
well a model recovers `generate_data.py`'s rules — not how it would advise a
real farmer. The agronomic structure is plausible, but real-world error would
very likely be larger. We say this on stage before a judge asks.

---

## Reproducing the numbers

```bash
cd federated

# headline table, cross-crop figures, privacy audit
uv run flwr run . local-sim --federation-config 'num-supernodes=2' --stream

# the cross-crop measurement on its own
uv run python -c "
import sys, json; sys.path.insert(0,'.')
import numpy as np
from grwflwr.data import farm_data
from grwflwr.model import macro_f1
g = json.load(open('$HOME/.grwflwr/global_model.json'))
s = json.load(open('$HOME/.grwflwr/solo_models.json'))
fed = [np.array(g['weights']), np.array(g['bias'])]
solo0 = [np.array(s['models']['0']['weights']), np.array(s['models']['0']['bias'])]
_, _, xm, ym = farm_data(1)
print('farmer_1 alone on maize:', round(macro_f1(solo0, xm, ym), 3))
print('federated on maize     :', round(macro_f1(fed, xm, ym), 3))
"
```
