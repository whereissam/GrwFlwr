# Data request

For whoever is extending `data/generate_data.py`.

**Short version: more days of the same data will not help. More farms, more
High-need events, and more difference between farms will.**

Everything below is measured against the current dataset, not guessed. The
commands to reproduce each number are at the bottom.

---

## What the current dataset supports

| | value |
|---|---|
| Farms (Flower clients) | 4 |
| Rows | 2,792 across 17 fields, one season |
| Label balance | 75% Low, 21% Medium, **2% High** |
| High rows per farm | 7–15 |
| Federated macro-F1 | 0.732 |
| Federated High recall | 77% — but that is **10 of 13** test rows |

The model is multinomial logistic regression with inverse-frequency class
weights and moisture×crop / moisture×stage interaction terms.

---

## Three measurements that shape the ask

### 1. The learning curve is flat — more days of the same data do nothing

Training on a random fraction of each farm's rows:

```
 25% of rows (High=10): macro-F1 = 0.709
 50% of rows (High=22): macro-F1 = 0.738
 75% of rows (High=33): macro-F1 = 0.706
100% of rows (High=41): macro-F1 = 0.704
```

A quarter of the data performs the same as all of it. Another full season of the
same shape would change nothing.

### 2. More farms does help

Holding each farm's data constant and varying how many farms take part:

```
2 farms: macro-F1 = 0.644
3 farms: macro-F1 = 0.653
4 farms: macro-F1 = 0.704
```

This is the only lever that moved the number in the direction we want.

### 3. The current bottleneck is model capacity, not data volume

Federated training macro-F1 is 0.726; test is 0.704. Almost no gap, so the model
is **underfitting** rather than running out of examples.

A useful side effect worth knowing: pooling every row into one centralized model
scores **0.680**, which is *lower* than the federated 0.704. Per-client class
weighting balances the rare High class better than a single pooled fit does. So
federating currently costs nothing against centralizing — which is the strongest
claim the project has, and it came from the data, not from tuning.

---

## What to change, in order of value

### ① More farms: 4 → 8–12

The one change with direct evidence behind it. Four clients is very few for
FedAvg and the averaging is noisy. Same total row count spread over more farms
would be better than the same farms getting more rows.

### ② More High-need events

41 High rows in training and 13 in test means every High metric is ±8% noise —
"77% recall" is literally 10 correct out of 13. Any conclusion about the class
the whole project is built around currently rests on 13 observations.

**Careful here:** if this is done by lowering the thresholds in `DATA_SPEC.md`,
it breaks the spec *and* makes the new data incompatible with the old. Prefer
generating more drought periods — longer dry spells, higher ET₀ runs — so High
arises from the weather rather than from moving the goalposts. Worth agreeing
before you start.

### ③ Make the farms genuinely different — the weakest part today

Right now **all four farms grow all three crops**. That makes the partitions
nearly IID, which quietly undercuts the entire premise: if every farm sees
everything, there is much less to gain from federating.

What would help:

- each farm specialises — one mostly potato on sand, another mostly sugar beet
  on clay
- **at least one farm that has never seen a crop another farm grows**
- soil type correlated with farm rather than mixed evenly across all of them

This is the difference between "federation gives +0.03 macro-F1" and "farm_3
literally cannot advise on maize alone".

### ④ A second and third season

One season means no year-to-year drift. Multiple seasons would let us show the
case that actually sells this: *a model trained alone on last year fails when
this year is drier, and the federated one does not.*

### ⑤ Sensor noise, gaps and failures

The data is clean in a way real telemetry never is. No missing readings, no
stuck sensors, no calibration drift. Two reasons to add them:

- the agent currently crashes on a missing field rather than degrading
- "we handle broken sensors" is a credible thing to show a judge

---

## What not to change

Per `DATA_SPEC.md`: do not rename columns, and do not alter the numeric
thresholds, Kc values, stage lengths, soil TAW, or maturity modifiers.

Items ①, ③, ④ and ⑤ need **none** of those changed — they are parameters of the
generator (how many farms, which crops each one grows, which years, what noise),
not of the labelling rule. Only ② risks touching the spec, which is why it needs
a conversation first.

---

## A caveat that belongs in the pitch

This dataset is **generated, not measured**. Every accuracy figure here
describes how well a model recovers `generate_data.py`'s rules — not how well it
would advise a real farmer. The agronomic structure is plausible, but real sensor
noise, failure modes and regional drift are absent, so error on real data would
very likely be larger. Say this on stage before a judge asks.

---

## Reproducing the numbers

```bash
cd federated

# learning curve and farm-count sweep
uv run python -c "
import sys; sys.path.insert(0,'.')
import numpy as np
from grwflwr.data import farm_data, region_data, NUM_FARMS
from grwflwr.model import init_params, train, macro_f1
xr, yr = region_data()
tr = [farm_data(f)[:2] for f in range(NUM_FARMS)]
def fedavg(s, rounds=10, ep=40, lr=0.3):
    p = init_params(); ns=[len(x) for x,_ in s]
    for _ in range(rounds):
        o=[train(p,x,y,ep,lr)[0] for x,y in s]
        p=[sum(a[k]*n for a,n in zip(o,ns))/sum(ns) for k in (0,1)]
    return p
for k in (2,3,4):
    print(k, 'farms:', round(macro_f1(fedavg(tr[:k]), xr, yr), 3))
"

# train-vs-test gap (underfitting check)
uv run flwr run . local-sim --stream
```
