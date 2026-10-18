# Cattle live-weight estimation from 3-D point clouds

Non-contact estimation of a cow's live weight from three Kinect v2 depth views
(left, right, top), with a prediction interval and a flag that says when the
estimate should not be trusted. Research code for a master's thesis (KhNURE)
and the core library behind the planned API, website and farm tablet app.

## What it does

```
3 PLY clouds ──► floor fit (RANSAC) ──► animal segmentation ──► body frame (metres kept)
            ──► trunk volume, mid-body cross-section, heights, side areas
            ──► ensemble (physics a·V+b, robust linear, LightGBM)
            ──► weight, 90 % interval, in-operating-range flag + reason
```

## Results on public data (`reports/results.md`)

Two public herds captured with the same three-Kinect setup:

| herd | animals | weight | 3-D ensemble MAE | manual tape model MAE | MAPE |
|---|---|---|---|---|---|
| CowDatabase2, Black Angus | 94 | 556–746 kg | 21.4 kg [18.1, 25.0] | 22.6 kg | 3.47 % |
| CowDatabase, Hereford | 101 | 243–605 kg | 17.5 kg [14.5, 21.0] | 22.1 kg | 4.02 % |

Nested 5-fold CV × 5 seeds, one row per animal, bootstrap CIs over animals.

- **Transfer to a new herd without calibration fails, and the failure is bias**
  (|bias|/MAE 0.8–1.0). With 20–30 reference weighings and a slope+intercept
  correction, the average error comes back to within 1 MAPE point of training
  on the herd itself (58–90 % of random reference sets meet that margin).
- **3-D versus manual measurements**: better on the Hereford herd by 4.6 kg
  (bootstrap CI [0.6, 8.7], Wilcoxon p = 0.016, Holm-adjusted p = 0.063); no
  claimable difference on the narrow-range Black Angus herd.
- **The public CowDatabase release mislabels files 22–87** (cloud file N is not
  table row N there). The join was re-derived from heights only; see
  `reports/cowdb1_join_audit.md`. Weight–volume correlation goes from 0.60 to 0.94.

## Install

Python 3.11+.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"          # Windows: .venv\Scripts\pip install -e ".[dev]"
```

## Data

Both datasets are public on GitHub. They are large (≈5 GB each); clone them
anywhere outside this repository:

```bash
git clone --depth 1 https://github.com/ruchaya/CowDatabase2
git clone --depth 1 https://github.com/ruchaya/CowDatabase
```

## Use it

```bash
# 1. Features for both herds -> data/features.csv (a few minutes, 4 processes)
cattle3d extract --cowdb2 ../CowDatabase2 --cowdb1 ../CowDatabase

# 2. Full study -> reports/results.md, reports/figures/*.png, reports/oof_predictions.csv
cattle3d evaluate

# 3. Deployable model trained on both herds -> artifacts/weight_model.joblib
cattle3d train

# 4. Predict one animal
cattle3d predict --left L.ply --right R.ply --top T.ply
```

`predict` prints JSON:

```json
{"weight_kg": 723.0, "lower_kg": 676.3, "upper_kg": 769.6,
 "in_operating_range": true, "size_score": 1.38, "reason": null}
```

### Calibrating for a new herd

Weigh 10–30 animals on a scale, put their clouds and weights in a CSV
(`left,right,top,weight_kg`), and store a correction for that herd:

```bash
cattle3d calibrate --herd my-farm --reference reference.csv
cattle3d predict --herd my-farm --left L.ply --right R.ply --top T.ply
```

With fewer than 10 animals only the intercept is corrected.

### From Python

```python
from pathlib import Path
from cattle3d.modeling.weight_model import WeightModel
from cattle3d.pipeline import features_from_files
from cattle3d.types import View

model = WeightModel.load(Path("artifacts/weight_model.joblib"))
features = features_from_files({View.LEFT: Path("L.ply"), View.RIGHT: Path("R.ply"), View.TOP: Path("T.ply")})
print(model.predict_row(features))
```

## Layout

```
src/cattle3d/
  types.py                 value objects (views, records, prediction)
  io/ply.py                binary PCL PLY reader
  io/datasets.py           CowDatabase2 / CowDatabase loaders and joins
  io/alignment.py          monotone file-to-row alignment (join audit)
  audit.py                 CowDatabase join audit (heights only)
  geometry/segment.py      floor RANSAC, segmentation, body frame
  features/geometry_features.py   volumes, areas, heights
  pipeline.py              clouds -> feature row (shared by CLI, API, experiments)
  modeling/models.py       physics, robust linear, LightGBM, ensemble
  modeling/weight_model.py deployable artifact: interval, range check, herd calibration
  evaluation/              metrics and bootstrap, CV and transfer, study, report
  cli.py                   thin command-line adapter
tests/                     synthetic-cloud unit tests, no dataset needed
reports/                   generated results, never edited by hand
```

## Evaluation rules

- The animal is the unit: no animal is in training and scoring at once, and
  bootstrap resampling draws animals.
- Hyperparameters are tuned inside each training fold (nested CV).
- MAE, RMSE, bias, MAPE, p95 and worst-case error are reported together;
  differences smaller than the bootstrap CI are not claimed.
- Reports are regenerated by `cattle3d evaluate`.

## Tests

```bash
python -m pytest -q
```
