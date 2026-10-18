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

A figure-by-figure explanation of the method and of how it plugs into the
planned API, website and tablet app is in [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md).

## Results on public data (`reports/results.md`)

Two public herds captured with the same three-Kinect setup:

| herd | animals | weight | 3-D ensemble MAE | manual tape model MAE | MAPE |
|---|---|---|---|---|---|
| CowDatabase2, Black Angus | 94 | 556–746 kg | 20.7 kg [17.6, 24.2] | 22.6 kg | 3.34 % |
| CowDatabase, Hereford | 101 | 243–605 kg | 16.1 kg [13.3, 19.4] | 22.1 kg | 3.70 % |

Nested 5-fold CV × 5 seeds, one row per animal, bootstrap CIs over animals.
The model includes a 10-slice back profile from the top view. That feature
set was chosen after comparing eight variants on these same animals, so its
gain (about 1.4 kg on Hereford, 0.7 kg on Black Angus) is exploratory, not
pre-registered.

- **Transfer to a new herd without calibration fails, and the failure is bias**
  for the deployable ensemble (|bias|/MAE 0.88–1.00). With 20–30 reference
  weighings and a slope+intercept correction, the average error comes back to
  within 1 MAPE point of training on the herd itself (90–99 % of random
  reference sets meet that margin).
- **3-D versus manual measurements**: better on the Hereford herd by 6.0 kg
  (bootstrap CI [2.2, 9.9], Wilcoxon p = 0.005, Holm-adjusted p = 0.019,
  exploratory because of the feature selection above); no claimable
  difference on the narrow-range Black Angus herd.
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

### Thesis figures

```bash
# 15 schemes and figures, Ukrainian and English -> reports/figures/thesis/{uk,en}/
cattle3d figures --cowdb2 ../CowDatabase2 --cowdb1 ../CowDatabase
```

Without the dataset paths only the schemes and the result figures are drawn.
Ukrainian captions: [`docs/thesis_figure_captions_uk.md`](docs/thesis_figure_captions_uk.md).

### Calibrating for a new herd

Weigh 10–30 animals on a scale, put their clouds and weights in a CSV
(`left,right,top,weight_kg`), and store a correction for that herd:

```bash
cattle3d calibrate --herd my-farm --reference reference.csv
cattle3d predict --herd my-farm --left L.ply --right R.ply --top T.ply
```

With fewer than 10 animals only the intercept is corrected.

### REST API

```bash
pip install -e ".[api]"
cattle3d serve            # http://127.0.0.1:8000/docs
```

Herds, scans, scale weighings, per-herd calibration and drift checks over
HTTP, stored in SQLite. See [`docs/API.md`](docs/API.md).

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
  modeling/calibration.py  herd correction, its interval, drift check
  api/                     SQLite store, farm service, FastAPI app (docs/API.md)
  evaluation/              metrics and bootstrap, CV and transfer, study, report
  figures/                 thesis schemes and figures (uk/en) drawn from the code itself
  cli.py                   thin command-line adapter
tests/                     synthetic-cloud unit tests, no dataset needed
reports/                   generated results and figures, never edited by hand
docs/                      how the method works, thesis captions
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
