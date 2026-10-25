# Weight API

A REST service around the same library call the study uses
(`features_from_clouds` → `WeightModel.predict`), so the API computes features
exactly as training did. The presentation website and the farm web app at
`/app/` are clients of this API.

```bash
pip install -e ".[api]"
cattle3d serve                                   # http://127.0.0.1:8000, docs at /docs
CATTLE3D_API_KEY=... cattle3d serve --host 0.0.0.0 --db /srv/farm.sqlite3
```

The presentation website is served at `/` and the thesis figures at
`/figures/{uk,en}/` (both public; `--no-site` turns the site off). Its live
demo calls this API and asks for the key when one is set.

Serving beyond localhost requires `CATTLE3D_API_KEY`. Every `/v1` route then
needs the `X-API-Key` header. Herds, scans, weighings and calibrations are
kept in one SQLite file. The model artifact is never modified by the API.

## How a farm uses it

1. **Create the herd**: `POST /v1/herds {"name": "North barn"}`.
2. **Collect reference weighings**: scan about 20 animals and send
   `scale_weight_kg` with each scan, or attach the weight later with
   `PUT /v1/scans/{id}/weighing`. Until the herd is calibrated, every weight
   is returned with status `flagged` and the reason "herd not calibrated":
   the study found that a model moved to a new herd is off by a shift of
   30–107 kg.
3. **Calibrate**: `POST /v1/herds/{id}/calibration` fits
   `weight = slope × model + intercept` on the latest weighed scan of each
   animal. With fewer than 10 animals only the intercept is fitted, and the
   minimum is 3. In the study, 20 animals brought the error back to within
   1 MAPE point of a herd-specific model in 90–96 % of random reference sets.
4. **Scan daily**: `POST /v1/herds/{id}/scans` (multipart: `animal_tag`,
   `left`, `right`, `top` PLY files). It takes about 1.2 s per animal on one
   CPU core.
5. **Spot-check**: put a few animals on the scale now and then and attach
   their weights. `GET /v1/herds/{id}` reports drift once 5 spot weighings
   exist. Drift is declared when the recent mean residual is at least 10 kg
   and its 95 % CI excludes zero; then recalibrate. The 10 kg threshold is an
   engineering default, not a validated limit.

## What a scan returns

| field | meaning |
|---|---|
| `status` | `trusted`: use the weight. `flagged`: a weight is returned, but `reasons` says why it is outside the validated range (herd not calibrated, body volume outside the training range, too few points in a view). `refused`: no weight; rescan. |
| `weight_kg`, `lower_kg`, `upper_kg` | herd-corrected weight and its 90 % interval |
| `raw_weight_kg` | model output before the herd correction (kept so the herd can be recalibrated without re-processing clouds) |
| `calibration_id` | which herd correction was applied |

The interval after calibration comes from leave-one-out residuals of the
correction on the reference animals (split conformal). It is never narrower
than the model's own ±42 kg. With 20 reference animals expect about ±50–95 kg,
the same range H4 in `reports/results.md` found for transfer.

Refused scans are stored as well, so a station's refusal rate can be watched
(`scans_by_status` in the herd status). Each scan also keeps its feature row
for later retraining.

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/health` | liveness, no key |
| GET | `/v1/model` | training herds, weight range, interval, CV metrics |
| POST / GET | `/v1/herds` | create / list herds |
| GET | `/v1/herds/{id}` | calibration, reference count, scans by status, drift |
| POST | `/v1/herds/{id}/calibration` | fit a new herd correction |
| POST | `/v1/herds/{id}/scans` | weigh one animal from three clouds (`preview=true` adds what the cameras saw) |
| GET | `/v1/herds/{id}/scans?limit=50` | the herd's recent scans, newest first |
| GET | `/v1/scans/{id}` | one scan |
| PUT | `/v1/scans/{id}/weighing` | attach a scale reading |
| GET | `/v1/herds/{id}/animals/{tag}/scans` | an animal's weight history |

Example:

```bash
curl -H "X-API-Key: $KEY" -F animal_tag=UA-0042 \
     -F left=@left.ply -F right=@right.ply -F top=@top.ply \
     http://localhost:8000/v1/herds/1/scans
```

```json
{"id": 31, "status": "trusted", "reasons": [], "weight_kg": 618.4,
 "lower_kg": 558.5, "upper_kg": 678.3, "raw_weight_kg": 612.3,
 "calibration_id": 1, "animal_tag": "UA-0042", "scale_weight_kg": null, ...}
```

With the form field `preview=true` the response also carries `preview`
(null for a refused scan): a depth thumbnail per view with the animal found in
it, the top-view body points (base64 int16 millimetres, x along the body,
y lateral, z height), the 64 trunk cross-sections that are integrated into the
volume, the main body measures, and processing times. The farm web app draws
its 3-D view from this; it adds about 160 KB.

`cattle3d demo` adds three read-only routes for the defence demo:
`GET /v1/demo` (demo farms and the held-out sample cows),
`GET /v1/demo/samples/{id}/{left|right|top}.ply` (their clouds) and
`GET /v1/demo/qr.svg` (the phone address as a QR code).

Errors: 401 wrong key, 404 unknown herd or scan, 409 duplicate herd name,
413 file over 16 MB, 422 invalid request (not a dense 512×424 PLY, a scale
weight outside 20–2000 kg, too few weighings to calibrate).

## Code

| module | owns |
|---|---|
| `cattle3d/modeling/calibration.py` | the herd correction, its interval, the drift check (pure functions) |
| `cattle3d/api/store.py` | SQLite schema and queries |
| `cattle3d/api/service.py` | scan → decision, reference selection, herd status; no HTTP |
| `cattle3d/api/app.py` | FastAPI routes and schemas only |
