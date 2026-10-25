"""REST API: a thin HTTP adapter over ``WeightService``.

Owns: routes, request/response schemas, upload limits, API-key check and the
mapping of service errors to status codes. No business logic lives here.

    uvicorn: cattle3d serve --model artifacts/weight_model.joblib --db data/farm.sqlite3
    docs:    http://localhost:8000/docs
"""

# No `from __future__ import annotations`: FastAPI must evaluate the
# annotations of routes defined inside create_app, which use its locals.
import hmac
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Security,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from cattle3d import __version__
from cattle3d.api.service import InvalidRequestError, NotFoundError, WeightService
from cattle3d.api.store import Calibration, DuplicateError, Scan, Store
from cattle3d.demo import Demo
from cattle3d.geometry.rig import Rig
from cattle3d.io.ply import PlyFormatError, parse_organized_cloud
from cattle3d.modeling.weight_model import WeightModel
from cattle3d.types import OrganizedCloud, View
from cattle3d.web import SITE_DIR

# A dense Kinect v2 frame with colour is about 3.5 MB; anything far above that
# is not one of our clouds.
MAX_UPLOAD_BYTES = 16 * 1024 * 1024


# ---- schemas --------------------------------------------------------------------------
class HerdIn(BaseModel):
    name: str = Field(min_length=1, max_length=200, examples=["North barn"])


class HerdOut(BaseModel):
    id: int
    name: str
    created_utc: str


class CalibrationOut(BaseModel):
    id: int
    created_utc: str
    slope: float
    intercept_kg: float
    n_reference: int
    interval_half_width_kg: float


class DriftOut(BaseModel):
    n_weighings: int
    mean_residual_kg: float | None = Field(description="scale weight minus prediction")
    ci_low_kg: float | None
    ci_high_kg: float | None
    drifted: bool
    message: str


class HerdStatusOut(BaseModel):
    herd: HerdOut
    calibrated: bool
    calibration: CalibrationOut | None
    n_reference_animals: int
    recommended_reference_animals: int
    scans_by_status: dict[str, int]
    drift: DriftOut | None


class ScanOut(BaseModel):
    id: int
    herd_id: int
    animal_tag: str
    created_utc: str
    status: Literal["trusted", "flagged", "refused"] = Field(
        description="trusted: use the weight; flagged: weight returned but outside the validated "
        "operating range (see reasons); refused: no weight, rescan"
    )
    reasons: list[str]
    weight_kg: float | None
    lower_kg: float | None
    upper_kg: float | None
    raw_weight_kg: float | None = Field(description="model output before the herd correction")
    calibration_id: int | None
    model_version: str
    scale_weight_kg: float | None
    weighed_utc: str | None


class ScanResultOut(ScanOut):
    preview: dict[str, Any] | None = Field(
        default=None,
        description="only when requested with preview=true: depth thumbnails with the animal mask per view, "
        "the body points (base64 int16 mm, x along the body, y lateral, z height; all three views when the "
        "station calibration is known, else the top view) with the view of each point, and the trunk "
        "cross-sections integrated into the volume; null for a refused scan",
    )


class WeighingIn(BaseModel):
    weight_kg: float = Field(gt=0, examples=[612.5])


def _scan(scan: Scan) -> ScanOut:
    return ScanOut(**asdict(scan))


def _calibration(c: Calibration) -> CalibrationOut:
    return CalibrationOut(
        id=c.id, created_utc=c.created_utc, slope=c.slope, intercept_kg=c.intercept_kg,
        n_reference=c.n_reference, interval_half_width_kg=c.half_width_kg,
    )


async def _read_cloud(upload: UploadFile, view: View) -> OrganizedCloud:
    raw = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{view.value} cloud is larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    try:
        return parse_organized_cloud(raw, f"{view.value} cloud")
    except PlyFormatError as exc:
        raise HTTPException(422, str(exc)) from exc


def _rig(text: str | None) -> Rig | None:
    if not text:
        return None
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError('expected an object {"left": ..., "right": ...}')
        return Rig.from_json(data)
    except KeyError as exc:
        raise HTTPException(422, f"rig: missing {exc}") from exc
    except (ValueError, TypeError) as exc:  # JSONDecodeError is a ValueError
        raise HTTPException(422, f"rig: {exc}") from exc


# ---- app ------------------------------------------------------------------------------
def create_app(
    service: WeightService,
    api_key: str | None = None,
    site_dir: Path | None = None,
    figures_dir: Path | None = None,
    demo: "Demo | None" = None,
) -> FastAPI:
    """Build the app around a service. With ``api_key`` set, every /v1 route
    requires the ``X-API-Key`` header. ``site_dir`` is served at ``/`` (the
    presentation website) and ``figures_dir`` at ``/figures`` (thesis figures,
    ``{uk,en}/NN_name.png``); both are optional and public."""
    app = FastAPI(
        title="cattle3d weight API",
        version=__version__,
        description="Live weight of cattle from left, right and top depth clouds, "
        "with per-herd calibration and drift checks.",
    )
    header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def authorised(key: Annotated[str | None, Security(header)]) -> None:
        if api_key and not (key and hmac.compare_digest(key, api_key)):
            raise HTTPException(401, "missing or wrong X-API-Key")

    for error, code in ((NotFoundError, 404), (DuplicateError, 409), (InvalidRequestError, 422)):
        def handler(_: Request, exc: Exception, code: int = code) -> JSONResponse:
            return JSONResponse({"detail": str(exc)}, status_code=code)

        app.add_exception_handler(error, handler)

    @app.get("/health", tags=["service"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    v1 = [Depends(authorised)]

    @app.get("/v1/model", tags=["service"], dependencies=v1)
    def model_card() -> dict[str, object]:
        """What the model was trained on, its validated range and interval."""
        card = service.model.card()
        card.pop("herd_corrections", None)  # farm corrections live in the database
        return card

    @app.post("/v1/herds", tags=["herds"], dependencies=v1, status_code=201)
    def create_herd(body: HerdIn) -> HerdOut:
        return HerdOut(**asdict(service.create_herd(body.name)))

    @app.get("/v1/herds", tags=["herds"], dependencies=v1)
    def list_herds() -> list[HerdOut]:
        return [HerdOut(**asdict(h)) for h in service.store.herds()]

    @app.get("/v1/herds/{herd_id}", tags=["herds"], dependencies=v1)
    def herd_status(herd_id: int) -> HerdStatusOut:
        """Calibration, reference weighings collected, scan counts and drift."""
        s = service.herd_status(herd_id)
        return HerdStatusOut(
            herd=HerdOut(**asdict(s.herd)),
            calibrated=s.calibration is not None,
            calibration=None if s.calibration is None else _calibration(s.calibration),
            n_reference_animals=s.n_reference_animals,
            recommended_reference_animals=s.recommended_reference_animals,
            scans_by_status=s.scans_by_status,
            drift=None if s.drift is None else DriftOut(**asdict(s.drift)),
        )

    @app.post("/v1/herds/{herd_id}/calibration", tags=["herds"], dependencies=v1, status_code=201)
    def calibrate(herd_id: int) -> CalibrationOut:
        """Fit the herd correction from the latest weighed scan of each animal.
        Scans made from now on use it."""
        return _calibration(service.calibrate(herd_id))

    @app.post("/v1/herds/{herd_id}/scans", tags=["scans"], dependencies=v1, status_code=201)
    async def scan(
        herd_id: int,
        animal_tag: Annotated[str, Form(min_length=1, max_length=100)],
        left: Annotated[UploadFile, File(description="left-view PLY, dense 512×424 frame")],
        right: Annotated[UploadFile, File(description="right-view PLY")],
        top: Annotated[UploadFile, File(description="top-view PLY")],
        scale_weight_kg: Annotated[float | None, Form(gt=0)] = None,
        preview: Annotated[bool, Form(description="also return what the cameras saw")] = False,
        rig: Annotated[str | None, Form(
            description='station calibration for the preview, JSON {"left": [[R row]×3, [t]], "right": ...} '
            "mapping each side camera into the top camera frame (metres); default: the herd's"
        )] = None,
    ) -> ScanResultOut:
        """Weigh one animal from its three clouds. Give ``scale_weight_kg`` when
        the animal was also put on the scale (reference or spot weighing)."""
        service.herd(herd_id)
        station = _rig(rig)
        clouds = {View.LEFT: await _read_cloud(left, View.LEFT),
                  View.RIGHT: await _read_cloud(right, View.RIGHT),
                  View.TOP: await _read_cloud(top, View.TOP)}
        # Segmentation is CPU-bound: keep it off the event loop.
        if preview:
            result, shown = await run_in_threadpool(
                service.scan_with_preview, herd_id, animal_tag, clouds, scale_weight_kg, station
            )
            return ScanResultOut(**asdict(result), preview=shown)
        result = await run_in_threadpool(service.scan, herd_id, animal_tag, clouds, scale_weight_kg, station)
        return ScanResultOut(**asdict(result))

    @app.get("/v1/herds/{herd_id}/scans", tags=["scans"], dependencies=v1)
    def herd_scans(herd_id: int, limit: int = 50) -> list[ScanOut]:
        """The herd's most recent scans, newest first."""
        return [_scan(s) for s in service.herd_scans(herd_id, max(1, min(limit, 500)))]

    @app.get("/v1/scans/{scan_id}", tags=["scans"], dependencies=v1)
    def get_scan(scan_id: int) -> ScanOut:
        return _scan(service.get_scan(scan_id))

    @app.put("/v1/scans/{scan_id}/weighing", tags=["scans"], dependencies=v1)
    def record_weighing(scan_id: int, body: WeighingIn) -> ScanOut:
        """Attach a scale reading to an earlier scan."""
        return _scan(service.record_weighing(scan_id, body.weight_kg))

    @app.get("/v1/herds/{herd_id}/animals/{animal_tag}/scans", tags=["scans"], dependencies=v1)
    def animal_history(herd_id: int, animal_tag: str) -> list[ScanOut]:
        return [_scan(s) for s in service.animal_history(herd_id, animal_tag)]

    if demo is not None:
        @app.get("/v1/demo", tags=["demo"], dependencies=v1)
        def demo_info() -> dict[str, object]:
            """Demo herds, sample animals the model did not train on, and the
            address a phone on the same network can open."""
            return demo.info()

        @app.get("/v1/demo/samples/{sample_id}/{view}.ply", tags=["demo"], dependencies=v1)
        def demo_cloud(sample_id: str, view: View) -> FileResponse:
            path = demo.cloud_path(sample_id, view)
            if path is None:
                raise HTTPException(404, f"no demo sample {sample_id!r}")
            return FileResponse(path, media_type="application/octet-stream", filename=f"{sample_id}_{view.value}.ply")

        @app.get("/v1/demo/qr.svg", tags=["demo"], dependencies=v1)
        def demo_qr() -> Response:
            svg = demo.qr_svg()
            if svg is None:
                raise HTTPException(404, "QR code needs the optional package qrcode")
            return Response(svg, media_type="image/svg+xml")

    # Mounted last so the API routes above take precedence over "/".
    if figures_dir is not None and Path(figures_dir).is_dir():
        app.mount("/figures", StaticFiles(directory=figures_dir), name="figures")
    if site_dir is not None:
        app.mount("/", StaticFiles(directory=site_dir, html=True), name="site")
    return app


def app_from_files(
    model_path: Path, db_path: Path, api_key: str | None = None, figures_dir: Path | None = None,
    site: bool = True,
) -> FastAPI:
    service = WeightService(WeightModel.load(model_path), Store(db_path))
    return create_app(service, api_key, SITE_DIR if site else None, figures_dir)
