"""The REST API end to end, on synthetic clouds and a small synthetic model."""

import io
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from cattle3d.api.app import create_app
from cattle3d.api.service import UNCALIBRATED, WeightService
from cattle3d.api.store import Store
from cattle3d.modeling.models import PHYSICS_FEATURE, VOLUME_FEATURES
from cattle3d.modeling.weight_model import WeightModel
from cattle3d.pipeline import features_from_clouds
from cattle3d.types import OrganizedCloud, View
from tests.conftest import write_pcl_ply

KEY = "test-key"
AUTH = {"X-API-Key": KEY}


def _table(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    volume = rng.uniform(0.6, 1.1, n)
    data = {name: volume * rng.uniform(0.9, 1.1) + rng.normal(0, 0.01, n) for name in VOLUME_FEATURES}
    data[PHYSICS_FEATURE] = volume
    data.update({"left_n_points": 20_000, "right_n_points": 20_000, "top_n_points": 30_000})
    df = pd.DataFrame(data)
    df["weight_kg"] = 600 * volume + rng.normal(0, 10, n)
    df["herd"] = "synthetic"
    return df


@pytest.fixture(scope="module")
def table():
    return _table()


@pytest.fixture(scope="module")
def model(table):
    return WeightModel.fit(table)


class RowsByTag:
    """Stand-in feature extractor: the 'clouds' are ignored and the next
    synthetic animal's features are returned, so the tests control the truth."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.rows = table.drop(columns=["weight_kg", "herd"]).to_dict("records")
        self.i = 0

    def __call__(self, clouds):
        row = self.rows[self.i % len(self.rows)]
        self.i += 1
        return row


def _ply(scene: np.ndarray) -> bytes:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v.ply"
        write_pcl_ply(path, scene.reshape(-1, 3))
        return path.read_bytes()


@pytest.fixture(scope="module")
def plys(side_scene_module, top_scene_module):
    side, top = _ply(side_scene_module), _ply(top_scene_module)
    return {"left": side, "right": side, "top": top}


@pytest.fixture(scope="module")
def side_scene_module():
    from tests.conftest import box_hit, plane_hit, render_depth

    return render_depth([plane_hit([0, -1, 0], 1.2), box_hit([-0.8, -0.1, 2.0], [0.8, 0.7, 2.5]),
                         plane_hit([0, 0, -1], 6.0)])


@pytest.fixture(scope="module")
def top_scene_module():
    from tests.conftest import box_hit, plane_hit, render_depth

    return render_depth([plane_hit([0, 0, -1], 3.0), box_hit([-0.75, -0.25, 1.8], [0.75, 0.25, 3.0])])


def _files(plys: dict[str, bytes]) -> dict[str, tuple[str, io.BytesIO, str]]:
    return {view: (f"{view}.ply", io.BytesIO(raw), "application/octet-stream") for view, raw in plys.items()}


@pytest.fixture
def client(model, tmp_path, table):
    service = WeightService(model, Store(tmp_path / "farm.sqlite3"), extract=RowsByTag(table))
    return TestClient(create_app(service, api_key=KEY))


@pytest.fixture
def real_client(model, tmp_path):
    service = WeightService(model, Store(tmp_path / "farm.sqlite3"), extract=features_from_clouds)
    return TestClient(create_app(service, api_key=KEY))


def _herd(client, name="North barn") -> int:
    r = client.post("/v1/herds", json={"name": name}, headers=AUTH)
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---- service and auth -----------------------------------------------------------------
def test_health_is_open_and_v1_needs_the_key(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/herds").status_code == 401
    assert client.get("/v1/herds", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/herds", headers=AUTH).json() == []


def test_model_card_describes_the_training_data(client):
    card = client.get("/v1/model", headers=AUTH).json()
    assert card["n_training_animals"] == 60 and "herd_corrections" not in card


def test_duplicate_herd_and_unknown_herd(client):
    _herd(client)
    assert client.post("/v1/herds", json={"name": "North barn"}, headers=AUTH).status_code == 409
    assert client.get("/v1/herds/99", headers=AUTH).status_code == 404


# ---- scans ----------------------------------------------------------------------------
def test_real_clouds_go_through_the_whole_pipeline(real_client, plys):
    herd = _herd(real_client)
    r = real_client.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "UA-001"}, files=_files(plys), headers=AUTH)
    assert r.status_code == 201, r.text
    scan = r.json()
    assert scan["status"] == "flagged"  # a box is no cow, and the herd is not calibrated
    assert UNCALIBRATED in scan["reasons"]
    assert scan["lower_kg"] < scan["weight_kg"] < scan["upper_kg"]


def test_a_view_without_an_animal_is_refused_and_stored(real_client, plys):
    from tests.conftest import plane_hit, render_depth

    herd = _herd(real_client)
    empty = _ply(render_depth([plane_hit([0, 0, -1], 3.0)]))
    r = real_client.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "UA-002"},
                         files=_files({**plys, "top": empty}), headers=AUTH)
    assert r.status_code == 201
    assert r.json()["status"] == "refused" and r.json()["weight_kg"] is None
    assert r.json()["reasons"][0].startswith("rescan")
    assert real_client.get(f"/v1/herds/{herd}", headers=AUTH).json()["scans_by_status"] == {"refused": 1}


def test_a_file_that_is_not_a_cloud_is_a_client_error(client, plys):
    herd = _herd(client)
    bad = {**plys, "left": b"not a ply"}
    r = client.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "x"}, files=_files(bad), headers=AUTH)
    assert r.status_code == 422 and "left cloud" in r.json()["detail"]


def test_implausible_scale_weight_is_rejected(client, plys):
    herd = _herd(client)
    r = client.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "x", "scale_weight_kg": "5000"},
                    files=_files(plys), headers=AUTH)
    assert r.status_code == 422


# ---- calibration and drift ------------------------------------------------------------------
def _scan(client, herd, plys, tag, weight=None) -> dict:
    data = {"animal_tag": tag} | ({} if weight is None else {"scale_weight_kg": str(weight)})
    r = client.post(f"/v1/herds/{herd}/scans", data=data, files=_files(plys), headers=AUTH)
    assert r.status_code == 201, r.text
    return r.json()


def test_calibration_removes_a_herd_shift_and_drift_is_detected(client, plys, table):
    herd = _herd(client)
    truth = table["weight_kg"].to_numpy()
    shift = 80.0  # this farm's animals are heavier than the model expects

    assert client.post(f"/v1/herds/{herd}/calibration", headers=AUTH).status_code == 422  # nothing weighed
    reference = [_scan(client, herd, plys, f"A{i}", truth[i] + shift) for i in range(20)]
    before = np.mean([abs(s["weight_kg"] - s["scale_weight_kg"]) for s in reference])
    assert before > 50

    cal = client.post(f"/v1/herds/{herd}/calibration", headers=AUTH)
    assert cal.status_code == 201 and cal.json()["n_reference"] == 20

    after = [_scan(client, herd, plys, f"B{i}") for i in range(20, 40)]
    errors = [abs(s["weight_kg"] - (truth[20 + k] + shift)) for k, s in enumerate(after)]
    assert np.mean(errors) < 25
    assert all(s["status"] == "trusted" and s["calibration_id"] == cal.json()["id"] for s in after)

    # spot weighings that agree: no drift
    for k, s in enumerate(after[:8]):
        client.put(f"/v1/scans/{s['id']}/weighing", json={"weight_kg": truth[20 + k] + shift}, headers=AUTH)
    status = client.get(f"/v1/herds/{herd}", headers=AUTH).json()
    assert status["calibrated"] and not status["drift"]["drifted"]
    assert status["n_reference_animals"] == 28

    # the herd gains 40 kg (season, feed): spot weighings now disagree
    for k, s in enumerate(after[8:]):
        client.put(f"/v1/scans/{s['id']}/weighing", json={"weight_kg": truth[28 + k] + shift + 40}, headers=AUTH)
    drift = client.get(f"/v1/herds/{herd}", headers=AUTH).json()["drift"]
    assert drift["drifted"] and drift["mean_residual_kg"] > 10


def test_animal_history_lists_scans_in_order(client, plys):
    herd = _herd(client)
    first, second = _scan(client, herd, plys, "UA-7"), _scan(client, herd, plys, "UA-7")
    _scan(client, herd, plys, "UA-8")
    history = client.get(f"/v1/herds/{herd}/animals/UA-7/scans", headers=AUTH).json()
    assert [s["id"] for s in history] == [first["id"], second["id"]]


def test_store_survives_a_restart(model, tmp_path, table, plys):
    path = tmp_path / "farm.sqlite3"
    c1 = TestClient(create_app(WeightService(model, Store(path), RowsByTag(table)), api_key=KEY))
    herd = _herd(c1)
    scan = _scan(c1, herd, plys, "UA-9", 600)
    c2 = TestClient(create_app(WeightService(model, Store(path), RowsByTag(table)), api_key=KEY))
    assert c2.get(f"/v1/scans/{scan['id']}", headers=AUTH).json()["scale_weight_kg"] == 600


def test_cloud_fixture_is_segmentable(side_scene_module, top_scene_module):
    clouds = {View.LEFT: OrganizedCloud(side_scene_module), View.RIGHT: OrganizedCloud(side_scene_module),
              View.TOP: OrganizedCloud(top_scene_module)}
    assert PHYSICS_FEATURE in features_from_clouds(clouds)


def test_web_console_is_served_at_the_root_without_hiding_the_api(model, tmp_path, table):
    from cattle3d.web import SITE_DIR

    figures = tmp_path / "figures" / "uk"
    figures.mkdir(parents=True)
    (figures / "09_within_herd_oof.png").write_bytes(b"\x89PNG")
    service = WeightService(model, Store(tmp_path / "farm.sqlite3"), RowsByTag(table))
    c = TestClient(create_app(service, api_key=KEY, site_dir=SITE_DIR, figures_dir=tmp_path / "figures"))
    page = c.get("/")
    assert page.status_code == 200 and 'lang="uk"' in page.text and '<div id="root">' in page.text
    for script in re.findall(r'src="(/assets/[^"]+\.js)"', page.text):  # the built React bundle
        assert c.get(script).status_code == 200
    # Browser-router pages get the app; API paths, missing assets and figures stay 404.
    for page_path in ("/herd", "/animals/UA-0427", "/weigh"):
        assert c.get(page_path).text == page.text
    assert c.get("/v1/nope", headers=AUTH).status_code == 404
    assert c.get("/assets/missing.js").status_code == 404
    assert c.get("/figures/uk/09_within_herd_oof.png").content == b"\x89PNG"
    assert c.get("/v1/herds").status_code == 401  # the console does not open the API
    points = c.get("/img/cow70_points.bin").content  # merged cloud: int16 x, y, z mm + uint8 view + pad
    assert len(points) % 8 == 0 and len(points) // 8 > 10_000
    assert set(points[6::8]) == {0, 1, 2}
    assert c.get("/figures/uk/09_within_herd_oof.png").content == b"\x89PNG"
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/v1/herds").status_code == 401  # the site does not open the API
