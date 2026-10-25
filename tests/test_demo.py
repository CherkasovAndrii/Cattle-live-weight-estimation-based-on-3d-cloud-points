"""The farm web app's API additions and the defence demo, on synthetic data."""

import base64

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from cattle3d.api.app import create_app
from cattle3d.api.service import WeightService
from cattle3d.api.store import Store
from cattle3d.demo import N_SAMPLES, Demo, Sample, choose_holdout
from cattle3d.modeling.calibration import RECOMMENDED_CALIBRATION_ANIMALS
from cattle3d.types import View
from tests.test_api import AUTH, KEY, RowsByTag, _files, _herd, _ply, model, plys, real_client, side_scene_module, table  # noqa: F401
from tests.test_api import top_scene_module  # noqa: F401


# ---- scan preview and herd listing ----------------------------------------------------
def test_preview_shows_what_the_cameras_saw(real_client, plys):  # noqa: F811
    herd = _herd(real_client)
    r = real_client.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "UA-1", "preview": "true"},
                         files=_files(plys), headers=AUTH)
    assert r.status_code == 201, r.text
    preview = r.json()["preview"]
    assert set(preview["views"]) == {"left", "right", "top"}
    for view in preview["views"].values():
        assert view["image"].startswith("data:image/png;base64,") and view["animal_points"] > 0
    points = np.frombuffer(base64.b64decode(preview["top_points_mm"]), np.int16)
    assert points.size == 3 * preview["n_top_points"] > 0
    assert len(preview["slices"]) == 64
    assert preview["trunk_m"][0] < 0 < preview["trunk_m"][1]  # x is centred on the trunk
    assert {"trunk_volume_m3", "height_m"} <= set(preview["measures"])
    assert preview["timing_ms"]["total_ms"] >= preview["timing_ms"]["segmentation_and_features_ms"]


def test_preview_is_off_unless_asked(real_client, plys):  # noqa: F811
    herd = _herd(real_client)
    r = real_client.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "UA-1"}, files=_files(plys), headers=AUTH)
    assert r.json()["preview"] is None


def test_refused_scan_has_no_preview(real_client, plys):  # noqa: F811
    herd = _herd(real_client)
    from tests.conftest import plane_hit, render_depth

    empty = dict(plys, top=_ply(render_depth([plane_hit([0, 0, -1], 3.0)])))  # floor only, no animal
    r = real_client.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "UA-1", "preview": "true"},
                         files=_files(empty), headers=AUTH)
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "refused" and r.json()["preview"] is None


def test_herd_scans_are_listed_newest_first(model, tmp_path, table):  # noqa: F811
    service = WeightService(model, Store(tmp_path / "farm.sqlite3"), RowsByTag(table))
    c = TestClient(create_app(service, api_key=KEY))
    herd, other = _herd(c), _herd(c, "South barn")
    features = table.drop(columns=["weight_kg", "herd"]).to_dict("records")
    for i in range(3):
        service.scan_from_features(herd, f"UA-{i}", features[i], float(table["weight_kg"][i]))
    service.scan_from_features(other, "UA-X", features[3])
    listed = c.get(f"/v1/herds/{herd}/scans", headers=AUTH).json()
    assert [s["animal_tag"] for s in listed] == ["UA-2", "UA-1", "UA-0"]
    assert listed[0]["scale_weight_kg"] == pytest.approx(table["weight_kg"][2])
    assert len(c.get(f"/v1/herds/{herd}/scans?limit=1", headers=AUTH).json()) == 1
    assert c.get("/v1/herds/99/scans", headers=AUTH).status_code == 404


def test_imported_features_follow_the_same_rules_as_clouds(model, tmp_path, table):  # noqa: F811
    service = WeightService(model, Store(tmp_path / "farm.sqlite3"), RowsByTag(table))
    herd = service.create_herd("North barn").id
    row = table.drop(columns=["weight_kg", "herd"]).iloc[0].to_dict()
    scan = service.scan_from_features(herd, "UA-1", row, 610.0)
    assert scan.status == "flagged"  # not calibrated yet
    assert scan.raw_weight_kg == pytest.approx(model.raw_predict(pd.DataFrame([row]))[0])
    assert scan.scale_weight_kg == 610.0


# ---- demo -----------------------------------------------------------------------------
def test_holdout_spans_the_weight_range_and_is_disjoint():
    rng = np.random.default_rng(1)
    t = pd.DataFrame({"animal_id": [str(i) for i in range(1, 96)], "weight_kg": rng.uniform(550, 750, 95),
                      "herd": "cowdb2", "error": pd.Series([np.nan] * 95, dtype=object)})
    t.loc[5, "error"] = "SegmentationError: no animal"
    demo, reference = choose_holdout(t, "cowdb2")
    assert len(demo) == N_SAMPLES and len(reference) == RECOMMENDED_CALIBRATION_ANIMALS
    assert not set(demo) & set(reference) and "6" not in demo + reference
    w = t.set_index("animal_id")["weight_kg"]
    ok = t[t["error"].isna()]["weight_kg"]
    assert w[demo].min() < ok.quantile(0.15) and w[demo].max() > ok.quantile(0.85)
    assert choose_holdout(t, "cowdb2") == (demo, reference)  # same animals on every run


@pytest.fixture
def demo_client(model, tmp_path, table):  # noqa: F811
    from cattle3d.web import SITE_DIR

    paths = {}
    for view in View:
        paths[view] = tmp_path / f"{view.value}.ply"
        paths[view].write_bytes(b"ply " + view.value.encode())
    service = WeightService(model, Store(tmp_path / "farm.sqlite3"), RowsByTag(table))
    herd = service.create_herd("Демо-ферма «Абердин-ангуси»").id
    demo = Demo([Sample("cowdb2-7", herd, "cowdb2", "7", 640.0, paths)], {"cowdb2": herd},
                "http://192.168.1.5:8000/app/#key=k")
    return TestClient(create_app(service, api_key=KEY, site_dir=SITE_DIR, demo=demo))


def test_demo_lists_farms_and_samples_and_serves_their_clouds(demo_client):
    assert demo_client.get("/v1/demo").status_code == 401
    info = demo_client.get("/v1/demo", headers=AUTH).json()
    assert info["herds"][0]["breed"] == "абердин-ангус"
    assert info["samples"][0] == {"id": "cowdb2-7", "herd_id": info["herds"][0]["id"], "herd": "cowdb2",
                                  "animal_id": "7", "breed": "абердин-ангус", "scale_weight_kg": 640.0}
    assert demo_client.get("/v1/demo/samples/cowdb2-7/top.ply", headers=AUTH).content == b"ply top"
    assert demo_client.get("/v1/demo/samples/nope/top.ply", headers=AUTH).status_code == 404
    assert demo_client.get("/v1/demo/samples/cowdb2-7/front.ply", headers=AUTH).status_code == 422


def test_demo_qr_code_points_phones_at_the_app(demo_client):
    pytest.importorskip("qrcode")
    r = demo_client.get("/v1/demo/qr.svg", headers=AUTH)
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/svg+xml") and "<svg" in r.text


def test_farm_app_is_served_and_demo_routes_need_a_demo(demo_client, model, tmp_path, table):  # noqa: F811
    page = demo_client.get("/app/")
    assert page.status_code == 200 and 'lang="uk"' in page.text and "viewport" in page.text
    assert demo_client.get("/app/app.js").status_code == 200
    plain = TestClient(create_app(WeightService(model, Store(tmp_path / "f2.sqlite3"), RowsByTag(table)), api_key=KEY))
    assert plain.get("/v1/demo", headers=AUTH).status_code == 404
