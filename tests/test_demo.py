"""The farm web app's API additions and the defence demo, on synthetic data."""

import base64
import json

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
    points = np.frombuffer(base64.b64decode(preview["points_mm"]), np.int16)
    views = np.frombuffer(base64.b64decode(preview["point_view"]), np.uint8)
    assert points.size == 3 * preview["n_points"] == 3 * views.size > 0
    assert not preview["views_merged"] and set(views) == {preview["view_order"].index("top")}
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
                "http://192.168.1.5:8000/weigh#key=k")
    return TestClient(create_app(service, api_key=KEY, site_dir=SITE_DIR, demo=demo))


def test_demo_lists_farms_and_samples_and_serves_their_clouds(demo_client):
    assert demo_client.get("/v1/demo").status_code == 401
    info = demo_client.get("/v1/demo", headers=AUTH).json()
    assert info["herds"][0]["breed"] == "абердин-ангус"
    assert info["samples"][0] == {"id": "cowdb2-7", "herd_id": info["herds"][0]["id"], "herd": "cowdb2",
                                  "animal_id": "7", "breed": "абердин-ангус", "scale_weight_kg": 640.0, "rig": None}
    assert demo_client.get("/v1/demo/samples/cowdb2-7/top.ply", headers=AUTH).content == b"ply top"
    assert demo_client.get("/v1/demo/samples/nope/top.ply", headers=AUTH).status_code == 404
    assert demo_client.get("/v1/demo/samples/cowdb2-7/front.ply", headers=AUTH).status_code == 422


def test_demo_qr_code_points_phones_at_the_app(demo_client):
    pytest.importorskip("qrcode")
    r = demo_client.get("/v1/demo/qr.svg", headers=AUTH)
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/svg+xml") and "<svg" in r.text


def test_console_opens_from_the_phone_link_and_demo_routes_need_a_demo(demo_client, model, tmp_path, table):  # noqa: F811
    for path in ("/weigh", "/app/"):  # old printed links still reach the console
        page = demo_client.get(path)
        assert page.status_code == 200 and 'lang="uk"' in page.text and "viewport" in page.text
    plain = TestClient(create_app(WeightService(model, Store(tmp_path / "f2.sqlite3"), RowsByTag(table)), api_key=KEY))
    assert plain.get("/v1/demo", headers=AUTH).status_code == 404


# ---- station calibration and the merged animal ----------------------------------------
def _rotation(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def test_rig_is_recovered_from_an_aligned_capture_and_survives_json():
    from cattle3d.geometry.rig import Rig, median_rig, rig_from_aligned

    rng = np.random.default_rng(0)
    raw = {v: rng.normal(size=(500, 3)) for v in View}
    truth = {View.LEFT: (_rotation(80), np.array([0.0, -2.7, 1.9])), View.RIGHT: (_rotation(-95), np.array([0.1, 2.3, 2.2]))}
    # blocks in the order top, left, right, all in the top camera frame
    aligned = np.vstack([raw[View.TOP]] + [raw[v] @ r.T + t for v, (r, t) in truth.items()])
    rig = median_rig([rig_from_aligned(raw, aligned), rig_from_aligned(raw, aligned)])
    for v, (r, t) in truth.items():
        assert np.allclose(rig.to_top[v][0], r, atol=1e-9) and np.allclose(rig.to_top[v][1], t, atol=1e-9)
    again = Rig.from_json(rig.to_json())
    assert np.allclose(again.to_top[View.LEFT][1], truth[View.LEFT][1], atol=1e-6)
    with pytest.raises(ValueError):
        Rig.from_json({"left": [[2, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], "right": rig.to_json()["right"]})


def test_matrices_file_is_read(tmp_path):
    from cattle3d.geometry.rig import read_matrices

    path = tmp_path / "1.matrices"
    path.write_text("Left -> Top\n1.0 0.0 0.0 0.49\n0.0 0.12 0.99 -2.45\n0.0 -0.99 0.12 1.84\n0.0 0.0 0.0 1.0\n\n"
                    "Right -> Top\n-1.0 -0.01 0.09 0.02\n-0.09 0.0 -1.0 2.22\n0.02 -1.0 0.0 2.17\n0.0 0.0 0.0 1.0\n")
    rig = read_matrices(path)
    r, t = rig.to_top[View.LEFT]
    assert np.allclose(r @ r.T, np.eye(3)) and np.allclose(t, [0.49, -2.45, 1.84])


def test_three_views_merge_into_one_animal(side_scene_module, top_scene_module):  # noqa: F811
    """The same box seen by a top camera and by side cameras placed with a
    known rig must land on the top view's points once merged."""
    from cattle3d.geometry.rig import Rig, merge_views
    from cattle3d.pipeline import analyse_clouds
    from cattle3d.types import OrganizedCloud

    clouds = {View.LEFT: OrganizedCloud(side_scene_module), View.RIGHT: OrganizedCloud(side_scene_module),
              View.TOP: OrganizedCloud(top_scene_module)}
    _, traces = analyse_clouds(clouds)
    identity = Rig({View.LEFT: (np.eye(3), np.zeros(3)), View.RIGHT: (np.eye(3), np.zeros(3))})
    merged = merge_views(clouds, traces, identity, refine=False)
    pts, ids = merged.points, merged.view_index
    n = [int((ids == i).sum()) for i in range(3)]
    assert n == [int(traces[v].animal.sum()) for v in (View.LEFT, View.RIGHT, View.TOP)]
    top = pts[ids == 2]
    assert np.allclose(top, traces[View.TOP].result.body_points)  # same frame as the volume slices


def test_preview_merges_views_with_the_herds_station_calibration(model, tmp_path, plys):  # noqa: F811
    from cattle3d.geometry.rig import Rig
    from cattle3d.pipeline import features_from_clouds

    service = WeightService(model, Store(tmp_path / "farm.sqlite3"), extract=features_from_clouds)
    c = TestClient(create_app(service, api_key=KEY))
    herd = _herd(c)
    identity = Rig({View.LEFT: (np.eye(3), np.zeros(3)), View.RIGHT: (np.eye(3), np.zeros(3))})
    service.station_rigs[herd] = identity
    r = c.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "UA-1", "preview": "true"}, files=_files(plys), headers=AUTH)
    preview = r.json()["preview"]
    assert preview["views_merged"]
    assert set(np.frombuffer(base64.b64decode(preview["point_view"]), np.uint8)) == {0, 1, 2}
    bad = c.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "UA-2", "preview": "true", "rig": "{not json"},
                 files=_files(plys), headers=AUTH)
    assert bad.status_code == 422 and "rig" in bad.text
    explicit = c.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "UA-3", "preview": "true",
                                                       "rig": json.dumps(identity.to_json())},
                      files=_files(plys), headers=AUTH)
    assert explicit.status_code == 201 and explicit.json()["preview"]["views_merged"]


# ---- the three views must show one animal ----------------------------------------------
def _merged(share: float, gap: float, shift: float):
    from cattle3d.geometry.rig import MergedViews

    return MergedViews(np.zeros((3, 3)), np.array([0, 1, 2], np.uint8), shift, share, gap)


def test_view_agreement_thresholds():
    assert _merged(0.95, 0.03, 0.07).same_animal
    assert not _merged(0.60, 0.03, 0.07).same_animal  # side body outside the top footprint
    assert not _merged(0.95, 0.20, 0.07).same_animal  # another back line
    assert not _merged(0.95, 0.03, 0.30).same_animal  # cameras would have to move 30 cm
    assert not _merged(0.60, 0.03, 0.07).contradictory and _merged(0.10, 0.03, 0.07).contradictory


def _station_client(model, tmp_path, merged, monkeypatch):  # noqa: F811
    from cattle3d.geometry.rig import Rig
    from cattle3d.pipeline import features_from_clouds

    service = WeightService(model, Store(tmp_path / "farm.sqlite3"), extract=features_from_clouds)
    c = TestClient(create_app(service, api_key=KEY))
    herd = _herd(c)
    service.station_rigs[herd] = Rig({View.LEFT: (np.eye(3), np.zeros(3)), View.RIGHT: (np.eye(3), np.zeros(3))})
    monkeypatch.setattr("cattle3d.api.service.merge_views", lambda clouds, traces, rig: merged)
    return c, herd


@pytest.mark.parametrize("preview", ["true", "false"])
def test_views_of_different_animals_are_flagged(model, tmp_path, plys, monkeypatch, preview):  # noqa: F811
    from cattle3d.api.service import MIXED_VIEWS

    c, herd = _station_client(model, tmp_path, _merged(0.6, 0.05, 0.1), monkeypatch)
    r = c.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "UA-1", "preview": preview}, files=_files(plys), headers=AUTH)
    assert r.status_code == 201 and r.json()["status"] == "flagged" and MIXED_VIEWS in r.json()["reasons"]
    assert r.json()["weight_kg"] is not None


def test_recording_quality_is_reported_and_checked(model, tmp_path, plys, monkeypatch):  # noqa: F811
    from cattle3d.geometry.quality import ViewQuality
    from cattle3d.pipeline import features_from_clouds

    service = WeightService(model, Store(tmp_path / "farm.sqlite3"), extract=features_from_clouds)
    c = TestClient(create_app(service, api_key=KEY))
    herd = _herd(c)
    r = c.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "UA-1", "preview": "true"}, files=_files(plys), headers=AUTH)
    quality = r.json()["preview"]["recording_quality"]
    assert set(quality) == {"left", "right", "top"}
    assert not any("depth noise" in x or "missing pixels" in x for x in r.json()["reasons"])

    worse = {View.LEFT: ViewQuality(View.LEFT, 0.012, 0.3), View.TOP: ViewQuality(View.TOP, 0.001, 1.2)}
    monkeypatch.setattr("cattle3d.api.service.view_quality",
                        lambda cloud, view, floor: worse.get(view, ViewQuality(view, 0.002, 0.2)))
    r = c.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "UA-2"}, files=_files(plys), headers=AUTH)
    reasons = r.json()["reasons"]
    assert r.json()["status"] == "flagged" and r.json()["weight_kg"] is not None
    assert any(x.startswith("depth noise in left view 1.2 %") for x in reasons)
    assert any(x.startswith("many scattered missing pixels in top view") for x in reasons)


def test_contradictory_views_are_refused(model, tmp_path, plys, monkeypatch):  # noqa: F811
    from cattle3d.api.service import CONTRADICTORY_VIEWS

    c, herd = _station_client(model, tmp_path, _merged(0.0, 0.6, 0.0), monkeypatch)
    r = c.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "UA-1", "preview": "true"}, files=_files(plys), headers=AUTH)
    body = r.json()
    assert body["status"] == "refused" and body["weight_kg"] is None and body["preview"] is None
    assert body["reasons"] == [f"rescan: {CONTRADICTORY_VIEWS}"]


def test_rig_field_must_be_an_object(real_client, plys):  # noqa: F811
    herd = _herd(real_client)
    for bad in ("[1, 2]", "{}", '{"left": 1}'):
        r = real_client.post(f"/v1/herds/{herd}/scans", data={"animal_tag": "UA-1", "rig": bad}, files=_files(plys), headers=AUTH)
        assert r.status_code == 422 and r.json()["detail"].startswith("rig:")
