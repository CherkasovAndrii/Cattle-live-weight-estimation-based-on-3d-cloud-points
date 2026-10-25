import openpyxl
import pytest

from cattle3d.io.datasets import DatasetLayoutError, load_cowdatabase, load_cowdatabase2
from cattle3d.types import View


def _touch_views(paths):
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")


def test_cowdatabase2_joins_by_frame_and_skips_non_integer_frames(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["№ ", "Frame number ", "Pluck number ", None, None, None, "Tag number", None, "Chip number",
               "Birthday", "Live weigth", "Withers height", "Hip height ", "Chest width", "Chest heart"])
    ws.append([None] * 15)
    ws.append([1, 3, 1, None, None, None, 1, None, 1, "01.01.2020", 633, 123, 122, 53, 214])
    ws.append([2, 42.52, 2, None, None, None, 2, None, 2, "01.01.2020", 600, 140, 141, 50, 210])
    ws.append([3, 7, 3, None, None, None, 3, None, 3, "01.01.2020", 617, 145, 144, 51, 216])
    wb.save(tmp_path / "Database2022v2.xlsx")
    _touch_views(tmp_path / "3" / "raw_data" / v / "cloud.ply" for v in ("left", "right", "top"))
    _touch_views(tmp_path / "7" / "raw_data" / v / "cloud.ply" for v in ("left", "top"))  # right missing

    herd = load_cowdatabase2(tmp_path)
    assert [a.animal_id for a in herd.animals] == ["3"]
    assert herd.animals[0].weight_kg == 633
    assert herd.animals[0].manual["heart_girth"] == 214
    assert {s.animal_id for s in herd.skipped} == {"42.52", "7"}


def test_cowdatabase_published_vs_audited_join(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["N", "live weithg", "withers height", "hip height", "chest depth", "chest width",
               "ilium width", "hip joint width", "oblique body length", "hip length", "heart girth"])
    for n in range(1, 25):
        ws.append([n, 400 + n, 115, 120, 60, 40, 42, 42, 130, 40, 170])
    wb.save(tmp_path / "Measurements.xlsx")
    for n in range(1, 25):
        _touch_views(tmp_path / f"{v}Clouds" / f"{n}.ply" for v in ("Left", "Right", "Top"))

    published = load_cowdatabase(tmp_path, join="published")
    assert len(published.animals) == 24
    by_row = {a.animal_id: a for a in published.animals}
    assert by_row["23"].view_paths[View.TOP].name == "23.ply"

    audited = {a.animal_id: a for a in load_cowdatabase(tmp_path, join="audited").animals}
    assert "22" not in audited  # row 22 has no capture in the audited join
    assert audited["23"].view_paths[View.TOP].name == "22.ply"


def test_wrong_folder_fails_loudly(tmp_path):
    with pytest.raises(DatasetLayoutError):
        load_cowdatabase2(tmp_path)
