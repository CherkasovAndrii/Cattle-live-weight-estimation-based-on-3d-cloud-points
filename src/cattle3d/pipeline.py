"""From point-cloud files to an animal's feature row.

Owns: the composition read → segment → features for one animal, and the
batch extraction for a herd. This is the single place the API, the CLI and
the experiments go through, so training and inference compute features
identically.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from cattle3d.features.geometry_features import animal_features, view_features
from cattle3d.geometry.segment import SegmentationConfig, segment_view
from cattle3d.io.datasets import AnimalRecord
from cattle3d.io.ply import read_organized_cloud
from cattle3d.types import ALL_VIEWS, OrganizedCloud, View


def features_from_clouds(
    clouds: dict[View, OrganizedCloud], config: SegmentationConfig = SegmentationConfig()
) -> dict[str, float]:
    missing = [v.value for v in ALL_VIEWS if v not in clouds]
    if missing:
        raise ValueError(f"all three views are required; missing {missing}")
    per_view = {view: view_features(segment_view(clouds[view], view, config)) for view in ALL_VIEWS}
    return animal_features(per_view)


def features_from_files(paths: dict[View, Path]) -> dict[str, float]:
    return features_from_clouds({view: read_organized_cloud(path) for view, path in paths.items()})


def _extract_one(record: AnimalRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "herd": record.herd,
        "animal_id": record.animal_id,
        "weight_kg": record.weight_kg,
        **{f"manual_{k}": v for k, v in record.manual.items()},
    }
    try:
        row.update(features_from_files(record.view_paths))
        row["error"] = ""
    except Exception as exc:  # recorded per animal and reported, never silently dropped
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def extract_herd(records: tuple[AnimalRecord, ...], workers: int = 4) -> pd.DataFrame:
    """Feature table, one row per animal; failed animals keep an ``error`` text."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_extract_one, records, chunksize=2))
    return pd.DataFrame(rows)
