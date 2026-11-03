"""From point-cloud files to an animal's feature row.

Owns: the composition read → segment → features for one animal, the batch
extraction for a herd, and reading a saved feature table back. This is the
single place the API, the CLI and the experiments go through, so training and
inference compute features identically.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TypeVar

import pandas as pd

from cattle3d.features.geometry_features import animal_features, view_features
from cattle3d.geometry.segment import SegmentationConfig, SegmentationTrace, trace_view
from cattle3d.io.ply import read_organized_cloud
from cattle3d.types import ALL_VIEWS, AnimalRecord, OrganizedCloud, View

T = TypeVar("T")
R = TypeVar("R")


def analyse_clouds(
    clouds: dict[View, OrganizedCloud], config: SegmentationConfig = SegmentationConfig()
) -> tuple[dict[str, float], dict[View, SegmentationTrace]]:
    """Features of one animal, plus every view's segmentation masks for
    interfaces that show the operator what the cameras saw."""
    missing = [v.value for v in ALL_VIEWS if v not in clouds]
    if missing:
        raise ValueError(f"all three views are required; missing {missing}")
    traces = {view: trace_view(clouds[view], view, config) for view in ALL_VIEWS}
    features = animal_features({view: view_features(t.result) for view, t in traces.items()})
    return features, traces


def features_from_clouds(
    clouds: dict[View, OrganizedCloud], config: SegmentationConfig = SegmentationConfig()
) -> dict[str, float]:
    return analyse_clouds(clouds, config)[0]


def read_views(paths: dict[View, Path]) -> dict[View, OrganizedCloud]:
    return {view: read_organized_cloud(path) for view, path in paths.items()}


def features_from_files(paths: dict[View, Path]) -> dict[str, float]:
    return features_from_clouds(read_views(paths))


def feature_row(
    record: AnimalRecord, compute: Callable[[AnimalRecord], dict[str, float]], manual: bool = True
) -> dict[str, object]:
    """One row of a feature table: the animal's label (and manual measurements),
    its features and an ``error`` text, empty when the views were usable."""
    row: dict[str, object] = {"herd": record.herd, "animal_id": record.animal_id, "weight_kg": record.weight_kg}
    if manual:
        row.update({f"manual_{k}": v for k, v in record.manual.items()})
    try:
        row.update(compute(record))
        row["error"] = ""
    except Exception as exc:  # recorded per animal and reported, never silently dropped
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def map_in_processes(function: Callable[[T], R], items: Iterable[T], workers: int = 4) -> list[R]:
    """``map`` over worker processes; segmentation is CPU-bound."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items, chunksize=2))


def _extract_one(record: AnimalRecord) -> dict[str, object]:
    return feature_row(record, lambda r: features_from_files(r.view_paths))


def extract_herd(records: tuple[AnimalRecord, ...], workers: int = 4) -> pd.DataFrame:
    """Feature table, one row per animal; failed animals keep an ``error`` text."""
    return pd.DataFrame(map_in_processes(_extract_one, records, workers))


def read_feature_table(path: Path) -> pd.DataFrame:
    """A feature table written by ``extract_herd``: ids stay text and an empty
    ``error`` reads as missing (NaN), a filled one as its text."""
    return pd.read_csv(path, dtype={"animal_id": str}, keep_default_na=False, na_values=[""])
