"""A self-contained farm demo on the public herds, for presenting the system.

Owns: choosing demo animals that the demo model never trained on, training
and caching that model, seeding one calibrated demo farm per herd, and the
address and QR code a phone on the same network can open. Must not know about
HTTP routes (the API exposes ``Demo`` read-only).

Honesty rule: the animals offered for live weighing and the 20 reference
animals of each farm are held out from the demo model, so a demo weighing is a
genuine out-of-sample prediction, like on a new farm after calibration.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cattle3d.api.service import WeightService
from cattle3d.api.store import Store
from cattle3d.io.datasets import HERD_A, HERD_B, load_cowdatabase, load_cowdatabase2
from cattle3d.modeling.calibration import RECOMMENDED_CALIBRATION_ANIMALS
from cattle3d.modeling.weight_model import WeightModel
from cattle3d.types import AnimalRecord, View

N_SAMPLES = 6
HERDS = {
    HERD_A: {"farm": "Демо-ферма «Абердин-ангуси»", "breed": "абердин-ангус", "dataset": "CowDatabase2"},
    HERD_B: {"farm": "Демо-ферма «Герефорди»", "breed": "герефорд", "dataset": "CowDatabase"},
}


@dataclass(frozen=True)
class Sample:
    id: str
    herd_id: int
    herd: str
    animal_id: str
    scale_weight_kg: float
    paths: dict[View, Path]


def choose_holdout(table: pd.DataFrame, herd: str, seed: int = 0) -> tuple[list[str], list[str]]:
    """Demo animals spread over the herd's weight range, and reference animals
    drawn at random from the rest. Both are excluded from the demo model."""
    rows = table[(table["herd"] == herd) & table["error"].isna()].sort_values("weight_kg")
    ids = rows["animal_id"].astype(str).tolist()
    picks = np.round(np.linspace(0.08, 0.92, N_SAMPLES) * (len(ids) - 1)).astype(int)
    demo = [ids[i] for i in sorted(set(picks))]
    rest = [i for i in ids if i not in demo]
    rng = np.random.default_rng(seed)
    reference = sorted(rng.choice(rest, RECOMMENDED_CALIBRATION_ANIMALS, replace=False).tolist(), key=int)
    return demo, reference


def demo_model(table: pd.DataFrame, holdout: dict[str, list[str]], path: Path) -> WeightModel:
    """Load the cached demo model if it was trained without exactly these
    animals, otherwise train it (a few seconds) and cache it."""
    meta = path.with_suffix(".json")
    key = {h: sorted(ids) for h, ids in sorted(holdout.items())}
    if path.is_file() and meta.is_file() and json.loads(meta.read_text()) == key:
        return WeightModel.load(path)
    held = pd.Series(False, index=table.index)
    for herd, ids in holdout.items():
        held |= (table["herd"] == herd) & table["animal_id"].astype(str).isin(ids)
    print(f"training the demo model without {int(held.sum())} held-out animals…", flush=True)
    model = WeightModel.fit(table[~held & table["error"].isna()].reset_index(drop=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    meta.write_text(json.dumps(key))
    return model


def lan_address() -> str:
    """This computer's address on the local network (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


class Demo:
    def __init__(self, samples: list[Sample], herds: dict[str, int], phone_url: str | None) -> None:
        self.samples = {s.id: s for s in samples}
        self.herds = herds
        self.phone_url = phone_url

    def info(self) -> dict[str, object]:
        return {
            "herds": [{"id": hid, "key": herd, **HERDS[herd]} for herd, hid in self.herds.items()],
            "samples": [
                {"id": s.id, "herd_id": s.herd_id, "herd": s.herd, "animal_id": s.animal_id,
                 "breed": HERDS[s.herd]["breed"], "scale_weight_kg": s.scale_weight_kg}
                for s in self.samples.values()
            ],
            "phone_url": self.phone_url,
        }

    def cloud_path(self, sample_id: str, view: View) -> Path | None:
        sample = self.samples.get(sample_id)
        return None if sample is None else sample.paths[view]

    def qr_svg(self) -> str | None:
        if not self.phone_url:
            return None
        try:
            import qrcode
            import qrcode.image.svg
        except ImportError:
            return None
        image = qrcode.make(self.phone_url, image_factory=qrcode.image.svg.SvgPathImage, border=2)
        return image.to_string(encoding="unicode")


def _seed_farm(service: WeightService, herd: str, table: pd.DataFrame, reference: list[str]) -> int:
    """Create the farm with its 20 reference weighings and calibrate it, once."""
    name = HERDS[herd]["farm"]
    for existing in service.store.herds():
        if existing.name == name:
            return existing.id
    farm = service.create_herd(name)
    rows = table[(table["herd"] == herd) & table["animal_id"].astype(str).isin(reference)]
    for _, row in rows.iterrows():
        features = {f: float(row[f]) for f in service.model.features}
        features.update({c: float(row[c]) for c in table.columns if c.endswith("_n_points")})
        service.scan_from_features(farm.id, f"К-{row['animal_id']}", features, float(row["weight_kg"]))
    service.calibrate(farm.id)
    return farm.id


def _read_features(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"animal_id": str}, keep_default_na=False, na_values=[""])


def ensure_features(path: Path, roots: dict[str, Path], workers: int = 4) -> None:
    """Extract features for every requested herd the table does not have yet,
    so a fresh checkout needs no separate ``cattle3d extract`` (about a second
    per animal, once)."""
    from cattle3d.pipeline import extract_herd

    old = _read_features(path) if path.is_file() else pd.DataFrame(columns=["herd"])
    loaders = {HERD_A: load_cowdatabase2, HERD_B: load_cowdatabase}
    missing = [h for h in roots if h not in set(old["herd"])]
    if not missing:
        return
    new = []
    for herd in missing:
        print(f"first run: computing features for {HERDS[herd]['dataset']}…", flush=True)
        new.append(extract_herd(loaders[herd](Path(roots[herd])).animals, workers=workers))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([old, *new] if len(old) else new).to_csv(path, index=False)


def prepare(
    features_path: Path, model_path: Path, db_path: Path,
    cowdb2: Path | None = None, cowdb1: Path | None = None, phone_url: str | None = None,
) -> tuple[WeightService, Demo]:
    loaders = {HERD_A: (cowdb2, load_cowdatabase2), HERD_B: (cowdb1, load_cowdatabase)}
    if cowdb2 is None and cowdb1 is None:
        raise ValueError("give at least one dataset folder (--cowdb2 or --cowdb1)")
    ensure_features(features_path, {h: root for h, (root, _) in loaders.items() if root is not None})
    table = _read_features(features_path)
    chosen = {herd: choose_holdout(table, herd) for herd, (root, _) in loaders.items() if root is not None}
    model = demo_model(table, {h: d + r for h, (d, r) in chosen.items()}, model_path)
    service = WeightService(model, Store(db_path))
    samples: list[Sample] = []
    herds: dict[str, int] = {}
    for herd, (demo_ids, reference) in chosen.items():
        root, load = loaders[herd]
        records: dict[str, AnimalRecord] = {a.animal_id: a for a in load(Path(root)).animals}  # type: ignore[arg-type]
        herds[herd] = _seed_farm(service, herd, table, reference)
        for animal_id in demo_ids:
            if animal_id in records:
                a = records[animal_id]
                samples.append(Sample(f"{herd}-{animal_id}", herds[herd], herd, animal_id, a.weight_kg, a.view_paths))
    return service, Demo(samples, herds, phone_url)
