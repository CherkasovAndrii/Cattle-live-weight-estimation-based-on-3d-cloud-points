"""Build-time setup of the public cloud demo (Railway, or any container host).

Runs the normal demo preparation once inside the image (features of both public
herds, the demo model trained without the held-out animals, and the two seeded
demo farms), then cuts each dataset checkout down to what the running demo
reads: the measurement tables, the station calibration files and the clouds of
the demo animals. The image then carries a few dozen clouds instead of the
whole public releases, and the server starts in seconds.

usage: python scripts/railway_prepare.py PATH_CowDatabase2 PATH_CowDatabase DATA_DIR
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from cattle3d.demo import Demo, prepare
from cattle3d.io.datasets import HERD_A
from cattle3d.types import View


def demo_for(cowdb2: Path, cowdb1: Path, data: Path) -> Demo:
    return prepare(data / "features.csv", data / "demo_model.joblib", data / "demo.sqlite3", cowdb2, cowdb1)[1]


def kept_files(demo: Demo, cowdb2: Path, cowdb1: Path) -> set[Path]:
    """Tables, demo animals' clouds and their capture calibration."""
    keep = {cowdb2 / "Database2022v2.xlsx", cowdb1 / "Measurements.xlsx"}
    for sample in demo.samples.values():
        keep |= set(sample.paths.values())
        left = sample.paths[View.LEFT]
        if sample.herd == HERD_A:
            keep.add(left.parents[2] / "aligned.ply")
        else:
            keep.add(left.parents[1] / "Matrices" / f"{left.stem}.matrices")
    return {p.resolve() for p in keep}


def prune(root: Path, keep: set[Path]) -> int:
    """Delete every file under root that is not kept, then empty folders."""
    removed = 0
    shutil.rmtree(root / ".git", ignore_errors=True)
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() and path.resolve() not in keep:
            path.unlink()
            removed += 1
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    return removed


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    cowdb2, cowdb1, data = (Path(a).resolve() for a in sys.argv[1:])
    demo = demo_for(cowdb2, cowdb1, data)
    keep = kept_files(demo, cowdb2, cowdb1)
    removed = prune(cowdb2, keep) + prune(cowdb1, keep)
    after = demo_for(cowdb2, cowdb1, data)  # the cut-down checkout must give the same demo
    if set(after.samples) != set(demo.samples):
        sys.exit(f"pruning changed the demo animals: {sorted(demo.samples)} -> {sorted(after.samples)}")
    print(f"demo ready: {len(demo.samples)} demo animals, {len(keep)} dataset files kept, {removed} removed", flush=True)


if __name__ == "__main__":
    main()
