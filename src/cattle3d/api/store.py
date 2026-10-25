"""SQLite storage for herds, scans, scale weighings and herd calibrations.

Owns: the schema and every SQL statement. Must not know about the model,
features or HTTP. One connection per operation, so the store is safe to share
between the server's worker threads.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS herds (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calibrations (
    id            INTEGER PRIMARY KEY,
    herd_id       INTEGER NOT NULL REFERENCES herds(id),
    created_utc   TEXT NOT NULL,
    slope         REAL NOT NULL,
    intercept_kg  REAL NOT NULL,
    n_reference   INTEGER NOT NULL,
    half_width_kg REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS scans (
    id              INTEGER PRIMARY KEY,
    herd_id         INTEGER NOT NULL REFERENCES herds(id),
    animal_tag      TEXT NOT NULL,
    created_utc     TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('trusted', 'flagged', 'refused')),
    reasons         TEXT NOT NULL,              -- JSON list of strings
    raw_weight_kg   REAL,                       -- model output before herd correction
    weight_kg       REAL,
    lower_kg        REAL,
    upper_kg        REAL,
    calibration_id  INTEGER REFERENCES calibrations(id),
    features        TEXT,                       -- JSON object, kept for retraining
    model_version   TEXT NOT NULL,
    scale_weight_kg REAL,
    weighed_utc     TEXT
);
CREATE INDEX IF NOT EXISTS scans_by_animal ON scans (herd_id, animal_tag, id);
"""


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Herd:
    id: int
    name: str
    created_utc: str


@dataclass(frozen=True)
class Calibration:
    id: int
    herd_id: int
    created_utc: str
    slope: float
    intercept_kg: float
    n_reference: int
    half_width_kg: float


@dataclass(frozen=True)
class Scan:
    id: int
    herd_id: int
    animal_tag: str
    created_utc: str
    status: str
    reasons: list[str]
    raw_weight_kg: float | None
    weight_kg: float | None
    lower_kg: float | None
    upper_kg: float | None
    calibration_id: int | None
    model_version: str
    scale_weight_kg: float | None
    weighed_utc: str | None


@dataclass(frozen=True)
class NewScan:
    herd_id: int
    animal_tag: str
    status: str
    reasons: list[str]
    raw_weight_kg: float | None
    weight_kg: float | None
    lower_kg: float | None
    upper_kg: float | None
    calibration_id: int | None
    features: dict[str, float] | None
    model_version: str


class DuplicateError(ValueError):
    """A herd with this name already exists."""


_SCAN_COLUMNS = (
    "id, herd_id, animal_tag, created_utc, status, reasons, raw_weight_kg, weight_kg, lower_kg, "
    "upper_kg, calibration_id, model_version, scale_weight_kg, weighed_utc"
)


def _scan(row: sqlite3.Row) -> Scan:
    return Scan(**{**dict(row), "reasons": json.loads(row["reasons"])})


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:  # commits on success, rolls back on error
                yield db
        finally:
            db.close()

    # ---- herds ---------------------------------------------------------------
    def add_herd(self, name: str) -> Herd:
        created = now_utc()
        try:
            with self._connect() as db:
                cur = db.execute("INSERT INTO herds (name, created_utc) VALUES (?, ?)", (name, created))
        except sqlite3.IntegrityError as exc:
            raise DuplicateError(f"herd {name!r} already exists") from exc
        return Herd(int(cur.lastrowid), name, created)

    def herd(self, herd_id: int) -> Herd | None:
        with self._connect() as db:
            row = db.execute("SELECT id, name, created_utc FROM herds WHERE id = ?", (herd_id,)).fetchone()
        return Herd(**dict(row)) if row else None

    def herds(self) -> list[Herd]:
        with self._connect() as db:
            rows = db.execute("SELECT id, name, created_utc FROM herds ORDER BY id").fetchall()
        return [Herd(**dict(r)) for r in rows]

    # ---- calibrations ------------------------------------------------------------
    def add_calibration(
        self, herd_id: int, slope: float, intercept_kg: float, n_reference: int, half_width_kg: float
    ) -> Calibration:
        created = now_utc()
        with self._connect() as db:
            cur = db.execute(
                "INSERT INTO calibrations (herd_id, created_utc, slope, intercept_kg, n_reference, half_width_kg)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (herd_id, created, slope, intercept_kg, n_reference, half_width_kg),
            )
        return Calibration(int(cur.lastrowid), herd_id, created, slope, intercept_kg, n_reference, half_width_kg)

    def current_calibration(self, herd_id: int) -> Calibration | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM calibrations WHERE herd_id = ? ORDER BY id DESC LIMIT 1", (herd_id,)
            ).fetchone()
        return Calibration(**dict(row)) if row else None

    # ---- scans -------------------------------------------------------------------
    def add_scan(self, scan: NewScan) -> Scan:
        with self._connect() as db:
            cur = db.execute(
                "INSERT INTO scans (herd_id, animal_tag, created_utc, status, reasons, raw_weight_kg, weight_kg,"
                " lower_kg, upper_kg, calibration_id, features, model_version)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    scan.herd_id, scan.animal_tag, now_utc(), scan.status, json.dumps(scan.reasons),
                    scan.raw_weight_kg, scan.weight_kg, scan.lower_kg, scan.upper_kg, scan.calibration_id,
                    None if scan.features is None else json.dumps(scan.features), scan.model_version,
                ),
            )
        found = self.scan(int(cur.lastrowid))
        assert found is not None
        return found

    def scan(self, scan_id: int) -> Scan | None:
        with self._connect() as db:
            row = db.execute(f"SELECT {_SCAN_COLUMNS} FROM scans WHERE id = ?", (scan_id,)).fetchone()
        return _scan(row) if row else None

    def set_scale_weight(self, scan_id: int, weight_kg: float) -> Scan | None:
        with self._connect() as db:
            db.execute(
                "UPDATE scans SET scale_weight_kg = ?, weighed_utc = ? WHERE id = ?",
                (weight_kg, now_utc(), scan_id),
            )
        return self.scan(scan_id)

    def animal_scans(self, herd_id: int, animal_tag: str) -> list[Scan]:
        with self._connect() as db:
            rows = db.execute(
                f"SELECT {_SCAN_COLUMNS} FROM scans WHERE herd_id = ? AND animal_tag = ? ORDER BY id",
                (herd_id, animal_tag),
            ).fetchall()
        return [_scan(r) for r in rows]

    def herd_scans(self, herd_id: int, limit: int = 50) -> list[Scan]:
        """The herd's most recent scans, newest first."""
        with self._connect() as db:
            rows = db.execute(
                f"SELECT {_SCAN_COLUMNS} FROM scans WHERE herd_id = ? ORDER BY id DESC LIMIT ?", (herd_id, limit)
            ).fetchall()
        return [_scan(r) for r in rows]

    def reference_weighings(self, herd_id: int) -> list[Scan]:
        """The latest weighed, usable scan of each animal in the herd."""
        with self._connect() as db:
            rows = db.execute(
                f"SELECT {_SCAN_COLUMNS} FROM scans WHERE id IN ("
                " SELECT MAX(id) FROM scans WHERE herd_id = ? AND scale_weight_kg IS NOT NULL"
                " AND raw_weight_kg IS NOT NULL GROUP BY animal_tag) ORDER BY id",
                (herd_id,),
            ).fetchall()
        return [_scan(r) for r in rows]

    def spot_weighings(self, calibration_id: int) -> list[Scan]:
        """Weighed scans predicted under a calibration, in weighing order."""
        with self._connect() as db:
            rows = db.execute(
                f"SELECT {_SCAN_COLUMNS} FROM scans WHERE calibration_id = ? AND scale_weight_kg IS NOT NULL"
                " AND weight_kg IS NOT NULL ORDER BY weighed_utc, id",
                (calibration_id,),
            ).fetchall()
        return [_scan(r) for r in rows]

    def count_scans(self, herd_id: int) -> dict[str, int]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT status, COUNT(*) AS n FROM scans WHERE herd_id = ? GROUP BY status", (herd_id,)
            ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}
