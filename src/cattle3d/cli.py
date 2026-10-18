"""Command-line interface: a thin adapter over the library.

Owns: argument parsing, paths and printing. No business logic lives here.

    cattle3d extract     --cowdb2 PATH --cowdb1 PATH          -> data/features.csv
    cattle3d audit-join  --cowdb1 PATH                         -> resources join + reports/cowdb1_join_audit.md
    cattle3d evaluate                                          -> reports/results.md, figures, OOF CSV
    cattle3d train                                             -> artifacts/weight_model.joblib
    cattle3d calibrate   --herd NAME --reference ref.csv       -> stores a herd correction in the model
    cattle3d predict     --left L.ply --right R.ply --top T.ply [--herd NAME]  -> JSON
    cattle3d figures     [--cowdb2 PATH --cowdb1 PATH]               -> reports/figures/thesis/{uk,en}/
    cattle3d serve       [--db data/farm.sqlite3 --port 8000]        -> REST API (needs the [api] extra)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

DEFAULT_FEATURES = Path("data/features.csv")
DEFAULT_MODEL = Path("artifacts/weight_model.joblib")
REPORTS = Path("reports")


def _load_features(path: Path) -> pd.DataFrame:
    if not path.is_file():
        sys.exit(f"{path} not found; run `cattle3d extract` first")
    table = pd.read_csv(path, dtype={"animal_id": str}, keep_default_na=False, na_values=[""])
    failed = table[table["error"].notna()]
    if len(failed):
        print(f"excluding {len(failed)} animal(s) whose views could not be segmented:")
        for _, row in failed.iterrows():
            print(f"  {row['herd']}/{row['animal_id']}: {row['error']}")
    return table[table["error"].isna()].reset_index(drop=True)


def cmd_extract(args: argparse.Namespace) -> None:
    from cattle3d.io.datasets import load_cowdatabase, load_cowdatabase2
    from cattle3d.pipeline import extract_herd

    tables = []
    herds = []
    if args.cowdb2:
        herds.append(load_cowdatabase2(Path(args.cowdb2)))
    if args.cowdb1:
        herds.append(load_cowdatabase(Path(args.cowdb1), join=args.join))
    if not herds:
        sys.exit("give --cowdb2 and/or --cowdb1")
    for herd in herds:
        for skipped in herd.skipped:
            print(f"skip {herd.herd}/{skipped.animal_id}: {skipped.reason}")
        table = extract_herd(herd.animals, workers=args.workers)
        print(f"{herd.herd}: {len(table)} animals, {int((table['error'] != '').sum())} failed")
        tables.append(table)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(out, index=False)
    print(f"wrote {out}")


def cmd_audit_join(args: argparse.Namespace) -> None:
    from importlib import resources

    from cattle3d.audit import audit_join
    from cattle3d.evaluation.report import audit_markdown
    from cattle3d.io.datasets import load_cowdatabase
    from cattle3d.pipeline import extract_herd

    herd = load_cowdatabase(Path(args.cowdb1), join="published")
    table = extract_herd(herd.animals, workers=args.workers)
    audit = audit_join(table[table["error"] == ""])
    control = None
    if args.cowdb2:
        from cattle3d.io.datasets import load_cowdatabase2

        table_a = extract_herd(load_cowdatabase2(Path(args.cowdb2)).animals, workers=args.workers)
        table_a = table_a[table_a["error"] == ""].copy()
        control = audit_join(table_a)
    join_path = Path(str(resources.files("cattle3d.resources").joinpath("cowdb1_audited_join.csv")))
    audit.mapping.to_csv(join_path, index=False)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "cowdb1_join_audit.md").write_text(audit_markdown(audit, control) + "\n")
    print(f"wrote {join_path} and {REPORTS / 'cowdb1_join_audit.md'}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from cattle3d.evaluation.report import (
        results_markdown,
        write_figures,
        write_oof_csv,
    )
    from cattle3d.evaluation.study import run_study

    table = _load_features(Path(args.features))
    seeds = tuple(range(args.seeds))
    result = run_study(table, seeds=seeds)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.md").write_text(results_markdown(result) + "\n")
    write_oof_csv(result, REPORTS / "oof_predictions.csv")
    figures = write_figures(result, REPORTS / "figures")
    print(f"wrote {REPORTS / 'results.md'}, {REPORTS / 'oof_predictions.csv'}")
    for path in figures:
        print(f"wrote {path}")


def cmd_train(args: argparse.Namespace) -> None:
    from cattle3d.modeling.weight_model import WeightModel

    table = _load_features(Path(args.features))
    if args.herds:
        table = table[table["herd"].isin(args.herds)].reset_index(drop=True)
    model = WeightModel.fit(table)
    model.save(Path(args.out))
    print(json.dumps(model.card(), indent=2, ensure_ascii=False))
    print(f"saved {args.out}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    from cattle3d.modeling.weight_model import WeightModel
    from cattle3d.pipeline import features_from_files
    from cattle3d.types import View

    reference = pd.read_csv(args.reference)
    required = {"left", "right", "top", "weight_kg"}
    if not required <= set(reference.columns):
        sys.exit(f"{args.reference} needs columns {sorted(required)}")
    base = Path(args.reference).parent
    rows = []
    for _, r in reference.iterrows():
        paths = {View.LEFT: base / r["left"], View.RIGHT: base / r["right"], View.TOP: base / r["top"]}
        rows.append({**features_from_files(paths), "weight_kg": float(r["weight_kg"])})
    model = WeightModel.load(Path(args.model))
    slope, intercept = model.calibrate_herd(args.herd, pd.DataFrame(rows))
    model.save(Path(args.model))
    print(f"herd {args.herd!r}: weight = {slope:.3f} × model + {intercept:+.1f} kg ({len(rows)} reference animals)")


def cmd_predict(args: argparse.Namespace) -> None:
    from cattle3d.modeling.weight_model import WeightModel
    from cattle3d.pipeline import features_from_files
    from cattle3d.types import View

    model = WeightModel.load(Path(args.model))
    features = features_from_files(
        {View.LEFT: Path(args.left), View.RIGHT: Path(args.right), View.TOP: Path(args.top)}
    )
    prediction = model.predict_row(features, herd=args.herd)
    print(json.dumps(asdict(prediction), indent=2))


def cmd_figures(args: argparse.Namespace) -> None:
    from cattle3d.figures.thesis import write_thesis_figures

    table = _load_features(Path(args.features)) if Path(args.features).is_file() else None
    study = None
    if table is not None and not args.no_results:
        from cattle3d.evaluation.study import run_study

        study = run_study(table, seeds=tuple(range(args.seeds)))
    half_width = None
    if Path(args.model).is_file():
        from cattle3d.modeling.weight_model import WeightModel

        half_width = WeightModel.load(Path(args.model)).interval_half_width_kg
    paths = write_thesis_figures(
        Path(args.out), tuple(args.lang), table,
        Path(args.cowdb2) if args.cowdb2 else None, Path(args.cowdb1) if args.cowdb1 else None,
        study, half_width,
    )
    for path in paths:
        print(f"wrote {path}")


def cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn

        from cattle3d.api.app import app_from_files
    except ImportError:
        sys.exit('the API needs its extra: pip install -e ".[api]"')
    api_key = os.environ.get("CATTLE3D_API_KEY")
    if not api_key and args.host not in ("127.0.0.1", "localhost"):
        sys.exit("set CATTLE3D_API_KEY before serving beyond localhost")
    app = app_from_files(Path(args.model), Path(args.db), api_key)
    uvicorn.run(app, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cattle3d", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="segment clouds and compute features for the public herds")
    p.add_argument("--cowdb2", help="path to a CowDatabase2 checkout (herd A)")
    p.add_argument("--cowdb1", help="path to a CowDatabase checkout (herd B)")
    p.add_argument("--join", choices=["audited", "published"], default="audited")
    p.add_argument("--out", default=str(DEFAULT_FEATURES))
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("audit-join", help="re-derive the CowDatabase file-to-row join")
    p.add_argument("--cowdb1", required=True)
    p.add_argument("--cowdb2", help="also run the control on CowDatabase2")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_audit_join)

    p = sub.add_parser("evaluate", help="run the study and write reports/")
    p.add_argument("--features", default=str(DEFAULT_FEATURES))
    p.add_argument("--seeds", type=int, default=5)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("train", help="fit and save the deployable model")
    p.add_argument("--features", default=str(DEFAULT_FEATURES))
    p.add_argument("--herds", nargs="*", help="limit training to these herds (default: all)")
    p.add_argument("--out", default=str(DEFAULT_MODEL))
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("calibrate", help="fit a herd correction from reference weighings")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--herd", required=True, help="name to store the correction under")
    p.add_argument(
        "--reference", required=True,
        help="CSV with columns left,right,top (PLY paths, relative to the CSV) and weight_kg",
    )
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("predict", help="predict one animal from its three PLY clouds")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--top", required=True)
    p.add_argument("--herd", help="apply this herd's stored calibration offset")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("figures", help="thesis schemes and figures, Ukrainian and English")
    p.add_argument("--features", default=str(DEFAULT_FEATURES))
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="its interval width goes on the decision scheme")
    p.add_argument("--cowdb2", help="CowDatabase2 checkout: stage figures on a real cow")
    p.add_argument("--cowdb1", help="CowDatabase checkout: the join-audit figure")
    p.add_argument("--lang", nargs="+", choices=["uk", "en"], default=["uk", "en"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--no-results", action="store_true", help="skip the study-result figures (no 30 s study run)")
    p.add_argument("--out", default=str(REPORTS / "figures" / "thesis"))
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("serve", help="run the REST API (API key from CATTLE3D_API_KEY)")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--db", default="data/farm.sqlite3", help="SQLite file for herds, scans and calibrations")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
