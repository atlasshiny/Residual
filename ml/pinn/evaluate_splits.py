"""Evaluate the residual models across split modes and seeds.

The single-run CLI in :mod:`ml.pinn.run` reports one split and one seed.  This
module reruns the same training pipeline for every requested
``(split, seed)`` pair, keeps each run's artifacts in its own directory, and
writes an aggregate report with mean/spread so a score cannot be selected from
a lucky seed.

Usage:
    python -m ml.pinn.evaluate_splits --data data/residual_table.csv \
        --splits random,time --seeds 0,1,2 \
        --out results/pinn_split_eval.json

``group`` splits need a table with at least two groups in ``--group-col`` and
are not enabled by default because the single-arc Moon table cannot support
them.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from .run import build_parser, run

__all__ = ["METRICS", "MODELS", "aggregate_runs", "build_run_args", "main"]

MODELS = ("pinn", "mlp", "sindy", "linreg")
METRICS = ("train_mse", "train_r2", "test_mse", "test_r2")


def _csv_values(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _finite_values(runs, model, metric):
    values = []
    for record in runs:
        results = record.get("results")
        if not results:
            continue
        try:
            value = float(results[model][metric])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def aggregate_runs(runs):
    """Aggregate per-model metrics over successful runs.

    ``runs`` is a list of records with a ``results`` entry shaped like the JSON
    written by :func:`ml.pinn.run.run`.  Non-finite values and failed runs are
    skipped, and ``n`` records how many values contributed to each statistic.
    """
    aggregate = {}
    for model in MODELS:
        aggregate[model] = {}
        for metric in METRICS:
            values = _finite_values(runs, model, metric)
            if not values:
                aggregate[model][metric] = {"n": 0}
                continue
            aggregate[model][metric] = {
                "n": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
    return aggregate


def build_run_args(
    data,
    split,
    seed,
    run_dir,
    group_col="arc_id",
    test_fraction=0.2,
    epochs=None,
    lr=None,
    features=None,
    targets=None,
    time_col=None,
    engineer=False,
):
    """Build the argument namespace for one isolated training run."""
    argv = [
        "--data",
        str(data),
        "--split",
        split,
        "--seed",
        str(seed),
        "--test-fraction",
        str(test_fraction),
        "--out",
        str(Path(run_dir) / "pinn_result.json"),
        "--weights-out",
        str(Path(run_dir) / "pinn_models.pt"),
        "--mlp-weights-out",
        str(Path(run_dir) / "mlp_model.pt"),
        "--predictions-out",
        str(Path(run_dir) / "pinn_predictions.json"),
        "--mlp-predictions-out",
        str(Path(run_dir) / "mlp_predictions.json"),
        "--mlp-predictions-csv-out",
        str(Path(run_dir) / "mlp_predictions.csv"),
        "--predictions-csv-out",
        str(Path(run_dir) / "pinn_predictions.csv"),
        "--sindy-predictions-csv-out",
        str(Path(run_dir) / "sindy_predictions.csv"),
    ]
    if group_col:
        argv += ["--group-col", group_col]
    if epochs is not None:
        argv += ["--epochs", str(epochs)]
    if lr is not None:
        argv += ["--lr", str(lr)]
    if features:
        argv += ["--features", features]
    if targets:
        argv += ["--targets", targets]
    if time_col:
        argv += ["--time-col", time_col]
    if engineer:
        argv += ["--engineer"]
    return build_parser().parse_args(argv)


def _summary_rows(aggregate):
    rows = []
    for model in MODELS:
        row = {"model": model}
        for metric in ("test_r2", "test_mse"):
            stats = aggregate[model][metric]
            row[metric] = stats.get("mean")
            row[f"{metric}_std"] = stats.get("std")
            row[f"{metric}_n"] = stats.get("n", 0)
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--splits",
        default="random,time",
        help="comma-separated split modes (random,time,group)",
    )
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    parser.add_argument("--group-col", default="arc_id")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--engineer", action="store_true")
    parser.add_argument("--out", default="results/pinn_split_eval.json")
    parser.add_argument("--run-dir", default="results/pinn_split_runs")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_root = Path(args.run_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    runs = []
    for split in _csv_values(args.splits):
        for seed_text in _csv_values(args.seeds):
            seed = int(seed_text)
            run_dir = run_root / f"{split}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== split={split} seed={seed} ===")
            try:
                run_args = build_run_args(
                    args.data,
                    split,
                    seed,
                    run_dir,
                    group_col=args.group_col,
                    test_fraction=args.test_fraction,
                    epochs=args.epochs,
                    lr=args.lr,
                    features=args.features,
                    targets=args.targets,
                    time_col=args.time_col,
                    engineer=args.engineer,
                )
                results, metrics_path = run(run_args)
            except ValueError as exc:
                runs.append(
                    {
                        "split": split,
                        "seed": seed,
                        "error": str(exc),
                    }
                )
                print(f"skipped split={split} seed={seed}: {exc}")
                continue
            runs.append(
                {
                    "split": split,
                    "seed": seed,
                    "metrics_path": str(metrics_path),
                    "split_metadata": results["split"],
                    "results": results,
                }
            )

    aggregate = aggregate_runs(runs)
    report = {
        "schema": "pinn-split-eval/v1",
        "data": str(args.data),
        "splits": _csv_values(args.splits),
        "seeds": [int(value) for value in _csv_values(args.seeds)],
        "test_fraction": args.test_fraction,
        "runs": runs,
        "aggregate": aggregate,
        "summary": _summary_rows(aggregate),
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote split evaluation to {out_path}")
    for row in report["summary"]:
        if row["test_r2_n"]:
            print(
                f"{row['model']:7s} test R2 "
                f"{row['test_r2']:.6f} +/- {row['test_r2_std']:.6f} "
                f"(n={row['test_r2_n']})"
            )
    return report, out_path


if __name__ == "__main__":
    main()
