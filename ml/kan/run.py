"""CLI / core entry point for training KAN, MLP, and SINDy on residual data."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

try:  # package import
    from .baselines import MLP, sindy, sindy_predict
    from .dataset import build_xy, load_residual_table, resolve_times, split_by_time
    from .kan import KAN, export_results, mse, r2_score, train
except ImportError:  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from baselines import MLP, sindy, sindy_predict
    from dataset import build_xy, load_residual_table, resolve_times, split_by_time
    from kan import KAN, export_results, mse, r2_score, train


def _synthetic_data():
    random.seed(0)
    N = 200

    def f(x0, x1):
        return 0.5 * x0 * x0 - 0.3 * math.sin(x1) + 0.1 * x0 * x1

    X = [[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
         for _ in range(N)]
    Y = [[f(x[0], x[1])] for x in X]
    return X, Y, "0.5*x0^2 - 0.3*sin(x1) + 0.1*x0*x1"


def _parse_cols(value):
    if value is None:
        return None
    return [c.strip() for c in value.split(",") if c.strip()]


def load_mission(path):
    """Load burn windows from a mission JSON file.

    Accepts the Rust ``Mission`` schema (``burns: [{label, start_s, end_s}]``)
    and is lenient about ``start``/``end`` spellings.  Burn windows are
    half-open ``[start_s, end_s)``, matching ``BurnWindow::covers``.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise TypeError(f"mission file must contain a JSON object: {path}")

    burns = []
    for idx, burn in enumerate(raw.get("burns") or []):
        if not isinstance(burn, dict):
            raise TypeError(f"mission burn #{idx} must be a JSON object")
        start = burn.get("start_s", burn.get("start"))
        end = burn.get("end_s", burn.get("end"))
        if start is None or end is None:
            raise ValueError(
                f"mission burn #{idx} needs start_s/end_s (or start/end)"
            )
        start, end = float(start), float(end)
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            raise ValueError(f"mission burn #{idx} has invalid bounds")
        burns.append({
            "label": str(burn.get("label", f"burn{idx}")),
            "start_s": start,
            "end_s": end,
        })
    return {"path": str(path), "name": raw.get("name"), "burns": burns}


def burn_keep_mask(times, burns):
    """Return per-sample booleans; ``True`` means the sample is usable."""
    return [
        not any(b["start_s"] <= float(t) < b["end_s"] for b in burns)
        for t in times
    ]


def apply_burn_mask(X, Y, times, burns):
    """Drop samples that fall inside a mission burn window.

    Returns ``(X, Y, times, masked_fraction, masked_samples)``.  ``times`` is
    required because the mask is applied in physical time before any split.
    """
    if not (len(X) == len(Y) == len(times)):
        raise ValueError("X, Y, and times must have the same length")
    if not burns:
        return list(X), list(Y), list(times), 0.0, 0

    keep = burn_keep_mask(times, burns)
    X_keep = [row for row, k in zip(X, keep) if k]
    Y_keep = [row for row, k in zip(Y, keep) if k]
    t_keep = [float(t) for t, k in zip(times, keep) if k]
    masked = len(times) - len(X_keep)
    masked_fraction = masked / len(times) if times else 0.0
    return X_keep, Y_keep, t_keep, masked_fraction, masked


def _eval(model, X, Y, predict=None):
    """Return ``(mse, r2)`` for a model or prediction function."""
    pred = model.forward(X) if predict is None else predict(X)
    return mse(pred, Y), r2_score(pred, Y)


def run(args) -> tuple:
    random.seed(0)

    mission = None
    if getattr(args, "mission", None):
        mission = load_mission(args.mission)

    if getattr(args, "data", None):
        df = load_residual_table(args.data)
        X, Y = build_xy(
            df,
            feature_cols=_parse_cols(getattr(args, "features", None)),
            target_cols=_parse_cols(getattr(args, "targets", None)),
        )
        target_desc = getattr(args, "targets", None) or "residual target"
        times = resolve_times(df, getattr(args, "time_col", None))
        if times is None:
            if mission is not None and mission["burns"]:
                raise ValueError(
                    "mission burn masking needs a time column in the residual "
                    "table; pass --time-col"
                )
            times = [float(i) for i in range(len(X))]
            time_source = "index"
        else:
            time_source = "column"
    else:
        X, Y, target_desc = _synthetic_data()
        times = [float(i) for i in range(len(X))]
        time_source = "index"

    n_samples_raw = len(X)
    masked_fraction = 0.0
    masked_samples = 0
    if mission is not None:
        X, Y, times, masked_fraction, masked_samples = apply_burn_mask(
            X, Y, times, mission["burns"]
        )

    if len(X) < 3:
        raise ValueError(
            f"not enough samples after burn masking ({len(X)}); need at least 3"
        )

    test_fraction = float(getattr(args, "test_fraction", 0.2))
    # Held-out evaluation: never shuffle residual time series.
    X_train, X_test, Y_train, Y_test = split_by_time(
        X, Y, t=times, test_fraction=test_fraction
    )
    if not X_train or not X_test:
        raise ValueError(
            f"time split produced an empty set (train={len(X_train)}, "
            f"test={len(X_test)})"
        )

    n_features = len(X_train[0])
    out_dim = len(Y_train[0])
    hidden = max(4, n_features)

    epochs = int(getattr(args, "epochs", 150))
    lr = float(getattr(args, "lr", 2e-1))

    # --- KAN ---
    kan = KAN([n_features, hidden, out_dim], num_basis=8, degree=3,
              grid_range=(-2.0, 2.0), standardize=True, l1=1e-4)
    train(kan, X_train, Y_train, epochs=epochs, lr=lr, verbose=False)

    # pruning + short retrain (matches original demo behaviour)
    kan.prune(1e-2)
    train(kan, X_train, Y_train, epochs=20, lr=1e-2, verbose=False)

    kan_train_mse, kan_train_r2 = _eval(kan, X_train, Y_train)
    kan_test_mse, kan_test_r2 = _eval(kan, X_test, Y_test)

    edges = []
    for i in range(n_features):
        for j in range(out_dim):
            fit = kan.symbolic_edge(0, i, j)
            edges.append({
                "layer": 0,
                "i": i,
                "j": j,
                "coeffs": fit.coeffs,
                "r2": fit.r2,
                "rmse": fit.rmse,
            })

    # --- MLP ---
    mlp = MLP([n_features, 8, out_dim])
    mlp.train(X_train, Y_train, epochs=epochs, lr=5e-1, verbose=False)
    mlp_train_mse, mlp_train_r2 = _eval(mlp, X_train, Y_train)
    mlp_test_mse, mlp_test_r2 = _eval(mlp, X_test, Y_test)

    # --- SINDy (fit on train, evaluate on both) ---
    sindy_degree = 3
    Xi, names, exprs = sindy(X_train, Y_train, threshold=0.05, degree=sindy_degree)
    sindy_predict_fn = lambda batch: sindy_predict(batch, Xi, degree=sindy_degree)
    sindy_train_mse, sindy_train_r2 = _eval(
        None, X_train, Y_train, predict=sindy_predict_fn
    )
    sindy_test_mse, sindy_test_r2 = _eval(
        None, X_test, Y_test, predict=sindy_predict_fn
    )

    results = {
        "n_samples": len(X),
        "n_samples_raw": n_samples_raw,
        "masked_fraction": masked_fraction,
        "burn": {
            "mission": mission["path"] if mission is not None else None,
            "name": mission.get("name") if mission is not None else None,
            "burns": mission["burns"] if mission is not None else [],
            "masked_samples": masked_samples,
            "masked_fraction": masked_fraction,
        },
        "split": {
            "strategy": "time",
            "test_fraction": test_fraction,
            "time_source": time_source,
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        "target": target_desc,
        "kan": {
            "dims": kan.dims,
            "num_basis": 8,
            "degree": 3,
            "grid_range": list(kan.grid_range),
            "standardize": kan.standardize,
            "l1": kan.l1,
            "num_parameters": kan.num_parameters(),
            "train_mse": kan_train_mse,
            "train_r2": kan_train_r2,
            "test_mse": kan_test_mse,
            "test_r2": kan_test_r2,
            "edges": edges,
        },
        "mlp": {
            "train_mse": mlp_train_mse,
            "train_r2": mlp_train_r2,
            "test_mse": mlp_test_mse,
            "test_r2": mlp_test_r2,
        },
        "sindy": {
            "degree": sindy_degree,
            "features": names,
            "expression": exprs[0] if exprs else "",
            "expressions": exprs,
            "train_mse": sindy_train_mse,
            "train_r2": sindy_train_r2,
            "test_mse": sindy_test_mse,
            "test_r2": sindy_test_r2,
        },
    }

    out_path = getattr(args, "out", None) or "ml/kan/kan_result.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_results(str(out_path), results)

    print(f"split: train={len(X_train)} test={len(X_test)} "
          f"masked_fraction={masked_fraction:.3f}")
    print(f"KAN   train mse={kan_train_mse:.6e} r2={kan_train_r2:.6f} | "
          f"test mse={kan_test_mse:.6e} r2={kan_test_r2:.6f}")
    print(f"MLP   train mse={mlp_train_mse:.6e} r2={mlp_train_r2:.6f} | "
          f"test mse={mlp_test_mse:.6e} r2={mlp_test_r2:.6f}")
    print(f"SINDy train mse={sindy_train_mse:.6e} r2={sindy_train_r2:.6f} | "
          f"test mse={sindy_test_mse:.6e} r2={sindy_test_r2:.6f} "
          f"expr={exprs[0] if exprs else ''}")
    print(f"wrote {out_path}")
    return results, str(out_path)


def build_parser():
    parser = argparse.ArgumentParser(description="Train KAN/MLP/SINDy.")
    parser.add_argument("--data", default=None, help="path to residual table (parquet/csv)")
    parser.add_argument("--features", default=None, help="comma-separated feature columns")
    parser.add_argument("--targets", default=None, help="comma-separated target columns")
    parser.add_argument("--mission", default=None,
                        help="mission JSON with burn windows to mask")
    parser.add_argument("--time-col", default=None,
                        help="time column used for the held-out split and burn mask")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="fraction of latest samples held out for testing")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-1)
    parser.add_argument("--out", default="ml/kan/kan_result.json")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


# Compatibility aliases for callers expecting a "core" entry point.
run_experiment = run
run_core = run


if __name__ == "__main__":
    main()