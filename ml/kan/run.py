"""CLI / core entry point for training KAN, MLP, and SINDy on residual data."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

try:  # package import
    from .baselines import MLP, sindy, sindy_predict
    from .baselines import mse as baseline_mse
    from .dataset import build_xy, load_residual_table
    from .kan import KAN, export_results, mse, r2_score, train
except ImportError:  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from baselines import MLP, sindy, sindy_predict
    from baselines import mse as baseline_mse
    from dataset import build_xy, load_residual_table
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


def run(args) -> tuple:
    random.seed(0)

    if getattr(args, "data", None):
        df = load_residual_table(args.data)
        X, Y = build_xy(
            df,
            feature_cols=_parse_cols(getattr(args, "features", None)),
            target_cols=_parse_cols(getattr(args, "targets", None)),
        )
        target_desc = getattr(args, "targets", None) or "residual target"
    else:
        X, Y, target_desc = _synthetic_data()

    n_features = len(X[0]) if X else 0
    out_dim = len(Y[0]) if Y else 1
    hidden = max(4, n_features)

    epochs = int(getattr(args, "epochs", 150))
    lr = float(getattr(args, "lr", 2e-1))

    # --- KAN ---
    kan = KAN([n_features, hidden, out_dim], num_basis=8, degree=3,
              grid_range=(-2.0, 2.0), standardize=True, l1=1e-4)
    train(kan, X, Y, epochs=epochs, lr=lr, verbose=False)

    # pruning + short retrain (matches original demo behaviour)
    kan.prune(1e-2)
    train(kan, X, Y, epochs=20, lr=1e-2, verbose=False)

    kan_pred = kan.forward(X)
    kan_mse = mse(kan_pred, Y)
    kan_r2 = r2_score(kan_pred, Y)

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
    mlp.train(X, Y, epochs=epochs, lr=5e-1, verbose=False)
    mlp_pred = mlp.forward(X)
    mlp_mse = baseline_mse(mlp_pred, Y)
    mlp_r2 = r2_score(mlp_pred, Y)

    # --- SINDy ---
    sindy_degree = 3
    Xi, names, exprs = sindy(X, Y, threshold=0.05, degree=sindy_degree)
    sp = sindy_predict(X, Xi, degree=sindy_degree)
    sindy_mse = baseline_mse(sp, Y)
    sindy_r2 = r2_score(sp, Y)

    results = {
        "n_samples": len(X),
        "target": target_desc,
        "kan": {
            "dims": kan.dims,
            "num_basis": 8,
            "degree": 3,
            "grid_range": list(kan.grid_range),
            "standardize": kan.standardize,
            "l1": kan.l1,
            "num_parameters": kan.num_parameters(),
            "train_mse": kan_mse,
            "train_r2": kan_r2,
            "edges": edges,
        },
        "mlp": {"train_mse": mlp_mse, "train_r2": mlp_r2},
        "sindy": {
            "degree": sindy_degree,
            "features": names,
            "expression": exprs[0] if exprs else "",
            "expressions": exprs,
            "train_mse": sindy_mse,
            "train_r2": sindy_r2,
        },
    }

    out_path = getattr(args, "out", None) or "ml/kan/kan_result.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_results(str(out_path), results)

    print(f"KAN  mse={kan_mse:.6e} r2={kan_r2:.6f}")
    print(f"MLP  mse={mlp_mse:.6e} r2={mlp_r2:.6f}")
    print(f"SINDy mse={sindy_mse:.6e} r2={sindy_r2:.6f} expr={exprs[0] if exprs else ''}")
    print(f"wrote {out_path}")
    return results, str(out_path)


def build_parser():
    parser = argparse.ArgumentParser(description="Train KAN/MLP/SINDy.")
    parser.add_argument("--data", default=None, help="path to residual table (parquet/csv)")
    parser.add_argument("--features", default=None, help="comma-separated feature columns")
    parser.add_argument("--targets", default=None, help="comma-separated target columns")
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