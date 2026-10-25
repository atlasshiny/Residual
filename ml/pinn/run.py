"""CLI / core entry point for training PINN, MLP, and SINDy on residual data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

try:
    from .baselines import MLP, linear_predict, linear_regression, sindy, sindy_predict
    from .dataset import (
        build_xy,
        load_residual_table,
        resolve_times,
    )
    from .features import engineer_features
    from .pinn import PINN, train_pinn
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from baselines import MLP, linear_predict, linear_regression, sindy, sindy_predict
    from dataset import (
        build_xy,
        load_residual_table,
        resolve_times,
    )
    from features import engineer_features
    from pinn import PINN, train_pinn

import numpy as np
from sklearn.preprocessing import StandardScaler


def _parse_cols(value):
    if value is None:
        return None
    return [c.strip() for c in value.split(",") if c.strip()]


def load_mission(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    burns = []
    for idx, burn in enumerate(raw.get("burns") or []):
        start = burn.get("start_s", burn.get("start"))
        end = burn.get("end_s", burn.get("end"))
        start, end = float(start), float(end)
        burns.append(
            {
                "label": str(burn.get("label", f"burn{idx}")),
                "start_s": start,
                "end_s": end,
            }
        )
    return {"path": str(path), "name": raw.get("name"), "burns": burns}


def _eval(model, X, Y, predict=None):
    pred = model.forward(X) if predict is None else predict(X)
    pred = np.asarray(pred)
    target = np.asarray(Y)
    mse = np.mean((pred - target) ** 2)
    mean_y = np.mean(target, axis=0)
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - mean_y) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mse, r2


def run(args) -> tuple:
    random.seed(0)
    mission = load_mission(args.mission) if args.mission else None

    if args.data:
        df = load_residual_table(args.data)
        time_col = args.time_col or "t"

        # Masking
        if mission:
            times = resolve_times(df, time_col)
            keep = [
                not any(b["start_s"] <= float(t) < b["end_s"] for b in mission["burns"])
                for t in times
            ]
            df = df[keep].reset_index(drop=True)
            masked_frac = (
                len(resolve_times(load_residual_table(args.data), time_col)) - len(df)
            ) / len(resolve_times(load_residual_table(args.data), time_col))
        else:
            masked_frac = 0.0

        # Feature Engineering
        if args.engineer or args.features is None:
            df, feature_cols = engineer_features(df, time_col)
        else:
            feature_cols = _parse_cols(args.features)

        # Target selection
        pos_cols = _parse_cols(args.targets) if args.targets else ["drx", "dry", "drz"]
        vel_cols = ["dvx", "dvy", "dvz"]

        # Temporal Split
        test_fraction = float(args.test_fraction)
        split_idx = int(len(df) * (1 - test_fraction))
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]

        # Scalers
        scaler_pos = StandardScaler()
        scaler_vel = StandardScaler()

        # Fit on train
        Y_pos_raw_train = df_train[pos_cols].values
        Y_vel_raw_train = df_train[vel_cols].values
        Y_pos_scaled_train = scaler_pos.fit_transform(Y_pos_raw_train)
        Y_vel_scaled_train = scaler_vel.fit_transform(Y_vel_raw_train)

        # Transform test
        Y_pos_scaled_test = scaler_pos.transform(df_test[pos_cols].values)
        scaler_vel.transform(df_test[vel_cols].values)

        # Split train into fit + val
        val_frac = 0.1
        n_val = max(1, int(len(df_train) * val_frac))
        df_fit = df_train.iloc[:-n_val]
        df_val = df_train.iloc[-n_val:]

        Y_pos_scaled_fit = Y_pos_scaled_train[:-n_val]
        Y_vel_scaled_fit = Y_vel_scaled_train[:-n_val]
        Y_pos_scaled_val = Y_pos_scaled_train[-n_val:]
        Y_vel_scaled_val = Y_vel_scaled_train[-n_val:]

        # X Data
        X_fit, _ = build_xy(df_fit, feature_cols=feature_cols, target_cols=pos_cols)
        X_val, _ = build_xy(df_val, feature_cols=feature_cols, target_cols=pos_cols)
        X_train, _ = build_xy(df_train, feature_cols=feature_cols, target_cols=pos_cols)
        X_test, _ = build_xy(df_test, feature_cols=feature_cols, target_cols=pos_cols)

        # PINN Specific Input (Time only)
        t_col = "t_norm" if "t_norm" in df.columns else time_col
        X_pinn_fit = df_fit[[t_col]].values.tolist()
        X_pinn_val = df_val[[t_col]].values.tolist()
        X_pinn_train = df_train[[t_col]].values.tolist()
        X_pinn_test = df_test[[t_col]].values.tolist()
    else:
        # Synthetic fallback
        X_fit, Y_pos_scaled_fit = [[0.0]] * 10, [[0.0] * 3] * 10
        X_val, Y_pos_scaled_val = [[0.0]] * 2, [[0.0] * 3] * 2
        X_train, Y_pos_scaled_train = [[0.0]] * 12, [[0.0] * 3] * 12
        X_test, Y_pos_scaled_test = [[0.0]] * 2, [[0.0] * 3] * 2
        X_pinn_fit = [[0.0]] * 10
        X_pinn_val = [[0.0]] * 2
        X_pinn_train = [[0.0]] * 12
        X_pinn_test = [[0.0]] * 2
        Y_vel_scaled_fit = [[0.0] * 3] * 10
        Y_vel_scaled_val = [[0.0] * 3] * 2
        Y_pos_scaled_fit_p = [[0.0] * 3] * 10
        Y_vel_scaled_fit_p = [[0.0] * 3] * 10
        Y_pos_scaled_val_p = [[0.0] * 3] * 2
        Y_vel_scaled_val_p = [[0.0] * 3] * 2
        Y_pos_scaled_train_p = [[0.0] * 3] * 12
        Y_pos_scaled_test_p = [[0.0] * 3] * 2
        Y_pos_raw_train = [[0.0] * 3] * 12
        Y_pos_raw_test = [[0.0] * 3] * 2
        df_test = None
        pos_cols = ["y"]
        feature_cols = "synthetic"
        masked_frac = 0.0
        scaler_pos = None
        df = None

    n_features = len(X_fit[0])
    out_dim = len(Y_pos_scaled_fit[0])

    # --- PINN ---
    pinn = PINN(n_features=1, out_dim=out_dim)
    train_pinn(
        pinn,
        X_pinn_fit,
        Y_pos_scaled_fit,
        Y_vel_scaled_fit,
        X_val=X_pinn_val,
        Y_pos_val=Y_pos_scaled_val,
        Y_vel_val=Y_vel_scaled_val,
        lr=args.lr,
        patience=args.patience,
    )

    # Eval PINN on original scale
    pinn_train_pred = pinn.forward(X_pinn_train)
    pinn_train_pred_orig = (
        scaler_pos.inverse_transform(pinn_train_pred) if scaler_pos else pinn_train_pred
    )
    pinn_train_mse, pinn_train_r2 = _eval(
        None, None, Y_pos_raw_train, predict=lambda _: pinn_train_pred_orig
    )

    pinn_test_pred = pinn.forward(X_pinn_test)
    pinn_test_pred_orig = (
        scaler_pos.inverse_transform(pinn_test_pred) if scaler_pos else pinn_test_pred
    )
    pinn_test_mse, pinn_test_r2 = _eval(
        None,
        None,
        Y_pos_raw_test if df_test is None else df_test[pos_cols].values,
        predict=lambda _: pinn_test_pred_orig,
    )

    # --- MLP ---
    mlp = MLP([n_features, 8, 4, out_dim], seed=0)
    mlp.fit(
        X_fit,
        Y_pos_scaled_fit,
        X_val=X_val,
        Y_val=Y_pos_scaled_val,
        patience=args.patience,
        weight_decay=1e-4,
    )

    mlp_train_pred = np.asarray(mlp.forward(X_train))
    mlp_train_pred_orig = (
        scaler_pos.inverse_transform(mlp_train_pred) if scaler_pos else mlp_train_pred
    )
    mlp_train_mse, mlp_train_r2 = _eval(
        None, None, Y_pos_raw_train, predict=lambda _: mlp_train_pred_orig
    )

    mlp_test_pred = np.asarray(mlp.forward(X_test))
    mlp_test_pred_orig = (
        scaler_pos.inverse_transform(mlp_test_pred) if scaler_pos else mlp_test_pred
    )
    mlp_test_target = (
        Y_pos_raw_train[: len(mlp_test_pred_orig)]
        if df_test is None
        else df_test[pos_cols].values
    )
    mlp_test_mse, mlp_test_r2 = _eval(
        None, None, mlp_test_target, predict=lambda _: mlp_test_pred_orig
    )

    # --- SINDy ---
    sindy_deg = 2
    Xi, names, exprs = sindy(X_train, Y_pos_raw_train, degree=sindy_deg)
    sindy_predict_fn = lambda batch: sindy_predict(batch, Xi, degree=sindy_deg)
    sin_train_mse, sin_train_r2 = _eval(
        None, X_train, Y_pos_raw_train, predict=sindy_predict_fn
    )
    sin_test_mse, sin_test_r2 = _eval(
        None,
        X_test,
        Y_pos_raw_train[: len(X_test)] if df_test is None else df_test[pos_cols].values,
        predict=sindy_predict_fn,
    )

    # --- Linear Regression ---
    W_lr = linear_regression(X_train, Y_pos_raw_train)
    lr_predict_fn = lambda batch: linear_predict(batch, W_lr)
    lr_train_mse, lr_train_r2 = _eval(
        None, X_train, Y_pos_raw_train, predict=lr_predict_fn
    )
    lr_test_mse, lr_test_r2 = _eval(
        None,
        X_test,
        Y_pos_raw_train[: len(X_test)] if df_test is None else df_test[pos_cols].values,
        predict=lr_predict_fn,
    )

    results = {
        "n_samples": len(df) if df is not None else 12,
        "masked_fraction": masked_frac,
        "pinn": {
            "train_mse": pinn_train_mse,
            "train_r2": pinn_train_r2,
            "test_mse": pinn_test_mse,
            "test_r2": pinn_test_r2,
        },
        "mlp": {
            "train_mse": mlp_train_mse,
            "train_r2": mlp_train_r2,
            "test_mse": mlp_test_mse,
            "test_r2": mlp_test_r2,
        },
        "sindy": {
            "train_mse": sin_train_mse,
            "train_r2": sin_train_r2,
            "test_mse": sin_test_mse,
            "test_r2": sin_test_r2,
            "expr": exprs[0] if exprs else "",
        },
        "linreg": {
            "train_mse": lr_train_mse,
            "train_r2": lr_train_r2,
            "test_mse": lr_test_mse,
            "test_r2": lr_test_r2,
        },
    }

    out_path = args.out or "ml/pinn/pinn_result.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"PINN  train mse={pinn_train_mse:.6e} r2={pinn_train_r2:.6f} | test mse={pinn_test_mse:.6e} r2={pinn_test_r2:.6f}"
    )
    print(
        f"MLP   train mse={mlp_train_mse:.6e} r2={mlp_train_r2:.6f} | test mse={mlp_test_mse:.6e} r2={mlp_test_r2:.6f}"
    )
    print(
        f"SINDy train mse={sin_train_mse:.6e} r2={sin_train_r2:.6f} | test mse={sin_test_mse:.6e} r2={sin_test_r2:.6f}"
    )
    return results, out_path


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--mission", default=None)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--engineer", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--out", default="ml/pinn/pinn_result.json")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
