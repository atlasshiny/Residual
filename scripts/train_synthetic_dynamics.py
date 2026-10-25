#!/usr/bin/env python3
"""Train a new state-conditioned acceleration model on synthetic rocket arcs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


def build_features(frame: pd.DataFrame):
    columns = {}
    for body, scale_r, scale_v in (("earth", 1e7, 1e4), ("moon", 1e8, 1e4)):
        for axis in "xyz":
            columns[f"r_{body}_{axis}"] = (
                frame[f"{axis}_m"].to_numpy() - frame[f"{body}_{axis}_m"].to_numpy()
            ) / scale_r
            columns[f"v_{body}_{axis}"] = (
                frame[f"v{axis}_mps"].to_numpy() - frame[f"{body}_v{axis}_mps"].to_numpy()
            ) / scale_v
    for axis in "xyz":
        columns[f"r_sun_{axis}"] = (
            frame[f"sun_{axis}_m"].to_numpy() - frame[f"{axis}_m"].to_numpy()
        ) / 1e11
    mass = frame["mass_kg"].to_numpy()
    dry_mass = frame["dry_mass_kg"].to_numpy()
    columns["mass_1e5kg"] = mass / 1e5
    columns["propellant_fraction"] = np.divide(
        np.maximum(mass - dry_mass, 0.0), mass,
        out=np.zeros_like(mass, dtype=float), where=mass > 0,
    )
    columns["area_100m2"] = frame["area_m2"].to_numpy() / 100.0
    columns["cr"] = frame["cr"].to_numpy()
    columns["isp_500s"] = frame["isp_s"].to_numpy() / 500.0
    columns["target_is_moon"] = (
        frame.get("target_is_moon", pd.Series(0, index=frame.index))
        .fillna(0).to_numpy()
    )
    for family in range(5):
        columns[f"vehicle_family_{family}"] = (
            frame["vehicle_family"].to_numpy() == family
        ).astype(float)
    names = list(columns)
    return np.column_stack([columns[name] for name in names]).astype(np.float32), names


def metrics_for(y, pred, mask):
    if not np.any(mask):
        return {"samples": 0, "rmse_mps2": None, "zero_baseline_rmse_mps2": None, "r2": None}
    actual, estimate = y[mask], pred[mask]
    rmse = float(np.sqrt(np.mean((actual - estimate) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(actual**2)))
    denom = float(np.sum((actual - actual.mean(axis=0)) ** 2))
    r2 = float(1.0 - np.sum((actual - estimate) ** 2) / denom) if denom > 0 else None
    return {
        "samples": int(np.sum(mask)),
        "rmse_mps2": rmse,
        "zero_baseline_rmse_mps2": zero_rmse,
        "r2": r2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic_rocket_transfer.csv")
    parser.add_argument("--moon-data", help="optional non-Artemis Moon residual rows; always added to training")
    parser.add_argument("--out-dir", default="ml/pinn/synthetic_transfer")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--burn-weight", type=float, default=5.0)
    parser.add_argument("--moon-weight", type=float, default=50.0)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    synthetic = pd.read_csv(args.data)
    # Apply known thrust as F/m. The network learns the remaining dynamics
    # residual (J2 + solar radiation pressure) from state and vehicle inputs.
    arc_ids = sorted(synthetic["arc_id"].unique().tolist())
    if len(arc_ids) < 6:
        raise ValueError("need at least six distinct arcs for train/validation/test splits")
    rng = np.random.default_rng(args.seed)
    shuffled = np.asarray(arc_ids, dtype=object)
    rng.shuffle(shuffled)
    n_test = max(1, round(len(arc_ids) * 0.17))
    n_val = max(1, round(len(arc_ids) * 0.17))
    test_arcs = set(shuffled[:n_test].tolist())
    val_arcs = set(shuffled[n_test:n_test + n_val].tolist())
    train_arcs = set(shuffled[n_test + n_val:].tolist())
    arc_values = synthetic["arc_id"].to_numpy()
    synthetic_train = np.isin(arc_values, list(train_arcs))
    synthetic_val = np.isin(arc_values, list(val_arcs))
    synthetic_test = np.isin(arc_values, list(test_arcs))
    moon_train = pd.DataFrame(columns=synthetic.columns)
    moon_val = pd.DataFrame(columns=synthetic.columns)
    if args.moon_data:
        moon_frame = pd.read_csv(args.moon_data).sort_values("t_s").reset_index(drop=True)
        moon_cut = int(len(moon_frame) * 0.8)
        moon_train = moon_frame.iloc[:moon_cut]
        moon_val = moon_frame.iloc[moon_cut:]
        frame = pd.concat([synthetic, moon_train, moon_val], ignore_index=True)
    else:
        frame = synthetic.copy()
    x_all, feature_names = build_features(frame)
    y_full = frame[["residual_ax_mps2", "residual_ay_mps2", "residual_az_mps2"]].to_numpy(dtype=np.float32)
    thrust_accel = frame[["thrust_n_x", "thrust_n_y", "thrust_n_z"]].to_numpy(dtype=np.float32)
    thrust_accel /= np.maximum(frame["mass_kg"].to_numpy(dtype=np.float32)[:, None], 1.0)
    y_all = y_full - thrust_accel
    active = frame["throttle"].to_numpy(dtype=bool)
    train_mask = np.concatenate((
        synthetic_train, np.ones(len(moon_train), dtype=bool),
        np.zeros(len(moon_val), dtype=bool),
    ))
    val_mask = np.concatenate((
        synthetic_val, np.zeros(len(moon_train), dtype=bool),
        np.ones(len(moon_val), dtype=bool),
    ))
    test_mask = np.concatenate((
        synthetic_test, np.zeros(len(moon_train) + len(moon_val), dtype=bool),
    ))
    arc_values = frame["arc_id"].to_numpy()

    x_mean = x_all[train_mask].mean(axis=0)
    x_scale = x_all[train_mask].std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = y_all[train_mask].mean(axis=0)
    y_scale = y_all[train_mask].std(axis=0)
    y_scale[y_scale < 1e-12] = 1.0
    x = (x_all - x_mean) / x_scale
    y = (y_all - y_mean) / y_scale

    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    val_weights = np.where(active[val_idx], args.burn_weight, 1.0).astype(np.float32)
    is_moon = frame.get("target_is_moon", pd.Series(0, index=frame.index)).fillna(0).to_numpy(dtype=bool)
    val_weights = np.where(is_moon[val_idx], args.moon_weight, val_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(x.shape[1], 96), nn.Tanh(),
        nn.Linear(96, 64), nn.Tanh(),
        nn.Linear(64, 3),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    best_val = float("inf")
    best_state = None
    stale = 0
    epochs_run = 0
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(train_idx)
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            xb, yb = x_t[idx], y_t[idx]
            batch_is_moon = torch.tensor(is_moon[idx], dtype=torch.bool, device=device)
            batch_is_burn = torch.tensor(active[idx], dtype=torch.bool, device=device)
            wb = torch.where(
                batch_is_moon,
                torch.tensor(args.moon_weight, device=device),
                torch.where(
                    batch_is_burn,
                    torch.tensor(args.burn_weight, device=device),
                    torch.tensor(1.0, device=device),
                ),
            )
            per_row = nn.functional.smooth_l1_loss(model(xb), yb, reduction="none").mean(dim=1)
            loss = torch.sum(per_row * wb) / torch.sum(wb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(x_t[val_idx])
            val_per_row = nn.functional.smooth_l1_loss(
                val_pred, y_t[val_idx], reduction="none"
            ).mean(dim=1)
            val_weight_t = torch.tensor(val_weights, dtype=torch.float32, device=device)
            val_loss = (torch.sum(val_per_row * val_weight_t) / torch.sum(val_weight_t)).item()
        epochs_run = epoch + 1
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_t).cpu().numpy()
    learned_residual = pred_scaled * y_scale + y_mean
    calibration_mask = val_mask & ~is_moon
    calibration_den = float(np.sum(learned_residual[calibration_mask] ** 2))
    output_gain = (
        float(np.sum(y_all[calibration_mask] * learned_residual[calibration_mask]) / calibration_den)
        if calibration_den > 0 else 0.0
    )
    output_gain = float(np.clip(output_gain, 0.0, 1.5))
    pred = output_gain * learned_residual + thrust_accel

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "architecture": [x.shape[1], 96, 64, 3],
        "feature_names": feature_names,
        "feature_mean": x_mean.tolist(), "feature_scale": x_scale.tolist(),
        "target_mean": y_mean.tolist(), "target_scale": y_scale.tolist(),
        "learned_residual_output_gain": output_gain,
        "training_arcs": sorted(train_arcs) + (["moon_training_2025_early"] if args.moon_data else []),
        "validation_arcs": sorted(val_arcs) + (["moon_training_2025_tail"] if args.moon_data else []),
        "test_arcs": sorted(test_arcs),
        "training_vehicle_families": sorted(
            int(value) for value in frame.loc[train_mask, "vehicle_family"].unique()
        ),
        "moon_residual_norm_p99_mps2": float(
            np.quantile(np.linalg.norm(y_all[train_mask & is_moon], axis=1), 0.99)
        ) if np.any(train_mask & is_moon) else None,
        "target": "J2 + solar radiation pressure residual after explicit thrust F/m",
        "explicit_thrust_model": "thrust acceleration = supplied thrust vector / current mass",
    }
    torch.save(checkpoint, output / "synthetic_state_acceleration_model.pt")
    metrics = {
        "model": "state-conditioned acceleration residual MLP",
        "data": args.data, "moon_training_data": args.moon_data,
        "seed": args.seed, "device": str(device),
        "epochs_run": epochs_run, "best_validation_huber_scaled": best_val,
        "burn_weight": args.burn_weight,
        "moon_weight": args.moon_weight,
        "learned_residual_output_gain": output_gain,
        "arc_split": {
            "train": sorted(train_arcs) + (["moon_training_2025_early"] if args.moon_data else []),
            "validation": sorted(val_arcs) + (["moon_training_2025_tail"] if args.moon_data else []),
            "test": sorted(test_arcs),
        },
        "validation": {
            "all": metrics_for(y_full, pred, val_mask),
            "moon_tail": metrics_for(
                y_full, pred,
                val_mask & frame.get("target_is_moon", pd.Series(0, index=frame.index)).fillna(0).to_numpy(dtype=bool),
            ),
            "synthetic_arcs": metrics_for(y_full, pred, val_mask & ~is_moon),
        },
        "test": {
            "full_correction": {
                "all": metrics_for(y_full, pred, test_mask),
                "coast": metrics_for(y_full, pred, test_mask & ~active),
                "burn": metrics_for(y_full, pred, test_mask & active),
            },
            "learned_non_thrust_residual": {
                "all": metrics_for(y_all, output_gain * learned_residual, test_mask),
                "coast": metrics_for(y_all, output_gain * learned_residual, test_mask & ~active),
                "burn": metrics_for(y_all, output_gain * learned_residual, test_mask & active),
            },
        },
        "caveat": "Synthetic-only diagnostic. Artemis missions were not used for fitting or model selection.",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    test_rows = frame.loc[test_mask, ["arc_id", "t_s", "throttle"]].copy()
    test_rows["actual_ax"] = y_full[test_mask, 0]
    test_rows["actual_ay"] = y_full[test_mask, 1]
    test_rows["actual_az"] = y_full[test_mask, 2]
    test_rows["predicted_ax"] = pred[test_mask, 0]
    test_rows["predicted_ay"] = pred[test_mask, 1]
    test_rows["predicted_az"] = pred[test_mask, 2]
    test_rows.to_csv(output / "heldout_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"wrote new checkpoint under {output}; existing checkpoints were not opened for writing")


if __name__ == "__main__":
    main()
