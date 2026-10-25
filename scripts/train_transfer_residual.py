#!/usr/bin/env python3
"""Train a state-conditioned acceleration residual model for transfer tests.

Unlike the production PINN, this experiment receives physical state and body
geometry rather than mission time. Training and test arcs are specified
individually so evaluation can hold out complete trajectories.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

MU_EARTH = 3.986004418e14
MU_MOON = 4.9048695e12
MU_SUN = 1.32712440018e20

FEATURES = [
    *(f"r_{axis}_1e8m" for axis in "xyz"),
    *(f"v_{axis}_1e3mps" for axis in "xyz"),
    *(f"r_earth_{axis}_1e8m" for axis in "xyz"),
    *(f"v_earth_{axis}_1e3mps" for axis in "xyz"),
    *(f"r_moon_{axis}_1e8m" for axis in "xyz"),
    *(f"v_moon_{axis}_1e3mps" for axis in "xyz"),
    *(f"r_sun_{axis}_1e11m" for axis in "xyz"),
    *(f"sun_tide_{axis}_1e-5mps2" for axis in "xyz"),
]


def load_arc(name: str, reference_path: str, moon_path: str, sun_path: str, residual_path: str):
    ref = pd.read_csv(reference_path)
    moon = pd.read_csv(moon_path)
    sun = pd.read_csv(sun_path)
    residual = pd.read_csv(residual_path)
    merged = ref.merge(residual, on="t", suffixes=("", "_res"), validate="one_to_one")
    if len(merged) != len(ref) or len(merged) < 5:
        raise ValueError(f"{name}: reference and residual timestamps do not align")

    t = merged["t"].to_numpy(dtype=float)
    moon_t = moon["t"].to_numpy(dtype=float)
    sun_t = sun["t"].to_numpy(dtype=float)
    if t[0] < moon_t[0] or t[-1] > moon_t[-1] or t[0] < sun_t[0] or t[-1] > sun_t[-1]:
        raise ValueError(f"{name}: Moon/Sun ephemeris does not cover the reference arc")

    r = merged[["x", "y", "z"]].to_numpy(dtype=float)
    v = merged[["vx", "vy", "vz"]].to_numpy(dtype=float)
    dr = merged[["drx", "dry", "drz"]].to_numpy(dtype=float)
    dv = merged[["dvx", "dvy", "dvz"]].to_numpy(dtype=float)
    moon_r = np.column_stack(
        [np.interp(t, moon_t, moon[axis].to_numpy(dtype=float)) for axis in "xyz"]
    )
    moon_v = np.column_stack(
        [np.interp(t, moon_t, moon[f"v{axis}"].to_numpy(dtype=float)) for axis in "xyz"]
    )
    sun_r = np.column_stack(
        [np.interp(t, sun_t, sun[axis].to_numpy(dtype=float)) for axis in "xyz"]
    )
    # The converted Moon track is barycentric; recover Earth by mass ratio.
    earth_r = -MU_MOON / MU_EARTH * moon_r
    earth_v = -MU_MOON / MU_EARTH * moon_v
    r_earth = r - earth_r
    v_earth = v - earth_v
    r_moon = r - moon_r
    v_moon = v - moon_v
    sun_vec = sun_r - r
    sun_dist = np.linalg.norm(sun_vec, axis=1)
    origin_dist = np.linalg.norm(sun_r, axis=1)
    sun_tide = MU_SUN * (
        sun_vec / sun_dist[:, None] ** 3 - sun_r / origin_dist[:, None] ** 3
    )
    features = np.column_stack(
        [r / 1e8, v / 1e3, r_earth / 1e8, v_earth / 1e3,
         r_moon / 1e8, v_moon / 1e3, sun_vec / 1e11, sun_tide / 1e-5]
    )
    # Residual is reference minus propagated state; d(dv)/dt is the residual
    # acceleration target. Drop one-sided derivative endpoints.
    accel = np.gradient(dv, t, axis=0, edge_order=2)
    return {
        "name": name,
        "t": t[1:-1],
        "features": features[1:-1],
        "accel": accel[1:-1],
        "dr": dr[1:-1],
    }


def vector_r2(actual, predicted):
    denom = float(np.sum((actual - actual.mean(axis=0)) ** 2))
    return float(1.0 - np.sum((actual - predicted) ** 2) / denom) if denom else 0.0


def parse_arc(value: list[str]):
    name, reference, moon, sun, residual = value
    return name, reference, moon, sun, residual


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-arc", nargs=5, action="append", required=True,
                    metavar=("NAME", "REFERENCE", "MOON", "SUN", "RESIDUAL"))
    ap.add_argument("--test-arc", nargs=5, action="append", required=True,
                    metavar=("NAME", "REFERENCE", "MOON", "SUN", "RESIDUAL"))
    ap.add_argument("--out-dir", default="ml/pinn/state_transfer")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_arcs = [load_arc(*parse_arc(item)) for item in args.train_arc]
    test_arcs = [load_arc(*parse_arc(item)) for item in args.test_arc]

    # Hold out the final fifth of every training trajectory for early stopping.
    train_x, train_y, val_x, val_y = [], [], [], []
    for arc in train_arcs:
        split = int(len(arc["t"]) * 0.8)
        train_x.append(arc["features"][:split])
        train_y.append(arc["accel"][:split])
        val_x.append(arc["features"][split:])
        val_y.append(arc["accel"][split:])
    x_train_raw = np.concatenate(train_x)
    y_train_raw = np.concatenate(train_y)
    x_val_raw = np.concatenate(val_x)
    y_val_raw = np.concatenate(val_y)

    x_mean = x_train_raw.mean(axis=0)
    x_scale = x_train_raw.std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    y_mean = y_train_raw.mean(axis=0)
    y_scale = y_train_raw.std(axis=0)
    y_scale[y_scale < 1e-12] = 1.0

    scale_x = lambda x: ((x - x_mean) / x_scale).astype(np.float32)
    scale_y = lambda y: ((y - y_mean) / y_scale).astype(np.float32)
    x_train, y_train = scale_x(x_train_raw), scale_y(y_train_raw)
    x_val, y_val = scale_x(x_val_raw), scale_y(y_val_raw)

    model = nn.Sequential(
        nn.Linear(len(FEATURES), 64), nn.Tanh(),
        nn.Linear(64, 32), nn.Tanh(),
        nn.Linear(32, 3),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_train_t = torch.tensor(x_train)
    y_train_t = torch.tensor(y_train)
    x_val_t = torch.tensor(x_val)
    y_val_t = torch.tensor(y_val)
    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.smooth_l1_loss(model(x_train_t), y_train_t, beta=1.0)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.smooth_l1_loss(model(x_val_t), y_val_t, beta=1.0)
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": FEATURES,
            "feature_mean": x_mean.tolist(),
            "feature_scale": x_scale.tolist(),
            "accel_mean": y_mean.tolist(),
            "accel_scale": y_scale.tolist(),
            "training_arcs": [arc["name"] for arc in train_arcs],
        },
        out_dir / "state_acceleration_model.pt",
    )

    metrics = {
        "model": "state-conditioned acceleration residual MLP",
        "train_arcs": {arc["name"]: len(arc["t"]) for arc in train_arcs},
        "test_arcs": {},
        "best_validation_huber_scaled": best_loss,
        "epochs_run": epoch + 1,
        "seed": args.seed,
    }
    for arc in test_arcs:
        x = scale_x(arc["features"])
        with torch.no_grad():
            pred = model(torch.tensor(x)).numpy() * y_scale + y_mean
        actual = arc["accel"]
        metrics["test_arcs"][arc["name"]] = {
            "samples": len(arc["t"]),
            "acceleration_r2": vector_r2(actual, pred),
            "acceleration_rmse_mps2": float(np.sqrt(np.mean((actual - pred) ** 2))),
            "zero_baseline_r2": vector_r2(actual, np.zeros_like(actual)),
            "out_of_training_feature_fraction_gt_3std": float(
                np.mean(np.any(np.abs(x) > 3.0, axis=1))
            ),
        }
        pd.DataFrame(
            {
                "t": arc["t"],
                "actual_ax": actual[:, 0], "actual_ay": actual[:, 1], "actual_az": actual[:, 2],
                "predicted_ax": pred[:, 0], "predicted_ay": pred[:, 1], "predicted_az": pred[:, 2],
            }
        ).to_csv(out_dir / f"{arc['name']}_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir / 'state_acceleration_model.pt'} and test predictions")


if __name__ == "__main__":
    main()
