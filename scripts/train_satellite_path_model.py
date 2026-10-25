#!/usr/bin/env python3
"""Train a grouped, one-step satellite state-transition model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


def features(frame):
    x = []
    for _, row in frame.iterrows():
        values = []
        for body, rscale, vscale in (("earth", 1e7, 1e4), ("moon", 1e6, 1e3)):
            values.extend([(row[f"{axis}_m"]-row[f"{body}_{axis}_m"])/rscale for axis in "xyz"])
            values.extend([(row[f"v{axis}_mps"]-row[f"{body}_v{axis}_mps"])/vscale for axis in "xyz"])
        values.extend([(row[f"sun_{axis}_m"]-row[f"{axis}_m"])/1e11 for axis in "xyz"])
        values.extend([
            float(row.get("dt_s", 600.0))/600.0,
            float(row["center"] == "earth"), float(row["center"] == "moon"),
            float(row.get("mass_kg", 1000.0))/1e4,
            float(row.get("area_m2", 10.0))/100.0,
            float(row.get("cr", 1.3)),
        ])
        x.append(values)
    return np.asarray(x, dtype=np.float32)


def score(actual, prediction):
    error = prediction - actual
    return {
        "samples": int(len(actual)),
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(error[:, :3]**2, axis=1)))),
        "velocity_rmse_mps": float(np.sqrt(np.mean(np.sum(error[:, 3:]**2, axis=1)))),
    }


def center_state(frame, center):
    prefix = "earth" if center == "earth" else "moon"
    return frame[[f"{prefix}_{a}_m" for a in "xyz"]].to_numpy(), frame[[f"{prefix}_v{a}_mps" for a in "xyz"]].to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/synthetic_satellite_trajectories.csv")
    ap.add_argument("--out-dir", default="ml/pinn/satellite_path_model")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = pd.read_csv(args.data)

    pairs = []
    for arc, group in data.groupby("arc_id", sort=True):
        group = group.sort_values("t_s").reset_index(drop=True)
        current = group.iloc[:-1].copy().reset_index(drop=True)
        future = group.iloc[1:].reset_index(drop=True)
        current["dt_s"] = future.t_s.to_numpy() - current.t_s.to_numpy()
        x = features(current)
        center = group.center.iloc[0]
        cpos0, cvel0 = center_state(current, center)
        cpos1, cvel1 = center_state(future, center)
        rel0 = np.column_stack((current[["x_m", "y_m", "z_m"]].to_numpy()-cpos0,
                                current[["vx_mps", "vy_mps", "vz_mps"]].to_numpy()-cvel0))
        rel1 = np.column_stack((future[["x_m", "y_m", "z_m"]].to_numpy()-cpos1,
                                future[["vx_mps", "vy_mps", "vz_mps"]].to_numpy()-cvel1))
        y = np.column_stack(((rel1[:, :3]-rel0[:, :3])/1e6,
                             (rel1[:, 3:]-rel0[:, 3:])/1e3)).astype(np.float32)
        actual = future[["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]].to_numpy()
        pairs.append({"arc": arc, "center": group.center.iloc[0], "band": group.orbit_band.iloc[0],
                      "x": x, "y": y, "current": current,
                      "actual": actual, "future": future})

    # Split whole trajectories within each orbital regime so every regime is
    # represented in training, validation, and test sets.
    rng = np.random.default_rng(args.seed)
    train_arcs, val_arcs, test_arcs = set(), set(), set()
    regimes = sorted({(p["center"], p["band"]) for p in pairs})
    for regime in regimes:
        ids = np.asarray([p["arc"] for p in pairs if (p["center"], p["band"]) == regime], dtype=object)
        rng.shuffle(ids)
        n = len(ids)
        n_test, n_val = max(1, int(round(n*0.25))), max(1, int(round(n*0.125)))
        test_arcs.update(ids[:n_test].tolist())
        val_arcs.update(ids[n_test:n_test+n_val].tolist())
        train_arcs.update(ids[n_test+n_val:].tolist())
    train = [p for p in pairs if p["arc"] in train_arcs]
    val = [p for p in pairs if p["arc"] in val_arcs]
    test = [p for p in pairs if p["arc"] in test_arcs]
    flatten = lambda subset, key: np.concatenate([p[key] for p in subset], axis=0)
    x_train, y_train = flatten(train, "x"), flatten(train, "y")
    x_val, y_val = flatten(val, "x"), flatten(val, "y")
    x_test, y_test = flatten(test, "x"), flatten(test, "y")
    x_mean, x_scale = x_train.mean(axis=0), x_train.std(axis=0)
    y_mean, y_scale = y_train.mean(axis=0), y_train.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_scale[y_scale < 1e-8] = 1.0
    scale_x = lambda x: ((x-x_mean)/x_scale).astype(np.float32)
    scale_y = lambda y: ((y-y_mean)/y_scale).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 6),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
    tx = torch.tensor(scale_x(x_train), device=device)
    ty = torch.tensor(scale_y(y_train), device=device)
    vx = torch.tensor(scale_x(x_val), device=device)
    vy = torch.tensor(scale_y(y_val), device=device)
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(tx), device=device)
        for idx in order.split(2048):
            loss = nn.functional.smooth_l1_loss(model(tx[idx]), ty[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.smooth_l1_loss(model(vx), vy).item()
        if val_loss < best:
            best, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_delta = model(torch.tensor(scale_x(x_test), device=device)).cpu().numpy()*y_scale+y_mean

    actual_test = flatten(test, "actual")
    current_test = pd.concat([p["current"] for p in test], ignore_index=True)
    current_r = current_test[["x_m", "y_m", "z_m"]].to_numpy()
    current_v = current_test[["vx_mps", "vy_mps", "vz_mps"]].to_numpy()
    center_pos0 = np.zeros_like(current_r); center_vel0 = np.zeros_like(current_v)
    # Per-row center is generally constant within each arc; construct mixed
    # Earth/Moon test rows in their own frames.
    center_pos1 = np.zeros_like(center_pos0); center_vel1 = np.zeros_like(center_vel0)
    for center in ("earth", "moon"):
        mask = current_test.center.to_numpy() == center
        if mask.any():
            cp, cv = center_state(current_test.loc[mask], center)
            center_pos0[mask], center_vel0[mask] = cp, cv
            prefix = "earth" if center == "earth" else "moon"
            dt = current_test.loc[mask, "dt_s"].to_numpy() if "dt_s" in current_test else 600.0
            center_pos1[mask] = cp + cv * np.asarray(dt)[:, None]
            center_vel1[mask] = cv
            # Correct the model/baseline with the actual next center state.
            next_states = []
            for p in test:
                if p["center"] == center:
                    nxt = p["future"]
                    next_states.append(np.column_stack((nxt[[f"{prefix}_{a}_m" for a in "xyz"]].to_numpy(),
                                                        nxt[[f"{prefix}_v{a}_mps" for a in "xyz"]].to_numpy())))
            if next_states:
                center_pos1[mask] = np.concatenate(next_states)[:, :3]
                center_vel1[mask] = np.concatenate(next_states)[:, 3:]
    rel0 = np.column_stack((current_r-center_pos0, current_v-center_vel0))
    baseline = np.column_stack((center_pos1+rel0[:, :3]+rel0[:, 3:]*600.0,
                                center_vel1+rel0[:, 3:]))
    model_pred = np.column_stack((center_pos1+rel0[:, :3]+pred_delta[:, :3]*1e6,
                                  center_vel1+rel0[:, 3:]+pred_delta[:, 3:]*1e3))
    metrics = {
        "model": "autoregressive state-transition MLP",
        "seed": args.seed, "device": str(device), "epochs_run": epoch+1,
        "best_validation_huber_scaled": best,
        "train_arcs": sorted(train_arcs), "validation_arcs": sorted(val_arcs), "test_arcs": sorted(test_arcs),
        "one_step_test": {
            "model": score(actual_test, model_pred),
            "constant_velocity_baseline": score(actual_test, baseline),
        },
        "caveat": "One-step scores are not free-rollout accuracy; use evaluate_satellite_path_model.py for external path tests.",
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.cpu().state_dict(), "architecture": [x_train.shape[1],128,128,6],
        "feature_mean": x_mean.tolist(), "feature_scale": x_scale.tolist(),
        "target_mean": y_mean.tolist(), "target_scale": y_scale.tolist(),
        "train_arcs": sorted(train_arcs), "validation_arcs": sorted(val_arcs), "test_arcs": sorted(test_arcs),
        "sample_seconds": 600.0,
    }
    torch.save(checkpoint, out/"satellite_path_model.pt")
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps(metrics, indent=2))
    print(f"saved new path model under {out}")


if __name__ == "__main__":
    main()
