#!/usr/bin/env python3
"""Fit a satellite acceleration correction on complete synthetic orbit arcs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from train_satellite_path_model import features


def vector_rmse(actual, predicted):
    return float(np.sqrt(np.mean(np.sum((predicted-actual)**2, axis=1))))


def r2(actual, predicted):
    ss_res = np.sum((actual-predicted)**2, axis=0)
    ss_tot = np.sum((actual-actual.mean(axis=0))**2, axis=0)
    vals = np.where(ss_tot > 1e-24, 1.0-ss_res/ss_tot, np.nan)
    return [float(v) for v in vals]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/synthetic_satellite_trajectories.csv")
    ap.add_argument("--out-dir", default="ml/pinn/satellite_dynamics_model")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--ensemble-size", type=int, default=3)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data = pd.read_csv(args.data)
    required = ["residual_ax_mps2", "residual_ay_mps2", "residual_az_mps2"]
    if not set(required).issubset(data.columns):
        raise ValueError("data is missing acceleration residual labels; regenerate with generate_synthetic_satellite_data.py")
    samples = []
    for arc, group in data.groupby("arc_id", sort=True):
        group = group.sort_values("t_s").reset_index(drop=True)
        x = features(group)
        y = group[required].to_numpy(dtype=np.float32)
        samples.append({"arc": arc, "regime": (group.center.iloc[0], group.orbit_band.iloc[0]), "x": x, "y": y})

    rng = np.random.default_rng(args.seed)
    train_arcs, val_arcs, test_arcs = set(), set(), set()
    for regime in sorted({s["regime"] for s in samples}):
        ids = np.asarray([s["arc"] for s in samples if s["regime"] == regime], dtype=object)
        if len(ids) < 3:
            raise ValueError(f"need at least 3 complete arcs per regime for grouped train/validation/test; got {len(ids)} for {regime}")
        rng.shuffle(ids)
        n_test = max(1, round(len(ids)*0.2))
        n_val = max(1, round(len(ids)*0.1))
        test_arcs.update(ids[:n_test].tolist())
        val_arcs.update(ids[n_test:n_test+n_val].tolist())
        train_arcs.update(ids[n_test+n_val:].tolist())
    train = [s for s in samples if s["arc"] in train_arcs]
    val = [s for s in samples if s["arc"] in val_arcs]
    test = [s for s in samples if s["arc"] in test_arcs]
    cat = lambda rows, key: np.concatenate([s[key] for s in rows], axis=0)
    x_train, y_train = cat(train, "x"), cat(train, "y")
    x_val, y_val = cat(val, "x"), cat(val, "y")
    x_test, y_test = cat(test, "x"), cat(test, "y")
    x_mean, x_scale = x_train.mean(0), x_train.std(0)
    y_mean, y_scale = y_train.mean(0), y_train.std(0)
    x_scale[x_scale < 1e-8] = 1.0
    y_scale[y_scale < 1e-14] = 1.0
    sx = lambda v: ((v-x_mean)/x_scale).astype(np.float32)
    sy = lambda v: ((v-y_mean)/y_scale).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tx, ty = torch.tensor(sx(x_train), device=device), torch.tensor(sy(y_train), device=device)
    vx, vy = torch.tensor(sx(x_val), device=device), torch.tensor(sy(y_val), device=device)
    members, member_predictions, val_predictions, best_losses, epochs_run = [], [], [], [], []
    for member_idx in range(args.ensemble_size):
        torch.manual_seed(args.seed + member_idx)
        model = nn.Sequential(nn.Linear(x_train.shape[1], 128), nn.Tanh(),
                              nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 3)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(args.epochs):
            model.train()
            for idx in torch.randperm(len(tx), device=device).split(2048):
                loss = nn.functional.mse_loss(model(tx[idx]), ty[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = nn.functional.mse_loss(model(vx), vy).item()
            if val_loss < best:
                best, stale = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience: break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            member_pred = model(torch.tensor(sx(x_test), device=device)).cpu().numpy()*y_scale+y_mean
            val_pred = model(vx).cpu().numpy()*y_scale+y_mean
        members.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        member_predictions.append(member_pred)
        val_predictions.append(val_pred)
        best_losses.append(best); epochs_run.append(epoch+1)
    member_predictions = np.stack(member_predictions)
    pred = member_predictions.mean(axis=0)
    disagreement = float(np.sqrt(np.mean(np.var(member_predictions, axis=0))))
    val_disagreement = np.sqrt(np.mean(np.var(np.stack(val_predictions), axis=0), axis=1))
    uncertainty_threshold = float(np.quantile(val_disagreement, 0.95))
    train_max_z = np.max(np.abs(sx(x_train)), axis=1)
    ood_threshold = float(np.quantile(train_max_z, 0.995))
    metrics = {
        "model": "state-conditioned J2/SRP acceleration correction; point gravity and thrust remain explicit",
        "epochs_run_per_member": epochs_run, "device": str(device),
        "best_validation_mse_scaled_per_member": best_losses, "ensemble_size": args.ensemble_size,
        "uncertainty_gate_threshold_mps2": uncertainty_threshold,
        "feature_ood_gate_threshold_z": ood_threshold,
        "train_arcs": sorted(train_arcs), "validation_arcs": sorted(val_arcs), "test_arcs": sorted(test_arcs),
        "heldout_acceleration": {
            "model_vector_rmse_mps2": vector_rmse(y_test, pred),
            "zero_correction_vector_rmse_mps2": vector_rmse(y_test, np.zeros_like(y_test)),
            "model_r2_xyz": r2(y_test, pred), "zero_correction_r2_xyz": r2(y_test, np.zeros_like(y_test)),
            "ensemble_disagreement_rms_mps2": disagreement,
        },
        "split_unit": "complete trajectory arc, stratified by center and orbit band",
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "members": members, "architecture": [x_train.shape[1], 128, 128, 3],
        "uncertainty_gate_threshold_mps2": uncertainty_threshold,
        "feature_ood_gate_threshold_z": ood_threshold,
        "feature_mean": x_mean.tolist(), "feature_scale": x_scale.tolist(),
        "target_mean": y_mean.tolist(), "target_scale": y_scale.tolist(),
        "train_arcs": sorted(train_arcs), "validation_arcs": sorted(val_arcs), "test_arcs": sorted(test_arcs),
        "target": required,
    }
    torch.save(checkpoint, out/"satellite_accel_model.pt")
    (out/"accel_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps(metrics, indent=2)); print(f"saved acceleration model under {out}")


if __name__ == "__main__":
    main()
