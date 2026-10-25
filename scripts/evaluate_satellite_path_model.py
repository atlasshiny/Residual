#!/usr/bin/env python3
"""Evaluate free-running satellite path predictions on held-out arcs and LRO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from train_satellite_path_model import features, score


def predict_step(model, checkpoint, row):
    x = features(pd.DataFrame([row]))
    x = (x - np.asarray(checkpoint["feature_mean"])) / np.asarray(checkpoint["feature_scale"])
    with torch.no_grad():
        y = model(torch.tensor(x, dtype=torch.float32)).numpy()[0]
    y = y * np.asarray(checkpoint["target_scale"]) + np.asarray(checkpoint["target_mean"])
    prefix = "earth" if row["center"] == "earth" else "moon"
    current_center = np.r_[row[[f"{prefix}_{a}_m" for a in "xyz"]].to_numpy(dtype=float),
                           row[[f"{prefix}_v{a}_mps" for a in "xyz"]].to_numpy(dtype=float)]
    next_center = np.r_[row[[f"next_{prefix}_{a}_m" for a in "xyz"]].to_numpy(dtype=float),
                         row[[f"next_{prefix}_v{a}_mps" for a in "xyz"]].to_numpy(dtype=float)]
    relative = np.r_[row[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
                     row[["vx_mps", "vy_mps", "vz_mps"]].to_numpy(dtype=float)] - current_center
    relative += np.r_[y[:3] * 1e6, y[3:] * 1e3]
    return next_center + relative


def rollout(group, model, checkpoint, center, mass, area, cr):
    group = group.sort_values("t_s").reset_index(drop=True)
    pred = group.loc[0, ["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]].to_numpy(dtype=float)
    output = [pred.copy()]
    for i in range(len(group)-1):
        row = group.iloc[i].copy()
        next_row = group.iloc[i+1]
        row["x_m"], row["y_m"], row["z_m"] = pred[:3]
        row["vx_mps"], row["vy_mps"], row["vz_mps"] = pred[3:]
        row["dt_s"] = group.t_s.iloc[i+1] - group.t_s.iloc[i]
        row["center"], row["mass_kg"], row["area_m2"], row["cr"] = center, mass, area, cr
        prefix = "earth" if center == "earth" else "moon"
        for col in [f"{prefix}_{a}_m" for a in "xyz"] + [f"{prefix}_v{a}_mps" for a in "xyz"]:
            row[f"next_{col}"] = next_row[col]
        pred = predict_step(model, checkpoint, row)
        output.append(pred.copy())
    return np.asarray(output)


def constant_velocity_rollout(group, center):
    group = group.sort_values("t_s").reset_index(drop=True)
    state = group.loc[0, ["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]].to_numpy(dtype=float)
    out = [state.copy()]
    prefix = "earth" if center == "earth" else "moon"
    for i in range(len(group)-1):
        cur, nxt = group.iloc[i], group.iloc[i+1]
        c0 = np.r_[cur[[f"{prefix}_{a}_m" for a in "xyz"]].to_numpy(dtype=float),
                   cur[[f"{prefix}_v{a}_mps" for a in "xyz"]].to_numpy(dtype=float)]
        c1 = np.r_[nxt[[f"{prefix}_{a}_m" for a in "xyz"]].to_numpy(dtype=float),
                   nxt[[f"{prefix}_v{a}_mps" for a in "xyz"]].to_numpy(dtype=float)]
        rel = state-c0
        rel[:3] += rel[3:] * (nxt.t_s-cur.t_s)
        state = c1+rel
        out.append(state.copy())
    return np.asarray(out)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    layers = [ckpt["architecture"][0], *ckpt["architecture"][1:-1], ckpt["architecture"][-1]]
    model = nn.Sequential(nn.Linear(layers[0], layers[1]), nn.Tanh(),
                          nn.Linear(layers[1], layers[2]), nn.Tanh(),
                          nn.Linear(layers[2], layers[3]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/synthetic_satellite_trajectories.csv")
    ap.add_argument("--checkpoint", default="ml/pinn/satellite_path_model/satellite_path_model.pt")
    ap.add_argument("--lro", default="data/reference_lro_artemis2_window_preserve.csv")
    ap.add_argument("--moon", default="data/reference_artemis2_preserve.csv")
    ap.add_argument("--sun", default="data/reference_sun_artemis2.csv")
    ap.add_argument("--out", default="ml/pinn/satellite_path_model/evaluation.json")
    args = ap.parse_args()
    model, ckpt = load_model(args.checkpoint)
    data = pd.read_csv(args.data)
    held = data[data.arc_id.isin(ckpt["test_arcs"])]
    rollouts = []
    for arc, group in held.groupby("arc_id"):
        meta = group.iloc[0]
        pred = rollout(group, model, ckpt, meta.center, meta.mass_kg, meta.area_m2, meta.cr)
        baseline = constant_velocity_rollout(group, meta.center)
        truth = group.sort_values("t_s")[["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]].to_numpy()
        rollouts.append({"arc_id": arc, **score(truth, pred),
                         "final_position_error_m": float(np.linalg.norm(pred[-1, :3]-truth[-1, :3])),
                         "constant_velocity_baseline": score(truth, baseline)})
    result = {"synthetic_heldout_free_rollout": {
        "arcs": rollouts,
        "mean_position_rmse_m": float(np.mean([x["position_rmse_m"] for x in rollouts])),
        "mean_velocity_rmse_mps": float(np.mean([x["velocity_rmse_mps"] for x in rollouts])),
    }}

    if all(Path(p).exists() for p in (args.lro, args.moon, args.sun)):
        lro = pd.read_csv(args.lro).rename(columns={"t":"t_s", "x":"x_m", "y":"y_m", "z":"z_m",
                                                      "vx":"vx_mps", "vy":"vy_mps", "vz":"vz_mps"})
        moon = pd.read_csv(args.moon).rename(columns={"t":"t_s", "x":"moon_x_m", "y":"moon_y_m", "z":"moon_z_m"})
        sun = pd.read_csv(args.sun).rename(columns={"t":"t_s", "x":"sun_x_m", "y":"sun_y_m", "z":"sun_z_m"})
        # convert_spacecraft.py already writes LRO into the Earth-Moon
        # barycentric frame, matching the converted Moon table.
        moon_v = pd.read_csv(args.moon)
        moon[["moon_vx_mps", "moon_vy_mps", "moon_vz_mps"]] = moon_v[["vx", "vy", "vz"]].to_numpy()
        lro = lro.merge(moon, on="t_s").merge(sun, on="t_s")
        moon_fraction = 4.9048695e12 / (3.986004418e14 + 4.9048695e12)
        for axis in "xyz":
            lro[f"earth_{axis}_m"] = -moon_fraction * lro[f"moon_{axis}_m"]
            lro[f"earth_v{axis}_mps"] = -moon_fraction * lro[f"moon_v{axis}_mps"]
        lro["center"], lro["mass_kg"], lro["area_m2"], lro["cr"] = "moon", 1000.0, 10.0, 1.3
        lro["t_s"] = lro["t_s"].astype(float)
        lro["dt_s"] = lro.t_s.diff().fillna(600.0)
        pred = rollout(lro, model, ckpt, "moon", 1000.0, 10.0, 1.3)
        baseline = constant_velocity_rollout(lro, "moon")
        truth = lro[["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]].to_numpy()
        result["lro_external_free_rollout"] = {
            **score(truth, pred),
            "final_position_error_m": float(np.linalg.norm(pred[-1, :3]-truth[-1, :3])),
            "constant_velocity_baseline": score(truth, baseline),
            "assumptions": "LRO is converted to the Earth-Moon barycentric frame; 1000 kg, 10 m^2, Cr=1.3 are placeholders because LRO vehicle properties are not provided.",
        }
    else:
        result["lro_external_free_rollout"] = {"skipped": "LRO, Moon, or Sun reference file is missing"}

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
