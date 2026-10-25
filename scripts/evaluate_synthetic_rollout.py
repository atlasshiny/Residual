#!/usr/bin/env python3
"""Compare physics-only and learned-correction free rollouts on held-out arcs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from generate_synthetic_rocket_data import G0, MU_EARTH, MU_MOON, MU_SUN
from train_synthetic_dynamics import build_features


def learned_at(r, v, mass, t, table, model, checkpoint, device):
    times = table["t_s"].to_numpy()
    j = min(max(int(np.searchsorted(times, t, side="right") - 1), 0), len(times) - 1)
    row = table.iloc[j]
    mass = float(np.interp(t, times, table["mass_kg"]))
    earth = np.array([np.interp(t, times, table[f"earth_{a}_m"]) for a in "xyz"])
    moon = np.array([np.interp(t, times, table[f"moon_{a}_m"]) for a in "xyz"])
    sun = np.array([np.interp(t, times, table[f"sun_{a}_m"]) for a in "xyz"])
    earth_v = np.array([np.interp(t, times, table[f"earth_v{a}_mps"]) for a in "xyz"])
    moon_v = np.array([np.interp(t, times, table[f"moon_v{a}_mps"]) for a in "xyz"])
    dry_mass = float(row["dry_mass_kg"])
    isp = float(row["isp_s"])
    sample = pd.DataFrame([{
            "x_m": r[0], "y_m": r[1], "z_m": r[2],
            "vx_mps": v[0], "vy_mps": v[1], "vz_mps": v[2],
            "earth_x_m": earth[0], "earth_y_m": earth[1], "earth_z_m": earth[2],
            "earth_vx_mps": earth_v[0], "earth_vy_mps": earth_v[1], "earth_vz_mps": earth_v[2],
            "moon_x_m": moon[0], "moon_y_m": moon[1], "moon_z_m": moon[2],
            "moon_vx_mps": moon_v[0], "moon_vy_mps": moon_v[1], "moon_vz_mps": moon_v[2],
            "sun_x_m": sun[0], "sun_y_m": sun[1], "sun_z_m": sun[2],
            "mass_kg": mass, "dry_mass_kg": dry_mass,
            "area_m2": float(row["area_m2"]), "cr": float(row["cr"]),
            "isp_s": isp, "vehicle_family": int(row["vehicle_family"]),
            "target_is_moon": 0,
    }])
    features, _ = build_features(sample)
    z = (features - np.asarray(checkpoint["feature_mean"])) / np.asarray(checkpoint["feature_scale"])
    with torch.no_grad():
        learned = model(torch.tensor(z, dtype=torch.float32, device=device)).cpu().numpy()[0]
    learned = learned * np.asarray(checkpoint["target_scale"]) + np.asarray(checkpoint["target_mean"])
    return learned * checkpoint.get("learned_residual_output_gain", 1.0)


def force_row(r, v, mass, t, table, correction=None):
    times = table["t_s"].to_numpy()
    j = min(max(int(np.searchsorted(times, t, side="right") - 1), 0), len(times) - 1)
    row = table.iloc[j]
    mass = float(np.interp(t, times, table["mass_kg"]))
    earth = np.array([np.interp(t, times, table[f"earth_{a}_m"]) for a in "xyz"])
    moon = np.array([np.interp(t, times, table[f"moon_{a}_m"]) for a in "xyz"])
    sun = np.array([np.interp(t, times, table[f"sun_{a}_m"]) for a in "xyz"])
    re, rm, rs = r - earth, r - moon, r - sun
    a = (
        -MU_EARTH * re / np.linalg.norm(re) ** 3
        - MU_MOON * rm / np.linalg.norm(rm) ** 3
        - MU_SUN * (rs / np.linalg.norm(rs) ** 3 + sun / np.linalg.norm(sun) ** 3)
    )
    earth_v = np.array([np.interp(t, times, table[f"earth_v{a}_mps"]) for a in "xyz"])
    mass = float(np.interp(t, times, table["mass_kg"]))
    control = np.zeros(3)
    isp = 0.0
    for burn in table.attrs["burns"]:
        if burn["start"] <= t < burn["stop"] and mass > float(row["dry_mass_kg"]):
            if burn.get("guidance") == "prograde":
                prograde = v - earth_v
                prograde /= np.linalg.norm(prograde)
                normal = np.cross(r - earth, prograde)
                normal /= np.linalg.norm(normal)
                side = np.cross(normal, prograde)
                side /= np.linalg.norm(side)
                direction = prograde + burn["side_angle"] * side + burn["normal_angle"] * normal
                direction /= np.linalg.norm(direction)
            else:
                direction = np.asarray(burn["direction"])
            control += float(burn["force_n"]) * direction
            isp = float(burn["isp_s"])
    mass_dot = 0.0  # Hold the measured synthetic mass history fixed in this rollout comparison.
    dry_mass = float(row["dry_mass_kg"])
    if mass <= dry_mass:
        control[:] = 0.0
        mass_dot = 0.0
    a += control / max(mass, 1.0)
    if correction is not None:
        a += correction
    return np.concatenate((v, a, [mass_dot]))


def rollout(table, burns, use_network, model, checkpoint, device):
    table = table.sort_values("t_s").reset_index(drop=True)
    table.attrs["burns"] = burns
    times = table["t_s"].to_numpy()
    dt = float(np.median(np.diff(times)))
    state = np.concatenate((table.loc[0, ["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]].to_numpy(dtype=float), [float(table.loc[0, "mass_kg"])]))
    states = [state.copy()]
    for i in range(len(times) - 1):
        t = times[i]
        state[6] = float(table.loc[i, "mass_kg"])
        # Reuse exact burn timing and prograde steering; use three RK4
        # substeps per 60-second output interval (20-second force cadence).
        n_sub = 3
        h = dt / n_sub
        for k in range(n_sub):
            tk = t + k * h
            mass_now = float(np.interp(tk, times, table["mass_kg"]))
            correction = learned_at(state[:3], state[3:6], mass_now, tk, table, model, checkpoint, device) if use_network and k == 0 else (correction if use_network else None)
            k1 = force_row(state[:3], state[3:6], state[6], tk, table, correction)
            k2 = force_row(state[:3] + 0.5*h*k1[:3], state[3:6] + 0.5*h*k1[3:6], state[6] + 0.5*h*k1[6], tk+0.5*h, table, correction)
            k3 = force_row(state[:3] + 0.5*h*k2[:3], state[3:6] + 0.5*h*k2[3:6], state[6] + 0.5*h*k2[6], tk+0.5*h, table, correction)
            k4 = force_row(state[:3] + h*k3[:3], state[3:6] + h*k3[3:6], state[6] + h*k3[6], tk+h, table, correction)
            state += h / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
        state[6] = float(table.loc[i + 1, "mass_kg"])
        states.append(state.copy())
    return np.asarray(states)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/synthetic_rocket_transfer.csv")
    ap.add_argument("--metrics", default="ml/pinn/synthetic_transfer/metrics.json")
    ap.add_argument("--model", default="ml/pinn/synthetic_transfer/synthetic_state_acceleration_model.pt")
    ap.add_argument("--out-dir", default="ml/pinn/synthetic_transfer/rollouts")
    args = ap.parse_args()
    metrics = json.loads(Path(args.metrics).read_text())
    manifest = json.loads(Path("data/synthetic_rocket_transfer.manifest.json").read_text())
    burn_schedules = {arc["arc_id"]: arc["burns"] for arc in manifest["trajectories"]}
    test_arcs = metrics["arc_split"]["test"]
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    width, h1, h2, out_width = checkpoint["architecture"]
    model = nn.Sequential(nn.Linear(width, h1), nn.Tanh(), nn.Linear(h1, h2), nn.Tanh(), nn.Linear(h2, out_width))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    device = torch.device("cpu")
    frame = pd.read_csv(args.data)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for arc in test_arcs:
        truth = frame[frame.arc_id == arc].sort_values("t_s").reset_index(drop=True)
        burns = burn_schedules[arc]
        physics = rollout(truth, burns, False, model, checkpoint, device)
        corrected = rollout(truth, burns, True, model, checkpoint, device)
        truth_r = truth[["x_m", "y_m", "z_m"]].to_numpy()
        truth_v = truth[["vx_mps", "vy_mps", "vz_mps"]].to_numpy()
        physics_rms = float(np.sqrt(np.mean(np.sum((physics[:, :3] - truth_r) ** 2, axis=1))))
        corrected_rms = float(np.sqrt(np.mean(np.sum((corrected[:, :3] - truth_r) ** 2, axis=1))))
        physics_final = float(np.linalg.norm(physics[-1, :3] - truth_r[-1]))
        corrected_final = float(np.linalg.norm(corrected[-1, :3] - truth_r[-1]))
        results[arc] = {
            "samples": len(truth), "physics_position_rms_m": physics_rms,
            "corrected_position_rms_m": corrected_rms,
            "physics_final_position_error_m": physics_final,
            "corrected_final_position_error_m": corrected_final,
            "rms_improvement_fraction": 1.0 - corrected_rms / physics_rms if physics_rms else 0.0,
        }
        pd.DataFrame({
            "t_s": truth.t_s,
            "truth_x_m": truth.x_m, "truth_y_m": truth.y_m, "truth_z_m": truth.z_m,
            "physics_x_m": physics[:, 0], "physics_y_m": physics[:, 1], "physics_z_m": physics[:, 2],
            "corrected_x_m": corrected[:, 0], "corrected_y_m": corrected[:, 1], "corrected_z_m": corrected[:, 2],
        }).to_csv(output_dir / f"{arc}_rollout.csv", index=False)
    report = {"test_arcs": test_arcs, "rollouts": results,
              "note": "Held-out rollouts use exact synthetic burn schedules, measured mass history, and 20-second RK4 substeps."}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
