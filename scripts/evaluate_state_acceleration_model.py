#!/usr/bin/env python3
"""Evaluate a saved acceleration model on held-out ephemeris residual rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from train_synthetic_dynamics import build_features, metrics_for


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ml/pinn/synthetic_transfer/synthetic_state_acceleration_model.pt")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out-dir", default="ml/pinn/synthetic_transfer/external_evaluation")
    ap.add_argument("--ood-threshold", type=float, default=0.2)
    args = ap.parse_args()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    width, hidden1, hidden2, out_width = checkpoint["architecture"]
    model = nn.Sequential(
        nn.Linear(width, hidden1), nn.Tanh(),
        nn.Linear(hidden1, hidden2), nn.Tanh(),
        nn.Linear(hidden2, out_width),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for source in args.data:
        frame = pd.read_csv(source)
        x, names = build_features(frame)
        if names != checkpoint["feature_names"]:
            raise ValueError(f"feature schema mismatch for {source}")
        mean = np.asarray(checkpoint["feature_mean"])
        scale = np.asarray(checkpoint["feature_scale"])
        y_mean = np.asarray(checkpoint["target_mean"])
        y_scale = np.asarray(checkpoint["target_scale"])
        z = (x - mean) / scale
        with torch.no_grad():
            learned = model(torch.tensor(z, dtype=torch.float32)).numpy() * y_scale + y_mean
        learned *= float(checkpoint.get("learned_residual_output_gain", 1.0))
        thrust = frame[["thrust_n_x", "thrust_n_y", "thrust_n_z"]].to_numpy(dtype=float)
        mass = frame["mass_kg"].to_numpy(dtype=float)
        thrust_acc = thrust / np.maximum(mass[:, None], 1.0)
        pred = learned + thrust_acc
        actual = frame[["residual_ax_mps2", "residual_ay_mps2", "residual_az_mps2"]].to_numpy(dtype=float)
        active = frame["throttle"].to_numpy(dtype=bool)
        is_moon = bool(frame.get("target_is_moon", pd.Series(0, index=frame.index)).max() == 1)
        categorical = {"target_is_moon", *(name for name in names if name.startswith("vehicle_family_"))}
        vehicle_only = {"mass_1e5kg", "propellant_fraction", "area_100m2", "cr", "isp_500s"}
        continuous = np.asarray([
            name not in categorical and not (is_moon and name in vehicle_only)
            for name in names
        ])
        ood_fraction = float(np.mean(np.any(np.abs(z[:, continuous]) > 3.0, axis=1)))
        families = sorted(int(value) for value in frame["vehicle_family"].unique())
        known_families = set(checkpoint.get("training_vehicle_families", []))
        unknown_family = not is_moon and any(value not in known_families for value in families)
        gate_reasons = []
        if ood_fraction > args.ood_threshold:
            gate_reasons.append("feature OOD fraction exceeds threshold")
        if unknown_family:
            gate_reasons.append("vehicle family absent from training")
        moon_limit = checkpoint.get("moon_residual_norm_p99_mps2")
        if is_moon and moon_limit is not None and np.any(
            np.linalg.norm(learned, axis=1) > 5.0 * moon_limit
        ):
            gate_reasons.append("predicted Moon correction exceeds five times the training 99th percentile")
        gate_open = not gate_reasons
        gated_learned = learned if gate_open else np.zeros_like(learned)
        gated_pred = gated_learned + thrust_acc
        key = Path(source).stem
        results[key] = {
            "source": source,
            "samples": len(frame),
            "target_is_moon": is_moon,
            "burn_inputs_present": bool(active.any()),
            "continuous_feature_ood_fraction_over_3std": ood_fraction,
            "unknown_vehicle_family": unknown_family,
            "ood_gate_reasons": gate_reasons,
            "ood_gate": "allow" if gate_open else "physics-only",
            "ungated_residual_acceleration": {
                "all": metrics_for(actual, pred, np.ones(len(frame), dtype=bool)),
                "coast": metrics_for(actual, pred, ~active),
                "burn": metrics_for(actual, pred, active),
            },
            "gated_residual_acceleration": {
                "all": metrics_for(actual, gated_pred, np.ones(len(frame), dtype=bool)),
                "coast": metrics_for(actual, gated_pred, ~active),
                "burn": metrics_for(actual, gated_pred, active),
            },
        }
        output = frame[["arc_id", "t_s", "throttle"]].copy()
        for j, axis in enumerate("xyz"):
            output[f"actual_a{axis}"] = actual[:, j]
            output[f"predicted_a{axis}"] = pred[:, j]
            output[f"gated_a{axis}"] = gated_pred[:, j]
        output.to_csv(out_dir / f"{key}_predictions.csv", index=False)
    report = {
        "model": args.model,
        "training_arcs": checkpoint["training_arcs"],
        "learned_residual_output_gain": checkpoint.get("learned_residual_output_gain", 1.0),
        "evaluation_arcs_not_used_for_training": results,
        "warning": "Artemis spacecraft acceleration residuals include unknown maneuvers when no burn history is supplied; interpret those scores as a diagnostic only.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
