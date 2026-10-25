#!/usr/bin/env python3
"""Fit a solar-tide model on paired Moon runs and test it on paired spacecraft runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transfer_sun_tide import MU_SUN, integrate_acceleration, rms_km, vector_r2


def load_effect(reference_path: str, effect_path: str, sun_path: str):
    reference = pd.read_csv(reference_path)
    effect = pd.read_csv(effect_path)
    sun = pd.read_csv(sun_path)
    joined = reference.merge(effect, on="t", suffixes=("", "_effect"), validate="one_to_one")
    if len(joined) != len(reference) or len(joined) < 4:
        raise ValueError(f"reference/effect timestamps do not align: {reference_path}")
    t = joined["t"].to_numpy(dtype=float)
    sun_t = sun["t"].to_numpy(dtype=float)
    if t[0] < sun_t[0] or t[-1] > sun_t[-1]:
        raise ValueError(f"Sun ephemeris does not cover {reference_path}")

    r = joined[["x", "y", "z"]].to_numpy(dtype=float)
    dr = joined[["drx", "dry", "drz"]].to_numpy(dtype=float)
    dv = joined[["dvx", "dvy", "dvz"]].to_numpy(dtype=float)
    sun_xyz = np.column_stack(
        [np.interp(t, sun_t, sun[axis].to_numpy(dtype=float)) for axis in "xyz"]
    )
    target_to_sun = sun_xyz - r
    target_dist = np.linalg.norm(target_to_sun, axis=1)
    origin_dist = np.linalg.norm(sun_xyz, axis=1)
    basis = MU_SUN * (
        target_to_sun / target_dist[:, None] ** 3
        - sun_xyz / origin_dist[:, None] ** 3
    )
    effect_accel = np.gradient(dv, t, axis=0, edge_order=2)
    return t, dr, effect_accel, basis


def fit_alpha(trajectories, train_fraction: float):
    numerators = []
    denominators = []
    holdouts = []
    for name, t, dr, accel, basis in trajectories:
        split = int(len(t) * train_fraction)
        fit = np.arange(1, split - 1)
        holdout = np.arange(split, len(t) - 1)
        numerators.append(float(np.sum(basis[fit] * accel[fit]) / len(fit)))
        denominators.append(float(np.sum(basis[fit] ** 2) / len(fit)))
        holdouts.append((name, t, dr, accel, basis, holdout))
    denominator = sum(denominators)
    if denominator <= 0:
        raise ValueError("training solar-tide basis has no usable energy")
    return sum(numerators) / denominator, holdouts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        nargs=4,
        action="append",
        metavar=("NAME", "REFERENCE", "EFFECT", "SUN"),
        required=True,
        help="training trajectory; repeat for each training arc",
    )
    parser.add_argument(
        "--test",
        nargs=4,
        action="append",
        metavar=("NAME", "REFERENCE", "EFFECT", "SUN"),
        required=True,
        help="held-out trajectory; repeat for each transfer target",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--out-dir", default="ml/pinn/transfer_solar_paired")
    args = parser.parse_args()
    if not 0.5 <= args.train_fraction < 1.0:
        parser.error("--train-fraction must be in [0.5, 1)")

    train_data = []
    for name, reference, effect, sun in args.train:
        t, dr, accel, basis = load_effect(reference, effect, sun)
        train_data.append((name, t, dr, accel, basis))
    alpha, holdouts = fit_alpha(train_data, args.train_fraction)

    metrics = {
        "experiment": "paired Sun/no-Sun force effect; pooled coefficient transfer",
        "alpha": alpha,
        "train_fraction": args.train_fraction,
        "training_trajectories": [
            {"name": name, "samples": len(t)} for name, t, *_ in train_data
        ],
        "temporal_holdouts": {},
        "transfer_tests": {},
        "interpretation": (
            "isolates the force-model transfer question; it does not model "
            "maneuvers or total spacecraft ephemeris error"
        ),
    }
    for name, t, dr, accel, basis, holdout in holdouts:
        pred_a = alpha * basis
        pred_r = integrate_acceleration(t, pred_a)
        metrics["temporal_holdouts"][name] = {
            "samples": int(len(holdout)),
            "acceleration_r2": vector_r2(accel[holdout], pred_a[holdout]),
            "position_effect_error_rms_km": rms_km(dr[holdout] - pred_r[holdout]),
            "position_effect_rms_km": rms_km(dr[holdout]),
        }

    transfer_data = []
    for name, reference, effect, sun in args.test:
        t, dr, accel, basis = load_effect(reference, effect, sun)
        pred_a = alpha * basis
        pred_r = integrate_acceleration(t, pred_a)
        metrics["transfer_tests"][name] = {
            "samples": len(t),
            "acceleration_r2": vector_r2(accel[1:-1], pred_a[1:-1]),
            "position_effect_error_rms_km": rms_km(dr - pred_r),
            "position_effect_rms_km": rms_km(dr),
        }
        transfer_data.append((name, t, dr, accel, pred_r, pred_a))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for name, t, dr, accel, pred_r, pred_a in transfer_data:
        pd.DataFrame(
            {
                "t": t,
                "actual_sun_drx": dr[:, 0],
                "actual_sun_dry": dr[:, 1],
                "actual_sun_drz": dr[:, 2],
                "predicted_sun_drx": pred_r[:, 0],
                "predicted_sun_dry": pred_r[:, 1],
                "predicted_sun_drz": pred_r[:, 2],
                "actual_sun_dax": accel[:, 0],
                "actual_sun_day": accel[:, 1],
                "actual_sun_daz": accel[:, 2],
                "predicted_sun_dax": pred_a[:, 0],
                "predicted_sun_day": pred_a[:, 1],
                "predicted_sun_daz": pred_a[:, 2],
            }
        ).to_csv(out / f"{name}_transfer.csv", index=False)
    print(f"Pooled solar coefficient alpha={alpha:.6g}")
    for name, values in metrics["temporal_holdouts"].items():
        print(
            f"train holdout {name}: acceleration R²={values['acceleration_r2']:.4f}, "
            f"position-effect error={values['position_effect_error_rms_km']:.2f} km"
        )
    for name, values in metrics["transfer_tests"].items():
        print(
            f"transfer {name}: acceleration R²={values['acceleration_r2']:.4f}, "
            f"position-effect error={values['position_effect_error_rms_km']:.2f} km / "
            f"effect RMS={values['position_effect_rms_km']:.2f} km"
        )
    print(f"wrote {out / 'metrics.json'} and {len(transfer_data)} transfer CSVs")


if __name__ == "__main__":
    main()
