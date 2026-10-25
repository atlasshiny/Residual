#!/usr/bin/env python3
"""Test whether a learned solar-tide correction transfers Moon -> spacecraft.

The experiment fits one coefficient multiplying the known solar tidal field
from the Moon residual acceleration, then applies that coefficient unchanged
to a spacecraft arc. It is intentionally separate from the time-conditioned
PINN and does not overwrite its weights or predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MU_SUN = 1.32712440018e20


def load_aligned(reference_path: str, residual_path: str, sun_path: str):
    reference = pd.read_csv(reference_path)
    residual = pd.read_csv(residual_path)
    sun = pd.read_csv(sun_path)
    merged = reference.merge(residual, on="t", suffixes=("", "_res"), validate="one_to_one")
    if len(merged) != len(reference) or len(merged) < 4:
        raise ValueError(f"reference/residual timestamps do not align: {reference_path}")

    t = merged["t"].to_numpy(dtype=float)
    sun_t = sun["t"].to_numpy(dtype=float)
    if t[0] < sun_t[0] or t[-1] > sun_t[-1]:
        raise ValueError(f"Sun ephemeris does not cover {reference_path}")

    r = merged[["x", "y", "z"]].to_numpy(dtype=float)
    dr = merged[["drx", "dry", "drz"]].to_numpy(dtype=float)
    dv = merged[["dvx", "dvy", "dvz"]].to_numpy(dtype=float)
    sun_xyz = np.column_stack(
        [np.interp(t, sun_t, sun[axis].to_numpy(dtype=float)) for axis in "xyz"]
    )

    # Relative solar acceleration in the freely-falling Earth-Moon barycentric
    # frame: acceleration at the target minus acceleration at the origin.
    target_to_sun = sun_xyz - r
    target_dist = np.linalg.norm(target_to_sun, axis=1)
    origin_dist = np.linalg.norm(sun_xyz, axis=1)
    if np.any(target_dist == 0) or np.any(origin_dist == 0):
        raise ValueError("degenerate Sun distance in input ephemeris")
    basis = MU_SUN * (
        target_to_sun / target_dist[:, None] ** 3
        - sun_xyz / origin_dist[:, None] ** 3
    )
    accel_residual = np.gradient(dv, t, axis=0, edge_order=2)
    return t, dr, accel_residual, basis


def vector_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.sum((actual - actual.mean(axis=0)) ** 2)
    if denominator == 0:
        return 0.0
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def integrate_acceleration(t: np.ndarray, accel: np.ndarray) -> np.ndarray:
    """Integrate a correction from zero initial position and velocity error."""
    dv = np.zeros_like(accel)
    dr = np.zeros_like(accel)
    for i, dt in enumerate(np.diff(t)):
        dv[i + 1] = dv[i] + 0.5 * (accel[i] + accel[i + 1]) * dt
        dr[i + 1] = dr[i] + 0.5 * (dv[i] + dv[i + 1]) * dt
    return dr


def rms_km(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(values**2, axis=1))) / 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moon-reference", required=True)
    parser.add_argument("--moon-residuals", required=True)
    parser.add_argument("--spacecraft-reference", required=True)
    parser.add_argument("--spacecraft-residuals", required=True)
    parser.add_argument("--sun", required=True, help="barycentric Sun track in metres")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--out-dir", default="ml/pinn/transfer_solar")
    args = parser.parse_args()
    if not 0.5 <= args.train_fraction < 1.0:
        parser.error("--train-fraction must be in [0.5, 1)")

    mt, moon_dr, moon_da, moon_basis = load_aligned(
        args.moon_reference, args.moon_residuals, args.sun
    )
    st, sc_dr, sc_da, sc_basis = load_aligned(
        args.spacecraft_reference, args.spacecraft_residuals, args.sun
    )

    # Fit alpha only on the early Moon arc. Exclude finite-difference endpoints.
    split = int(len(mt) * args.train_fraction)
    fit_rows = np.arange(1, split - 1)
    holdout_rows = np.arange(split, len(mt) - 1)
    denominator = float(np.sum(moon_basis[fit_rows] ** 2))
    if denominator <= 0:
        raise ValueError("Moon solar-tide basis has no usable energy")
    alpha = float(
        np.sum(moon_basis[fit_rows] * moon_da[fit_rows]) / denominator
    )

    moon_pred_da = alpha * moon_basis
    sc_pred_da = alpha * sc_basis
    moon_pred_dr = integrate_acceleration(mt, moon_pred_da)
    sc_pred_dr = integrate_acceleration(st, sc_pred_da)

    metrics = {
        "experiment": "moon-trained solar-tide coefficient transferred to spacecraft",
        "alpha": alpha,
        "train_fraction": args.train_fraction,
        "moon_samples": len(mt),
        "spacecraft_samples": len(st),
        "moon_holdout_acceleration_r2": vector_r2(
            moon_da[holdout_rows], moon_pred_da[holdout_rows]
        ),
        "moon_holdout_position_r2": vector_r2(
            moon_dr[holdout_rows], moon_pred_dr[holdout_rows]
        ),
        "moon_holdout_baseline_position_rms_km": rms_km(moon_dr[holdout_rows]),
        "moon_holdout_corrected_position_error_rms_km": rms_km(
            moon_dr[holdout_rows] - moon_pred_dr[holdout_rows]
        ),
        "spacecraft_acceleration_r2": vector_r2(sc_da[1:-1], sc_pred_da[1:-1]),
        "spacecraft_position_r2": vector_r2(sc_dr, sc_pred_dr),
        "spacecraft_baseline_position_rms_km": rms_km(sc_dr),
        "spacecraft_corrected_position_error_rms_km": rms_km(sc_dr - sc_pred_dr),
        "interpretation": (
            "single-track diagnostic; positive transfer scores do not establish "
            "generalization beyond this arc"
        ),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame(
        {
            "t": st,
            "actual_drx": sc_dr[:, 0],
            "actual_dry": sc_dr[:, 1],
            "actual_drz": sc_dr[:, 2],
            "predicted_drx": sc_pred_dr[:, 0],
            "predicted_dry": sc_pred_dr[:, 1],
            "predicted_drz": sc_pred_dr[:, 2],
            "actual_dax": sc_da[:, 0],
            "actual_day": sc_da[:, 1],
            "actual_daz": sc_da[:, 2],
            "predicted_dax": sc_pred_da[:, 0],
            "predicted_day": sc_pred_da[:, 1],
            "predicted_daz": sc_pred_da[:, 2],
        }
    ).to_csv(out_dir / "spacecraft_transfer.csv", index=False)

    print(f"Moon-trained solar coefficient alpha={alpha:.6g}")
    print(
        "Moon temporal holdout: "
        f"accel R²={metrics['moon_holdout_acceleration_r2']:.4f}, "
        "corrected position RMS="
        f"{metrics['moon_holdout_corrected_position_error_rms_km']:.1f} km"
    )
    print(
        "Spacecraft transfer: "
        f"accel R²={metrics['spacecraft_acceleration_r2']:.4f}, "
        "baseline/corrected position RMS="
        f"{metrics['spacecraft_baseline_position_rms_km']:.1f}/"
        f"{metrics['spacecraft_corrected_position_error_rms_km']:.1f} km"
    )
    print(f"wrote {out_dir / 'metrics.json'} and {out_dir / 'spacecraft_transfer.csv'}")


if __name__ == "__main__":
    main()
