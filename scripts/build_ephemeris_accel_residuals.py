#!/usr/bin/env python3
"""Create state/acceleration residual rows from an ephemeris CSV arc.

The central-difference acceleration label is reference acceleration minus the
same Earth/Moon/Sun point-gravity baseline used by the synthetic generator.
Moon arcs are appropriate training data; spacecraft arcs are diagnostics when
burn histories are unavailable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MU_EARTH = 3.986004418e14
MU_MOON = 4.9048695e12
MU_SUN = 1.32712440018e20


def interp_track(track, t, position=True):
    cols = ["x", "y", "z"] if position else ["vx", "vy", "vz"]
    return np.column_stack([np.interp(t, track.t, track[c]) for c in cols])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=("moon", "spacecraft"), required=True)
    ap.add_argument("--reference", required=True, help="barycentric target state CSV")
    ap.add_argument("--moon", help="barycentric Moon state CSV; required for spacecraft")
    ap.add_argument("--sun", required=True, help="barycentric Sun state CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mass-kg", type=float, default=26500.0)
    ap.add_argument("--dry-mass-kg", type=float, default=20000.0)
    ap.add_argument("--area-m2", type=float, default=100.0)
    ap.add_argument("--cr", type=float, default=1.3)
    ap.add_argument("--vehicle-family", type=int, default=4)
    args = ap.parse_args()
    if args.target == "spacecraft" and not args.moon:
        ap.error("--moon is required for --target spacecraft")

    ref = pd.read_csv(args.reference)
    sun = pd.read_csv(args.sun)
    if args.target == "moon":
        moon = ref
    else:
        moon = pd.read_csv(args.moon)
    t = ref.t.to_numpy(dtype=float)
    if len(t) < 5 or np.any(np.diff(t) <= 0):
        raise ValueError("reference must contain at least five increasing timestamps")
    if t[0] < moon.t.min() or t[-1] > moon.t.max() or t[0] < sun.t.min() or t[-1] > sun.t.max():
        raise ValueError("Moon and Sun ephemerides must cover the target arc")

    r = ref[["x", "y", "z"]].to_numpy(dtype=float)
    v = ref[["vx", "vy", "vz"]].to_numpy(dtype=float)
    moon_r = interp_track(moon, t, True)
    moon_v = interp_track(moon, t, False)
    sun_r = interp_track(sun, t, True)
    earth_r = -MU_MOON / MU_EARTH * moon_r
    earth_v = -MU_MOON / MU_EARTH * moon_v

    re = r - earth_r
    rm = r - moon_r
    rs = r - sun_r
    sun_origin = np.linalg.norm(sun_r, axis=1)
    acceleration = (
        -MU_EARTH * re / np.linalg.norm(re, axis=1)[:, None] ** 3
        -MU_SUN * (rs / np.linalg.norm(rs, axis=1)[:, None] ** 3
                   + sun_r / sun_origin[:, None] ** 3)
    )
    if args.target == "spacecraft":
        acceleration -= MU_MOON * rm / np.linalg.norm(rm, axis=1)[:, None] ** 3
    observed_acc = np.gradient(v, t, axis=0, edge_order=2)
    residual_acc = observed_acc - acceleration

    rows = pd.DataFrame({
        "arc_id": Path(args.out).stem,
        "t_s": t,
        "x_m": r[:, 0], "y_m": r[:, 1], "z_m": r[:, 2],
        "vx_mps": v[:, 0], "vy_mps": v[:, 1], "vz_mps": v[:, 2],
        "earth_x_m": earth_r[:, 0], "earth_y_m": earth_r[:, 1], "earth_z_m": earth_r[:, 2],
        "earth_vx_mps": earth_v[:, 0], "earth_vy_mps": earth_v[:, 1], "earth_vz_mps": earth_v[:, 2],
        "moon_x_m": moon_r[:, 0], "moon_y_m": moon_r[:, 1], "moon_z_m": moon_r[:, 2],
        "moon_vx_mps": moon_v[:, 0], "moon_vy_mps": moon_v[:, 1], "moon_vz_mps": moon_v[:, 2],
        "sun_x_m": sun_r[:, 0], "sun_y_m": sun_r[:, 1], "sun_z_m": sun_r[:, 2],
        "mass_kg": args.mass_kg if args.target == "spacecraft" else 0.0,
        "dry_mass_kg": args.dry_mass_kg if args.target == "spacecraft" else 0.0,
        "area_m2": args.area_m2 if args.target == "spacecraft" else 0.0,
        "cr": args.cr if args.target == "spacecraft" else 0.0,
        "vehicle_family": args.vehicle_family if args.target == "spacecraft" else -1,
        "target_is_moon": int(args.target == "moon"),
        "thrust_n_x": 0.0, "thrust_n_y": 0.0, "thrust_n_z": 0.0,
        "isp_s": 0.0, "throttle": 0,
        "truth_ax_mps2": observed_acc[:, 0], "truth_ay_mps2": observed_acc[:, 1], "truth_az_mps2": observed_acc[:, 2],
        "baseline_ax_mps2": acceleration[:, 0], "baseline_ay_mps2": acceleration[:, 1], "baseline_az_mps2": acceleration[:, 2],
        "residual_ax_mps2": residual_acc[:, 0],
        "residual_ay_mps2": residual_acc[:, 1],
        "residual_az_mps2": residual_acc[:, 2],
    }).iloc[1:-1].reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out, index=False, float_format="%.12g")
    norms = np.linalg.norm(residual_acc[1:-1], axis=1)
    print(f"wrote {len(rows)} {args.target} residual rows to {args.out}")
    print(f"acceleration residual norm median={np.median(norms):.6g}, p95={np.quantile(norms, .95):.6g} m/s^2")


if __name__ == "__main__":
    main()
