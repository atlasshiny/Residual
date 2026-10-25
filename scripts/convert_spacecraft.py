#!/usr/bin/env python3
"""Convert a geocentric spacecraft arc (km, s) to the solver's barycentric frame.

Uses a companion Moon arc (raw_moon.csv format) to determine the rotation that
aligns the epoch Earth->Moon vector with +x and the orbital angular momentum
with +z, then applies the same rotation + barycentric split to the spacecraft.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

MU_EARTH = 3.986004418e14
MU_MOON = 4.9048695e12
TOTAL_MU = MU_EARTH + MU_MOON
EARTH_MOON_DISTANCE = 3.844e8


def _norm(v):
    return math.sqrt(sum(c * c for c in v))


def _load(path):
    df = pd.read_csv(path)
    for c in ("t", "x", "y", "z", "vx", "vy", "vz"):
        if c not in df.columns:
            raise ValueError(f"missing column {c} in {path}")
    return df


def convert(spacecraft_csv, moon_csv, out_csv, preserve_distance=False):
    sc = _load(spacecraft_csv)
    moon = _load(moon_csv)

    # Scale km -> m, s stays.
    for c in ("x", "y", "z", "vx", "vy", "vz"):
        sc[c] = sc[c] * 1000.0
        moon[c] = moon[c] * 1000.0
    sc = sc.sort_values("t").reset_index(drop=True)
    moon = moon.sort_values("t").reset_index(drop=True)
    if moon["t"].iloc[0] > sc["t"].iloc[0] or moon["t"].iloc[-1] < sc["t"].iloc[-1]:
        raise ValueError("Moon ephemeris must cover the full spacecraft time span")

    # Alignment: epoch Earth->Moon vector.
    d0 = (moon["x"].iloc[0], moon["y"].iloc[0], moon["z"].iloc[0])
    h = (
        d0[1] * moon["vz"].iloc[0] - d0[2] * moon["vy"].iloc[0],
        d0[2] * moon["vx"].iloc[0] - d0[0] * moon["vz"].iloc[0],
        d0[0] * moon["vy"].iloc[0] - d0[1] * moon["vx"].iloc[0],
    )
    # Build rotation: +x along d0, +z along h, +y = z x x.
    xhat = [c / _norm(d0) for c in d0]
    zhat = [c / _norm(h) for c in h]
    yhat = (
        zhat[1] * xhat[2] - zhat[2] * xhat[1],
        zhat[2] * xhat[0] - zhat[0] * xhat[2],
        zhat[0] * xhat[1] - zhat[1] * xhat[0],
    )
    # Rotation matrix R (new coords = R @ old). Rows are new basis in old frame.
    R = (xhat, yhat, zhat)

    def rotate_vec(vx, vy, vz):
        return (
            R[0][0] * vx + R[0][1] * vy + R[0][2] * vz,
            R[1][0] * vx + R[1][1] * vy + R[1][2] * vz,
            R[2][0] * vx + R[2][1] * vy + R[2][2] * vz,
        )

    # Interpolate the Moon state at each spacecraft timestamp. The Moon and
    # spacecraft Horizons queries may use different cadences.
    moon_state = {
        col: np.interp(sc["t"], moon["t"], moon[col])
        for col in ("x", "y", "z", "vx", "vy", "vz")
    }

    # Barycentric split using the full, time-varying Earth-to-Moon vector.
    rows = []
    for i, (_, r) in enumerate(sc.iterrows()):
        moon_r = tuple(moon_state[c][i] for c in ("x", "y", "z"))
        moon_v = tuple(moon_state[c][i] for c in ("vx", "vy", "vz"))
        if not preserve_distance:
            separation = _norm(moon_r)
            factor = EARTH_MOON_DISTANCE / separation
            moon_r = tuple(value * factor for value in moon_r)
            moon_v = tuple(value * factor for value in moon_v)
        earth_offset = tuple(-MU_MOON / TOTAL_MU * value for value in rotate_vec(*moon_r))
        earth_velocity = tuple(-MU_MOON / TOTAL_MU * value for value in rotate_vec(*moon_v))
        # Rotate the Earth-centred spacecraft state and add the actual
        # barycentric Earth position and velocity at this epoch.
        rx, ry, rz = rotate_vec(r["x"], r["y"], r["z"])
        vx, vy, vz = rotate_vec(r["vx"], r["vy"], r["vz"])
        rows.append({
            "t": r["t"],
            "x": rx + earth_offset[0],
            "y": ry + earth_offset[1],
            "z": rz + earth_offset[2],
            "vx": vx + earth_velocity[0],
            "vy": vy + earth_velocity[1],
            "vz": vz + earth_velocity[2],
        })

    out = pd.DataFrame(rows, columns=["t", "x", "y", "z", "vx", "vy", "vz"])
    out.to_csv(out_csv, index=False)
    print(f"wrote {len(out)} rows to {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spacecraft_csv")
    ap.add_argument("moon_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--preserve-distance", action="store_true")
    args = ap.parse_args()
    convert(args.spacecraft_csv, args.moon_csv, args.out_csv, args.preserve_distance)


if __name__ == "__main__":
    main()
