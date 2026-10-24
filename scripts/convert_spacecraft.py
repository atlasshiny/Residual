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

    # Barycentric split using nominal or instantaneous separation.
    rows = []
    for _, r in sc.iterrows():
        # Find nearest moon sample by time for separation (or use nominal).
        if preserve_distance:
            # match by index assuming same cadence; fallback nominal
            idx = min(int(r["t"] / 600.0), len(moon) - 1)
            D = _norm((moon["x"].iloc[idx], moon["y"].iloc[idx], moon["z"].iloc[idx]))
            if D <= 0:
                D = EARTH_MOON_DISTANCE
        else:
            D = EARTH_MOON_DISTANCE
        # Earth position in barycentric frame (rotated) = -D * mu_moon/total_mu along xhat
        earth_offset = -D * MU_MOON / TOTAL_MU
        # Apply rotation to spacecraft geocentric vector, then shift by earth offset.
        rx, ry, rz = rotate_vec(r["x"], r["y"], r["z"])
        vx, vy, vz = rotate_vec(r["vx"], r["vy"], r["vz"])
        rows.append({
            "t": r["t"],
            "x": rx + earth_offset,
            "y": ry,
            "z": rz,
            "vx": vx,
            "vy": vy,
            "vz": vz,
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
