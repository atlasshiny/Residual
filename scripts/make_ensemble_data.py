"""Generate a Monte-Carlo ensemble for the UI's 3D viewer.

Propagates a cloud of test particles seeded from the reference state with an
Earth-only two-body model (velocity-Verlet) and writes
``tauri-src/src/data/ensemble.json``.  This is deliberately *not* the n-body
propagator: the reference object is the Moon itself, so treating it as a point
mass would make nearby particles fall into the attractor.  The result is a
demo artifact that shows bounded orbital dispersion without a GPU.

Usage:
    uv run python scripts/make_ensemble_data.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tauri-src" / "src" / "data" / "ensemble.json"

MU_EARTH = 3.986004418e14
N_PARTICLES = 400
N_STEPS = 60
DT = 600.0  # seconds
# Initial dispersion: 20,000 km and 5 m/s stay well clear of the Earth.
SIGMA_POS_M = 20_000_000.0
SIGMA_VEL_M_S = 5.0


def accel(r: np.ndarray, t: float) -> np.ndarray:
    # Earth-only: see the module docstring for why the Moon is not an attractor.
    norm = np.linalg.norm(r)
    return -MU_EARTH * r / (norm**3 + 1e-30)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Nominal state: the first reference sample, so the cloud starts on the arc.
    import csv

    with (ROOT / "data" / "reference.csv").open(newline="") as stream:
        first = next(csv.DictReader(stream))
    r0 = np.array([float(first["x"]), float(first["y"]), float(first["z"])])
    v0 = np.array([float(first["vx"]), float(first["vy"]), float(first["vz"])])
    rng = np.random.default_rng(42)
    pos = np.tile(r0, (N_PARTICLES, 1)).astype(float)
    vel = np.tile(v0, (N_PARTICLES, 1)).astype(float)
    pos += rng.normal(0, SIGMA_POS_M, pos.shape)
    vel += rng.normal(0, SIGMA_VEL_M_S, vel.shape)

    nominal_pos = r0.copy()
    nominal_vel = v0.copy()

    epochs = [i * DT for i in range(N_STEPS + 1)]
    nominal_traj = [nominal_pos.copy()]
    clouds = [pos.copy()]
    stats = []

    def rms(a):
        return float(np.sqrt(np.mean(a**2)))

    for step in range(N_STEPS):
        t = step * DT
        # velocity-Verlet for the whole cloud (vectorised over particles)
        a0 = np.array([accel(p, t) for p in pos])
        vel = vel + a0 * (0.5 * DT)
        pos = pos + vel * DT
        a1 = np.array([accel(p, t + DT) for p in pos])
        vel = vel + a1 * (0.5 * DT)

        # nominal
        an0 = accel(nominal_pos, t)
        nominal_vel = nominal_vel + an0 * (0.5 * DT)
        nominal_pos = nominal_pos + nominal_vel * DT
        an1 = accel(nominal_pos, t + DT)
        nominal_vel = nominal_vel + an1 * (0.5 * DT)

        disp = pos - nominal_pos
        stats.append({
            "rms_m": rms(disp.flatten()),
            "std_x_m": float(np.std(disp[:, 0])),
            "std_y_m": float(np.std(disp[:, 1])),
            "std_z_m": float(np.std(disp[:, 2])),
        })
        clouds.append(pos.copy())
        nominal_traj.append(nominal_pos.copy())

    payload = {
        "schema": "ensemble/v1",
        "model": "earth-two-body",
        "note": "demo dispersion cloud; not the n-body propagator",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_particles": N_PARTICLES,
        "n_epochs": N_STEPS + 1,
        "dt_s": DT,
        "epochs_s": epochs,
        "units": "m",
        "nominal": [[float(x) for x in p] for p in nominal_traj],
        "clouds": [[[float(x) for x in p] for p in cloud] for cloud in clouds],
        "stats": stats,
    }
    OUT.write_text(json.dumps(payload))
    print(f"wrote {OUT} ({N_PARTICLES} particles x {N_STEPS + 1} epochs)")


if __name__ == "__main__":
    main()