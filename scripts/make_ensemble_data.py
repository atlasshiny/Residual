"""Generate a Monte-Carlo ensemble for the UI's 3D viewer.

Propagates a cloud of particles around the nominal lunar arc using the same
circular Earth-Moon model as ``dynamics`` (velocity-Verlet, J2 off) and writes
``tauri-src/src/data/ensemble.json``.  The result is a demo artifact: it shows
the dispersion the CUDA propagator is built for, without requiring a GPU.

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
MU_MOON = 4.9048695e12
MOON_DISTANCE = 3.844e8
N_PARTICLES = 400
N_STEPS = 60
DT = 600.0  # seconds


def earth_moon_positions(t: float) -> tuple[np.ndarray, np.ndarray]:
    total_mu = MU_EARTH + MU_MOON
    theta = (total_mu / MOON_DISTANCE**3) ** 0.5 * t
    moon_fraction = MU_MOON / total_mu
    c, s = np.cos(theta), np.sin(theta)
    earth = np.array([-MOON_DISTANCE * moon_fraction * c,
                      -MOON_DISTANCE * moon_fraction * s, 0.0])
    moon = np.array([MOON_DISTANCE * (1.0 - moon_fraction) * c,
                     MOON_DISTANCE * (1.0 - moon_fraction) * s, 0.0])
    return earth, moon


def accel(r: np.ndarray, t: float) -> np.ndarray:
    earth, moon = earth_moon_positions(t)
    d_e = r - earth
    d_m = r - moon
    a = -MU_EARTH * d_e / (np.linalg.norm(d_e) ** 3 + 1e-30)
    a = a - MU_MOON * d_m / (np.linalg.norm(d_m) ** 3 + 1e-30)
    return a


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Nominal: circular lunar orbit.
    r0 = np.array([MOON_DISTANCE, 0.0, 0.0])
    v0 = np.array([0.0, (MU_EARTH / MOON_DISTANCE) ** 0.5, 0.0])
    # Use a slightly inclined, elliptical-ish spread for visual interest.
    rng = np.random.default_rng(42)
    pos = np.tile(r0, (N_PARTICLES, 1)).astype(float)
    vel = np.tile(v0, (N_PARTICLES, 1)).astype(float)
    pos += rng.normal(0, 2e6, pos.shape)
    vel += rng.normal(0, 50.0, vel.shape)

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