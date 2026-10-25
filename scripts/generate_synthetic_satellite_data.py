#!/usr/bin/env python3
"""Generate varied synthetic Earth- and Moon-orbit satellite trajectories.

The output is a sequence dataset: each row is a satellite state at a known
epoch, with Earth/Moon/Sun ephemerides and vehicle properties. A path model
learns the transition to the next row and is evaluated by free rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generate_synthetic_rocket_data import (
    AU, EARTH_MOON_A, EARTH_RADIUS, G0, J2, MU_EARTH, MU_MOON, MU_SUN,
    MOON_PERIOD, SOLAR_PRESSURE_1AU, TOTAL_MU, body_states,
)

MOON_RADIUS = 1_737_400.0
MOON_J2 = 2.03263e-4
EARTH_PERIOD = 365.256363004 * 86400.0


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else np.zeros(3)


def j2_acceleration(r, center, mu, radius, j2):
    q = r - center
    d = np.linalg.norm(q)
    z = q[2] / d
    c = 1.5 * j2 * mu * radius**2 / d**5
    return c * np.array([q[0] * (5*z*z - 1), q[1] * (5*z*z - 1), q[2] * (5*z*z - 3)])


def point_gravity(r, t, phases):
    earth, _, moon, _, sun = body_states(t, *phases)
    return (
        -MU_EARTH * (r-earth) / np.linalg.norm(r-earth)**3
        -MU_MOON * (r-moon) / np.linalg.norm(r-moon)**3
        -MU_SUN * ((r-sun) / np.linalg.norm(r-sun)**3 + sun / np.linalg.norm(sun)**3)
    )


def active_control(t, burns, dry_mass, wet_mass):
    force = np.zeros(3)
    mass_flow = 0.0
    for burn in burns:
        elapsed = max(0.0, min(t, burn["stop_s"]) - burn["start_s"])
        mass_flow += burn["force_n"] * elapsed / (burn["isp_s"] * G0)
        if burn["start_s"] <= t < burn["stop_s"]:
            force += burn["force_n"] * np.asarray(burn["direction"])
    mass = max(dry_mass, wet_mass - mass_flow)
    return force / mass, mass


def acceleration(r, wet_mass, area, cr, t, phases, burns, dry_mass):
    earth, _, moon, _, sun = body_states(t, *phases)
    control, mass = active_control(t, burns, dry_mass, wet_mass)
    a = point_gravity(r, t, phases)
    a += j2_acceleration(r, earth, MU_EARTH, EARTH_RADIUS, J2)
    a += j2_acceleration(r, moon, MU_MOON, MOON_RADIUS, MOON_J2)
    sun_rel = r - sun
    a += SOLAR_PRESSURE_1AU * cr * area / mass * (AU / np.linalg.norm(sun_rel))**2 * unit(sun_rel)
    a += control
    return a


def make_initial_state(rng, center, altitude, inclination, phase, node, argp, eccentricity, phases):
    earth, earth_v, moon, moon_v, _ = body_states(0.0, *phases)
    if center == "earth":
        body, body_v, mu, radius = earth, earth_v, MU_EARTH, EARTH_RADIUS
    else:
        body, body_v, mu, radius = moon, moon_v, MU_MOON, MOON_RADIUS
    inc = inclination
    p = np.array([np.cos(node)*np.cos(argp)-np.sin(node)*np.sin(argp)*np.cos(inc),
                  np.sin(node)*np.cos(argp)+np.cos(node)*np.sin(argp)*np.cos(inc),
                  np.sin(argp)*np.sin(inc)])
    q = np.array([-np.cos(node)*np.sin(argp)-np.sin(node)*np.cos(argp)*np.cos(inc),
                  -np.sin(node)*np.sin(argp)+np.cos(node)*np.cos(argp)*np.cos(inc),
                  np.cos(argp)*np.sin(inc)])
    radial = np.cos(phase)*p + np.sin(phase)*q
    tangent = -np.sin(phase)*p + np.cos(phase)*q
    # Treat the configured altitude as periapsis altitude so eccentric arcs
    # remain above the central body's surface.
    periapsis_radius = radius + altitude
    semi_major = periapsis_radius / (1.0 - eccentricity)
    semi_latus = semi_major * (1.0 - eccentricity**2)
    rmag = semi_latus / (1.0 + eccentricity*np.cos(phase))
    r = body + rmag * radial
    vr = np.sqrt(mu/semi_latus) * eccentricity*np.sin(phase)
    vt = np.sqrt(mu/semi_latus) * (1.0 + eccentricity*np.cos(phase))
    v = body_v + vr*radial + vt*tangent
    return r, v


def generate_arc(arc_number, center, orbit_band, rng, duration, sample_dt, internal_dt):
    phases = (
        float(rng.uniform(0, 2*np.pi)), float(rng.uniform(0, 2*np.pi)),
        float(rng.uniform(-0.12, 0.12)), float(rng.uniform(0, 2*np.pi)),
    )
    if center == "earth":
        altitude = {
            "leo": float(rng.uniform(400e3, 2_000e3)),
            "meo": float(rng.uniform(5_000e3, 20_000e3)),
            "high": float(rng.uniform(20_000e3, 35_000e3)),
        }[orbit_band]
    else:
        altitude = {
            "low_lunar": float(rng.uniform(100e3, 500e3)),
            "high_lunar": float(rng.uniform(500e3, 5_000e3)),
        }[orbit_band]
    inclination = float(rng.uniform(0.0, np.pi))
    phase = float(rng.uniform(0, 2*np.pi))
    node, argp = float(rng.uniform(0, 2*np.pi)), float(rng.uniform(0, 2*np.pi))
    eccentricity = float(rng.uniform(0.0, 0.35))
    mass = float(rng.uniform(450, 5_000))
    area = float(rng.uniform(4, 40))
    cr = float(rng.uniform(1.1, 1.8))
    r, v = make_initial_state(rng, center, altitude, inclination, phase, node, argp, eccentricity, phases)
    dry_mass = 0.995 * mass
    burns = []
    for _ in range(int(rng.integers(1, 4))):
        burn_duration = float(rng.uniform(300.0, min(1800.0, duration/8)))
        start = float(rng.uniform(600.0, max(601.0, duration-burn_duration-600.0)))
        direction = unit(v + rng.normal(0.0, 0.25, size=3)).tolist()
        force = float(mass * rng.uniform(1e-5, 2e-4))
        burns.append({"start_s": start, "stop_s": start+burn_duration, "force_n": force,
                      "isp_s": float(rng.uniform(180.0, 320.0)), "direction": direction})
    n_sub = int(round(sample_dt / internal_dt))
    if n_sub <= 0 or abs(n_sub*internal_dt-sample_dt) > 1e-8:
        raise ValueError("sample cadence must be an integer multiple of integration cadence")
    count = int(duration / sample_dt) + 1
    states = []
    state = np.concatenate((r, v))
    for i in range(count):
        t = i * sample_dt
        earth, earth_v, moon, moon_v, sun = body_states(t, *phases)
        control, current_mass = active_control(t, burns, dry_mass, mass)
        total_accel = acceleration(state[:3], mass, area, cr, t, phases, burns, dry_mass)
        correction = total_accel - point_gravity(state[:3], t, phases) - control
        states.append({
            "arc_id": f"sat_{arc_number:03d}", "center": center,
            "orbit_band": orbit_band, "t_s": t,
            "x_m": state[0], "y_m": state[1], "z_m": state[2],
            "vx_mps": state[3], "vy_mps": state[4], "vz_mps": state[5],
            "earth_x_m": earth[0], "earth_y_m": earth[1], "earth_z_m": earth[2],
            "earth_vx_mps": earth_v[0], "earth_vy_mps": earth_v[1], "earth_vz_mps": earth_v[2],
            "moon_x_m": moon[0], "moon_y_m": moon[1], "moon_z_m": moon[2],
            "moon_vx_mps": moon_v[0], "moon_vy_mps": moon_v[1], "moon_vz_mps": moon_v[2],
            "sun_x_m": sun[0], "sun_y_m": sun[1], "sun_z_m": sun[2],
            "mass_kg": current_mass, "initial_mass_kg": mass, "area_m2": area, "cr": cr,
            "control_ax_mps2": control[0], "control_ay_mps2": control[1], "control_az_mps2": control[2],
            "residual_ax_mps2": correction[0], "residual_ay_mps2": correction[1], "residual_az_mps2": correction[2],
        })
        if i == count - 1:
            break
        for k in range(n_sub):
            tk = t + k*internal_dt
            _, mk1 = active_control(tk, burns, dry_mass, mass)
            k1 = np.concatenate((state[3:], acceleration(state[:3], mk1, area, cr, tk, phases, burns, dry_mass)))
            s2 = state + 0.5*internal_dt*k1
            _, mk2 = active_control(tk+0.5*internal_dt, burns, dry_mass, mass)
            k2 = np.concatenate((s2[3:], acceleration(s2[:3], mk2, area, cr, tk+0.5*internal_dt, phases, burns, dry_mass)))
            s3 = state + 0.5*internal_dt*k2
            _, mk3 = active_control(tk+0.5*internal_dt, burns, dry_mass, mass)
            k3 = np.concatenate((s3[3:], acceleration(s3[:3], mk3, area, cr, tk+0.5*internal_dt, phases, burns, dry_mass)))
            s4 = state + internal_dt*k3
            _, mk4 = active_control(tk+internal_dt, burns, dry_mass, mass)
            k4 = np.concatenate((s4[3:], acceleration(s4[:3], mk4, area, cr, tk+internal_dt, phases, burns, dry_mass)))
            state = state + internal_dt/6.0*(k1+2*k2+2*k3+k4)
    return pd.DataFrame(states), {
        "arc_id": f"sat_{arc_number:03d}", "center": center,
        "orbit_band": orbit_band, "altitude_m": altitude,
        "inclination_rad": inclination, "initial_mass_kg": mass,
        "eccentricity": eccentricity, "raan_rad": node, "argument_periapsis_rad": argp,
        "burns": burns,
        "area_m2": area, "cr": cr, "duration_s": duration,
        "sample_dt_s": sample_dt, "internal_dt_s": internal_dt,
        "moon_phase_rad": phases[0], "sun_phase_rad": phases[1],
        "sun_tilt_rad": phases[2], "sun_node_rad": phases[3],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic_satellite_trajectories.csv")
    ap.add_argument("--manifest", default="data/synthetic_satellite_trajectories.manifest.json")
    ap.add_argument("--arcs-per-regime", type=int, default=8)
    ap.add_argument("--duration-days", type=float, default=2.0)
    ap.add_argument("--sample-seconds", type=float, default=600.0)
    ap.add_argument("--integration-seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    regimes = [("earth", band) for band in ("leo", "meo", "high")]
    regimes += [("moon", band) for band in ("low_lunar", "high_lunar")]
    frames, manifests = [], []
    arc_id = 0
    for center, band in regimes:
        for _ in range(args.arcs_per_regime):
            frame, metadata = generate_arc(
                arc_id, center, band, rng, args.duration_days*86400,
                args.sample_seconds, args.integration_seconds,
            )
            frames.append(frame)
            manifests.append(metadata)
            arc_id += 1
    data = pd.concat(frames, ignore_index=True)
    output, manifest_path = Path(args.out), Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, float_format="%.12g")
    manifest = {
        "generator": "scripts/generate_synthetic_satellite_data.py",
        "seed": args.seed, "arcs": len(manifests), "rows": len(data),
        "duration_days": args.duration_days, "sample_seconds": args.sample_seconds,
        "integration_seconds": args.integration_seconds,
        "truth_model": ["Earth/Moon/Sun gravity", "Earth and Moon J2", "solar radiation pressure"],
        "transition_target": "J2 and SRP acceleration residual after explicit point gravity and known thrust",
        "trajectories": manifests,
        "warning": "Synthetic only. Maneuver schedules are synthetic; LRO and Artemis remain external tests.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(data)} rows from {len(manifests)} satellite arcs to {output}")
    print(f"wrote provenance to {manifest_path}")


if __name__ == "__main__":
    main()
