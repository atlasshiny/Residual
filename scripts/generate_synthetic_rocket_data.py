#!/usr/bin/env python3
"""Generate reproducible synthetic cislunar rocket state/acceleration data.

The truth trajectories include finite-duration thrust, propellant mass flow,
Earth J2, and solar radiation pressure. The baseline omits thrust and the two
non-point-mass effects. Each row contains state/vehicle/control features and
the truth-minus-baseline acceleration label for a state-conditioned dynamics
model. This is synthetic training data, not an Artemis reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MU_EARTH = 3.986004418e14
MU_MOON = 4.9048695e12
MU_SUN = 1.32712440018e20
TOTAL_MU = MU_EARTH + MU_MOON
EARTH_MOON_A = 384_400_000.0
MOON_PERIOD = 27.321661 * 86400.0
AU = 149_597_870_700.0
G0 = 9.80665
EARTH_RADIUS = 6_378_136.3
J2 = 1.08262668e-3
SOLAR_PRESSURE_1AU = 4.56e-6


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else np.zeros(3)


def body_states(t: float, moon_phase: float, sun_phase: float, sun_tilt: float, sun_node: float):
    """Circular Earth/Moon and Sun ephemerides in an inertial barycentric frame."""
    moon_angle = moon_phase + 2.0 * np.pi * t / MOON_PERIOD
    sun_angle = sun_phase + 2.0 * np.pi * t / (365.256363004 * 86400.0)
    moon_rel = EARTH_MOON_A * np.array([np.cos(moon_angle), np.sin(moon_angle), 0.0])
    moon_rel_v = EARTH_MOON_A * (2.0 * np.pi / MOON_PERIOD) * np.array(
        [-np.sin(moon_angle), np.cos(moon_angle), 0.0]
    )
    earth = -MU_MOON / TOTAL_MU * moon_rel
    earth_v = -MU_MOON / TOTAL_MU * moon_rel_v
    moon = MU_EARTH / TOTAL_MU * moon_rel
    moon_v = MU_EARTH / TOTAL_MU * moon_rel_v
    e1 = np.array([np.cos(sun_node), np.sin(sun_node), 0.0])
    e2 = np.array([
        -np.sin(sun_node) * np.cos(sun_tilt),
        np.cos(sun_node) * np.cos(sun_tilt),
        np.sin(sun_tilt),
    ])
    sun = AU * (np.cos(sun_angle) * e1 + np.sin(sun_angle) * e2)
    return earth, earth_v, moon, moon_v, sun


def point_gravity(r: np.ndarray, t: float, phases: tuple[float, ...]) -> np.ndarray:
    earth, _, moon, _, sun = body_states(t, *phases)
    re, rm, rs = r - earth, r - moon, r - sun
    # Sun term is differential acceleration relative to the Earth-Moon origin.
    return (
        -MU_EARTH * re / np.linalg.norm(re) ** 3
        - MU_MOON * rm / np.linalg.norm(rm) ** 3
        - MU_SUN * (rs / np.linalg.norm(rs) ** 3 + sun / np.linalg.norm(sun) ** 3)
    )


def burn_at(t: float, burns: list[dict], mass: float, dry_mass: float, r: np.ndarray, v: np.ndarray, earth: np.ndarray, earth_v: np.ndarray):
    thrust = np.zeros(3)
    isp = 0.0
    for burn in burns:
        if burn["start"] <= t < burn["stop"] and mass > dry_mass:
            if burn.get("guidance") == "prograde":
                prograde = unit(v - earth_v)
                normal = unit(np.cross(r - earth, prograde))
                side = unit(np.cross(normal, prograde))
                direction = unit(
                    prograde + burn["side_angle"] * side + burn["normal_angle"] * normal
                )
            else:
                direction = np.asarray(burn["direction"], dtype=float)
            thrust += burn["force_n"] * direction
            isp = burn["isp_s"]
    return thrust, isp


def truth_acceleration(
    r: np.ndarray,
    v: np.ndarray,
    mass: float,
    t: float,
    phases: tuple[float, ...],
    burns: list[dict],
    dry_mass: float,
    area_m2: float,
    cr: float,
):
    earth, _, _, _, sun = body_states(t, *phases)
    rel = r - earth
    rel_norm = np.linalg.norm(rel)
    z_ratio = rel[2] / rel_norm
    j2_factor = 1.5 * J2 * MU_EARTH * EARTH_RADIUS**2 / rel_norm**5
    j2_acc = j2_factor * np.array(
        [rel[0] * (5 * z_ratio**2 - 1), rel[1] * (5 * z_ratio**2 - 1), rel[2] * (5 * z_ratio**2 - 3)]
    )
    sun_rel = r - sun
    sun_dist = np.linalg.norm(sun_rel)
    srp = SOLAR_PRESSURE_1AU * cr * area_m2 / mass * (AU / sun_dist) ** 2 * unit(sun_rel)
    _, earth_v, _, _, _ = body_states(t, *phases)
    thrust, isp = burn_at(t, burns, mass, dry_mass, r, v, earth, earth_v)
    return point_gravity(r, t, phases) + j2_acc + srp + thrust / mass, thrust, isp


def make_burns(rng: np.random.Generator, r0: np.ndarray, v0: np.ndarray, duration: float):
    prograde = unit(v0)
    normal = unit(np.cross(r0, v0))
    transverse = unit(np.cross(normal, prograde))
    direction = unit(prograde + rng.normal(0, 0.12) * transverse + rng.normal(0, 0.05) * normal)
    tli_start = float(rng.uniform(300.0, 1800.0))
    tli_duration = float(rng.uniform(360.0, 900.0))
    burns = [{
        "kind": "injection", "start": tli_start,
        "stop": min(tli_start + tli_duration, duration),
        "force_n": float(rng.uniform(180_000, 320_000)),
        "isp_s": float(rng.uniform(350, 455)), "direction": direction.tolist(),
        "guidance": "prograde", "side_angle": float(rng.normal(0, 0.08)),
        "normal_angle": float(rng.normal(0, 0.04)),
    }]
    if duration > 30_000:
        tcm_start = float(rng.uniform(0.55, 0.8) * duration)
        tcm_direction = unit(prograde + rng.normal(0, 0.25) * transverse + rng.normal(0, 0.1) * normal)
        burns.append({
            "kind": "correction", "start": tcm_start,
            "stop": min(tcm_start + float(rng.uniform(20, 120)), duration),
            "force_n": float(rng.uniform(2_000, 18_000)),
            "isp_s": float(rng.uniform(280, 450)), "direction": tcm_direction.tolist(),
        })
    return burns


def generate_arc(arc_id: int, rng: np.random.Generator, duration: float, sample_dt: float, integration_dt: float):
    phases = (
        float(rng.uniform(0, 2 * np.pi)),
        float(rng.uniform(0, 2 * np.pi)),
        float(rng.uniform(-0.12, 0.12)),
        float(rng.uniform(0, 2 * np.pi)),
    )
    earth, earth_v, _, _, _ = body_states(0.0, *phases)
    radius = EARTH_RADIUS + float(rng.uniform(180_000, 600_000))
    phase = float(rng.uniform(0, 2 * np.pi))
    inclination = float(rng.uniform(-0.35, 0.35))
    radial = np.array([np.cos(phase), np.sin(phase) * np.cos(inclination), np.sin(phase) * np.sin(inclination)])
    r = earth + radius * radial
    tangent = unit(np.cross(np.array([0.0, 0.0, 1.0]), radial))
    if np.linalg.norm(tangent) < 0.1:
        tangent = unit(np.cross(np.array([0.0, 1.0, 0.0]), radial))
    speed = np.sqrt(MU_EARTH / radius) + float(rng.uniform(-80, 80))
    v = earth_v + speed * tangent
    burns = make_burns(rng, r - earth, v - earth_v, duration)
    mass0 = float(rng.uniform(58_000, 90_000))
    dry_mass = float(rng.uniform(18_000, 28_000))
    dry_mass = min(dry_mass, 0.8 * mass0)
    area = float(rng.uniform(45, 150))
    cr = float(rng.uniform(1.1, 1.8))
    vehicle_family = int(rng.integers(0, 4))

    rows = []
    steps_per_sample = int(round(sample_dt / integration_dt))
    if abs(steps_per_sample * integration_dt - sample_dt) > 1e-8:
        raise ValueError("sample_dt must be an integer multiple of integration_dt")
    n_samples = int(duration / sample_dt) + 1
    mass = mass0

    def derivative(state, current_mass, t):
        pos, vel = state[:3], state[3:]
        acc, thrust, isp = truth_acceleration(pos, vel, current_mass, t, phases, burns, dry_mass, area, cr)
        mdot = -np.linalg.norm(thrust) / (max(isp, 1.0) * G0) if np.linalg.norm(thrust) else 0.0
        if current_mass <= dry_mass:
            mdot = 0.0
        return np.concatenate((vel, acc)), mdot

    state = np.concatenate((r, v))
    for sample_index in range(n_samples):
        t = sample_index * sample_dt
        earth, earth_v, moon, moon_v, sun = body_states(t, *phases)
        accel, thrust, isp = truth_acceleration(state[:3], state[3:], mass, t, phases, burns, dry_mass, area, cr)
        baseline_accel = point_gravity(state[:3], t, phases)
        rows.append({
            "arc_id": f"synthetic_{arc_id:03d}", "sample": sample_index, "t_s": t,
            "x_m": state[0], "y_m": state[1], "z_m": state[2],
            "vx_mps": state[3], "vy_mps": state[4], "vz_mps": state[5],
            "earth_x_m": earth[0], "earth_y_m": earth[1], "earth_z_m": earth[2],
            "earth_vx_mps": earth_v[0], "earth_vy_mps": earth_v[1], "earth_vz_mps": earth_v[2],
            "moon_x_m": moon[0], "moon_y_m": moon[1], "moon_z_m": moon[2],
            "moon_vx_mps": moon_v[0], "moon_vy_mps": moon_v[1], "moon_vz_mps": moon_v[2],
            "sun_x_m": sun[0], "sun_y_m": sun[1], "sun_z_m": sun[2],
            "mass_kg": mass, "dry_mass_kg": dry_mass, "area_m2": area, "cr": cr,
            "vehicle_family": vehicle_family,
            "target_is_moon": 0,
            "thrust_n_x": thrust[0], "thrust_n_y": thrust[1], "thrust_n_z": thrust[2],
            "isp_s": isp,
            "truth_ax_mps2": accel[0], "truth_ay_mps2": accel[1], "truth_az_mps2": accel[2],
            "baseline_ax_mps2": baseline_accel[0], "baseline_ay_mps2": baseline_accel[1], "baseline_az_mps2": baseline_accel[2],
            "residual_ax_mps2": accel[0] - baseline_accel[0],
            "residual_ay_mps2": accel[1] - baseline_accel[1],
            "residual_az_mps2": accel[2] - baseline_accel[2],
            "throttle": float(np.linalg.norm(thrust) > 0),
        })
        if sample_index == n_samples - 1:
            break
        for substep in range(steps_per_sample):
            ti = t + substep * integration_dt
            k1, m1 = derivative(state, mass, ti)
            k2, m2 = derivative(state + 0.5 * integration_dt * k1, mass + 0.5 * integration_dt * m1, ti + 0.5 * integration_dt)
            k3, m3 = derivative(state + 0.5 * integration_dt * k2, mass + 0.5 * integration_dt * m2, ti + 0.5 * integration_dt)
            k4, m4 = derivative(state + integration_dt * k3, mass + integration_dt * m3, ti + integration_dt)
            state += integration_dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            mass += integration_dt / 6.0 * (m1 + 2 * m2 + 2 * m3 + m4)
            mass = max(mass, dry_mass)
    frame = pd.DataFrame(rows)
    burn_manifest = [{**b, "direction": [float(x) for x in b["direction"]]} for b in burns]
    return frame, {
        "arc_id": f"synthetic_{arc_id:03d}", "samples": len(frame),
        "duration_s": duration, "sample_dt_s": sample_dt,
        "initial_mass_kg": mass0, "dry_mass_kg": dry_mass,
        "final_mass_kg": mass, "area_m2": area, "cr": cr,
        "vehicle_family": vehicle_family, "moon_phase_rad": phases[0],
        "sun_phase_rad": phases[1], "sun_tilt_rad": phases[2],
        "sun_node_rad": phases[3], "burns": burn_manifest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic_rocket_transfer.csv")
    parser.add_argument("--manifest", default="data/synthetic_rocket_transfer.manifest.json")
    parser.add_argument("--arcs", type=int, default=12)
    parser.add_argument("--duration-days", type=float, default=4.0)
    parser.add_argument("--sample-seconds", type=float, default=60.0)
    parser.add_argument("--integration-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if args.arcs < 2 or args.duration_days <= 0:
        parser.error("need at least two arcs and a positive duration")
    rng = np.random.default_rng(args.seed)
    duration = args.duration_days * 86400.0
    frames, arc_manifests = [], []
    for arc_id in range(args.arcs):
        frame, metadata = generate_arc(
            arc_id, rng, duration, args.sample_seconds, args.integration_seconds
        )
        frames.append(frame)
        arc_manifests.append(metadata)
        print(f"generated {metadata['arc_id']}: {metadata['samples']} rows, "
              f"mass {metadata['initial_mass_kg']:.0f}->{metadata['final_mass_kg']:.0f} kg")
    output = Path(args.out)
    manifest_path = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(output, index=False, float_format="%.12g")
    manifest = {
        "generator": "scripts/generate_synthetic_rocket_data.py",
        "seed": args.seed, "arcs": args.arcs,
        "duration_days": args.duration_days,
        "sample_seconds": args.sample_seconds,
        "integration_seconds": args.integration_seconds,
        "truth_model": ["Earth/Moon/Sun point gravity", "Earth J2", "solar radiation pressure", "finite thrust", "rocket-equation mass flow"],
        "baseline_model": ["Earth/Moon/Sun point gravity"],
        "label": "truth acceleration minus baseline acceleration at the truth state",
        "rows": len(data), "columns": list(data.columns), "trajectories": arc_manifests,
        "warning": "Synthetic data supports model development; Artemis remains an external validation set and must not be used to tune this dataset.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(data)} rows to {output}")
    print(f"wrote provenance and burn schedules to {manifest_path}")


if __name__ == "__main__":
    main()
