#!/usr/bin/env python3
"""Compare gated acceleration-model and point-gravity RK4 free rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from train_satellite_path_model import features, score

MU_EARTH = 3.986004418e14
MU_MOON = 4.9048695e12
MU_SUN = 1.32712440018e20
G0 = 9.80665
AU = 149_597_870_700.0
EARTH_RADIUS = 6_378_136.3
EARTH_J2 = 1.08262668e-3
MOON_RADIUS = 1_737_400.0
MOON_J2 = 2.03263e-4
SOLAR_PRESSURE = 4.56e-6
TOTAL_MU = MU_EARTH + MU_MOON


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else np.zeros(3)


def j2(r, center, mu, radius, coeff):
    q = r-center
    d = np.linalg.norm(q)
    z = q[2]/d
    c = 1.5*coeff*mu*radius**2/d**5
    return c*np.array([q[0]*(5*z*z-1), q[1]*(5*z*z-1), q[2]*(5*z*z-3)])


def load_ensemble(checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    n_in, h1, h2, n_out = ckpt["architecture"]
    models = []
    for state in ckpt["members"]:
        model = nn.Sequential(nn.Linear(n_in, h1), nn.Tanh(), nn.Linear(h1, h2), nn.Tanh(), nn.Linear(h2, n_out))
        model.load_state_dict(state); model.eval(); models.append(model)
    return ckpt, models


class ArcEnvironment:
    def __init__(self, frame, burns=None):
        self.frame = frame.sort_values("t_s").reset_index(drop=True)
        self.times = self.frame.t_s.to_numpy(dtype=float)
        self.burns = burns or []
        self.center = str(self.frame.center.iloc[0])
        self.initial_mass = float(self.frame.get("initial_mass_kg", self.frame.mass_kg).iloc[0])
        self.dry_mass = 0.995*self.initial_mass
        self.area = float(self.frame.area_m2.iloc[0]) if "area_m2" in self.frame else 10.0
        self.cr = float(self.frame.cr.iloc[0]) if "cr" in self.frame else 1.3

    def row(self, t, state):
        t = float(np.clip(t, self.times[0], self.times[-1]))
        row = self.frame.iloc[0].copy()
        row["t_s"] = t
        for prefix, axes in (("earth", "xyz"), ("moon", "xyz"), ("sun", "xyz")):
            for axis in axes:
                for suffix in ("_m", "_mps") if prefix != "sun" else ("_m",):
                    name = f"{prefix}_{axis}{suffix}"
                    if name in self.frame:
                        row[name] = np.interp(t, self.times, self.frame[name].to_numpy(dtype=float))
        for axis, value in zip("xyz", state[:3]): row[f"{axis}_m"] = value
        for axis, value in zip("xyz", state[3:]): row[f"v{axis}_mps"] = value
        mass = self.mass_at(t)
        row["mass_kg"], row["area_m2"], row["cr"], row["dt_s"] = mass, self.area, self.cr, 600.0
        return row

    def mass_at(self, t):
        spent = sum(b["force_n"]*max(0.0, min(t, b["stop_s"])-b["start_s"])/(b["isp_s"]*G0) for b in self.burns)
        return max(self.dry_mass, self.initial_mass-spent)

    def control(self, t):
        total_force = np.zeros(3)
        for burn in self.burns:
            if burn["start_s"] <= t < burn["stop_s"]:
                total_force += burn["force_n"]*np.asarray(burn["direction"])
        return total_force/self.mass_at(t)

    def point_gravity(self, r, env):
        earth = env[[f"earth_{a}_m" for a in "xyz"]].to_numpy(dtype=float)
        moon = env[[f"moon_{a}_m" for a in "xyz"]].to_numpy(dtype=float)
        sun = env[[f"sun_{a}_m" for a in "xyz"]].to_numpy(dtype=float)
        re, rm, rs = r-earth, r-moon, r-sun
        return (-MU_EARTH*re/np.linalg.norm(re)**3 - MU_MOON*rm/np.linalg.norm(rm)**3
                - MU_SUN*(rs/np.linalg.norm(rs)**3 + sun/np.linalg.norm(sun)**3))

    def correction(self, state, t, checkpoint, models):
        row = self.row(t, state)
        x = features(pd.DataFrame([row]))
        z = (x-np.asarray(checkpoint["feature_mean"]))/np.asarray(checkpoint["feature_scale"])
        ood = float(np.max(np.abs(z)))
        xt = torch.tensor(z, dtype=torch.float32)
        with torch.no_grad():
            pred = np.stack([m(xt).numpy()[0] for m in models])
        pred = pred*np.asarray(checkpoint["target_scale"])+np.asarray(checkpoint["target_mean"])
        mean = pred.mean(axis=0)
        uncertainty = float(np.sqrt(np.mean(np.var(pred, axis=0))))
        rejected = (ood > checkpoint["feature_ood_gate_threshold_z"] or
                    uncertainty > checkpoint["uncertainty_gate_threshold_mps2"])
        return (np.zeros(3) if rejected else mean), rejected, uncertainty, ood

    def derivative(self, t, state, checkpoint, models, learned):
        row = self.row(t, state)
        acceleration = self.point_gravity(state[:3], row) + self.control(t)
        rejected, uncertainty, ood = False, 0.0, 0.0
        if learned:
            correction, rejected, uncertainty, ood = self.correction(state, t, checkpoint, models)
            acceleration += correction
        return np.r_[state[3:], acceleration], rejected, uncertainty, ood

    def integrate(self, state, t, dt, checkpoint, models, learned):
        def f(tt, ss): return self.derivative(tt, ss, checkpoint, models, learned)[0]
        k1 = f(t, state); k2 = f(t+dt/2, state+dt*k1/2)
        k3 = f(t+dt/2, state+dt*k2/2); k4 = f(t+dt, state+dt*k3)
        result = state+dt*(k1+2*k2+2*k3+k4)/6
        _, rejected, uncertainty, ood = self.derivative(t+dt/2, (state+result)/2, checkpoint, models, learned)
        return result, rejected, uncertainty, ood


def horizon_metrics(actual, predicted, horizon):
    return {"horizon_s": float(horizon), **score(np.asarray([actual]), np.asarray([predicted])),
            "position_error_m": float(np.linalg.norm(actual[:3]-predicted[:3]))}


def rollout_horizons(env, initial, checkpoints, step_seconds, horizons):
    results = {"point_gravity": {}, "learned_gated": {}}
    states = {"point_gravity": initial.copy(), "learned_gated": initial.copy()}
    current_t = float(env.times[0]); desired = sorted(h for h in horizons if h <= env.times[-1]-current_t)
    rejected_steps = 0; total_steps = 0; max_uncertainty = 0.0; max_ood = 0.0
    for horizon in desired:
        while current_t < env.times[0]+horizon-1e-6:
            dt = min(step_seconds, env.times[0]+horizon-current_t)
            for key, learned in (("point_gravity", False), ("learned_gated", True)):
                states[key], rejected, uncertainty, ood = env.integrate(states[key], current_t, dt, *checkpoints, learned)
                if key == "learned_gated":
                    total_steps += 1; rejected_steps += int(rejected)
                    max_uncertainty = max(max_uncertainty, uncertainty); max_ood = max(max_ood, ood)
            current_t += dt
        idx = int(np.argmin(np.abs(env.times-(env.times[0]+horizon))))
        truth = env.frame.loc[idx, ["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]].to_numpy(dtype=float)
        for key in results:
            results[key][str(int(horizon))] = horizon_metrics(truth, states[key], horizon)
    results["gate"] = {"steps": total_steps, "rejected_steps": rejected_steps,
                        "rejected_fraction": rejected_steps/total_steps if total_steps else 0.0,
                        "max_ensemble_uncertainty_mps2": max_uncertainty, "max_feature_z": max_ood}
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/synthetic_satellite_trajectories.csv")
    ap.add_argument("--manifest", default="data/synthetic_satellite_trajectories.manifest.json")
    ap.add_argument("--checkpoint", default="ml/pinn/satellite_dynamics_model/satellite_accel_model.pt")
    ap.add_argument("--step-seconds", type=float, default=60.0)
    ap.add_argument("--lro", default="data/reference_lro_artemis2_window_preserve.csv")
    ap.add_argument("--moon", default="data/reference_artemis2_preserve.csv")
    ap.add_argument("--sun", default="data/reference_sun_artemis2.csv")
    ap.add_argument("--out", default="ml/pinn/satellite_dynamics_model/evaluation.json")
    args = ap.parse_args()
    checkpoint, models = load_ensemble(args.checkpoint)
    data = pd.read_csv(args.data)
    manifest = json.loads(Path(args.manifest).read_text())
    burn_map = {t["arc_id"]: t.get("burns", []) for t in manifest["trajectories"]}
    test_arcs = checkpoint["test_arcs"]
    horizons = [3600.0, 21600.0, 86400.0, 172800.0]
    synthetic = {}
    for arc, group in data[data.arc_id.isin(test_arcs)].groupby("arc_id"):
        env = ArcEnvironment(group, burn_map.get(arc, []))
        initial = group.iloc[0][["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]].to_numpy(dtype=float)
        synthetic[arc] = rollout_horizons(env, initial, (checkpoint, models), args.step_seconds, horizons)

    result = {"synthetic_grouped_test_rollouts": synthetic,
              "interpretation": "Physics-only is point gravity plus known synthetic thrust. Learned adds a gated ensemble correction for J2/SRP."}
    if all(Path(p).exists() for p in (args.lro, args.moon, args.sun)):
        lro = pd.read_csv(args.lro).rename(columns={"t":"t_s", "x":"x_m", "y":"y_m", "z":"z_m",
                                                      "vx":"vx_mps", "vy":"vy_mps", "vz":"vz_mps"})
        moon_raw = pd.read_csv(args.moon)
        moon = moon_raw.rename(columns={"t":"t_s", "x":"moon_x_m", "y":"moon_y_m", "z":"moon_z_m",
                                         "vx":"moon_vx_mps", "vy":"moon_vy_mps", "vz":"moon_vz_mps"})
        sun = pd.read_csv(args.sun).rename(columns={"t":"t_s", "x":"sun_x_m", "y":"sun_y_m", "z":"sun_z_m"})
        lro = lro.merge(moon, on="t_s").merge(sun, on="t_s")
        frac = MU_MOON/TOTAL_MU
        for axis in "xyz":
            lro[f"earth_{axis}_m"] = -frac*lro[f"moon_{axis}_m"]
            lro[f"earth_v{axis}_mps"] = -frac*lro[f"moon_v{axis}_mps"]
        lro["center"], lro["mass_kg"], lro["initial_mass_kg"] = "moon", 1000.0, 1000.0
        lro["area_m2"], lro["cr"] = 10.0, 1.3
        env = ArcEnvironment(lro, [])
        initial = lro.iloc[0][["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]].to_numpy(dtype=float)
        ext_horizons = [h for h in horizons if h <= env.times[-1]]
        ext_horizons += [env.times[-1]] if env.times[-1] not in ext_horizons else []
        result["lro_external_rollout"] = rollout_horizons(env, initial, (checkpoint, models), args.step_seconds, ext_horizons)
        result["lro_assumptions"] = "No LRO maneuver history is applied; mass=1000 kg, area=10 m^2, Cr=1.3 are placeholders. This is a ballistic diagnostic, not mission-accurate validation."
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2)); print(f"wrote {out}")


if __name__ == "__main__": main()
