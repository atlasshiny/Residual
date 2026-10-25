#!/usr/bin/env python3
"""Bundle Horizons truth, Moon, and corrected paths into viewer_data.json (km).

python scripts/export_viewer_data.py --reference data/reference_artemis2.csv \
  --moon data/raw_moon_artemis2.csv --path mlp=out/corrected_mlp.csv \
  --path pinn=out/corrected_pinn.csv --path sindy=out/corrected_sindy.csv \
  --out viewer/viewer_data.json
"""
import argparse, json, os
import numpy as np, pandas as pd

MU_E, MU_M, D_NOM = 3.986004418e14, 4.9048695e12, 3.844e8
ap = argparse.ArgumentParser()
ap.add_argument("--reference", required=True)
ap.add_argument("--moon", required=True)
ap.add_argument("--path", action="append", default=[], help="name=csv")
ap.add_argument("--sun", default=None, help="geocentric Sun CSV (Horizons target 10, same columns as the Moon file, km)")
ap.add_argument("--step", type=int, default=3, help="keep every Nth row")
ap.add_argument("--out", default="viewer/viewer_data.json")
a = ap.parse_args()

ref = pd.read_csv(a.reference)
t = ref["t"].values
moon = pd.read_csv(a.moon)
m = moon[["x", "y", "z"]].values * 1000.0
v0 = moon[["vx", "vy", "vz"]].values[0] * 1000.0
xh = m[0] / np.linalg.norm(m[0])
zh = np.cross(m[0], v0); zh /= np.linalg.norm(zh)
R = np.vstack([xh, np.cross(zh, xh), zh])
off = -D_NOM * MU_M / (MU_E + MU_M)
M = m @ R.T + np.array([off, 0, 0])
Mi = np.column_stack([np.interp(t, moon["t"].values, M[:, i]) for i in range(3)])

paths = {"truth": ref[["x", "y", "z"]].values}
for spec in a.path:
    name, f = spec.split("=", 1)
    c = pd.read_csv(f)
    paths[name] = np.column_stack([np.interp(t, c["t"], c[k]) for k in "xyz"])

s = slice(None, None, a.step)
km = lambda arr: (arr[s] / 1000.0).round(2).tolist()
data = {"t": t[s].round(1).tolist(), "earth": [round(off / 1000.0, 2), 0, 0],
        "moon": km(Mi), "paths": {k: km(v) for k, v in paths.items()}}
if a.sun:
    sn = pd.read_csv(a.sun)
    S = sn[["x", "y", "z"]].values @ R.T          # same rotation as the Moon; direction only
    Si = np.column_stack([np.interp(t, sn["t"].values, S[:, i]) for i in range(3)])
    Si /= np.linalg.norm(Si, axis=1, keepdims=True)
    data["sun_dir"] = Si[s].round(5).tolist()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(data, open(a.out, "w"))
print("wrote", a.out, len(data["t"]), "samples,", list(paths))