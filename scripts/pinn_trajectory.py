#!/usr/bin/env python3
"""Build a trajectory from PINN predictions.

baseline  = reference - actual_residual      (what the analytic model gave)
corrected = baseline  + predicted_residual   (baseline + PINN correction)
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--reference", required=True)    # t,x,y,z,vx,vy,vz (m, m/s)
ap.add_argument("--predictions", required=True)  # PINN CSV
ap.add_argument("--model", choices=("pinn", "mlp", "sindy"), default="pinn")
ap.add_argument("--anchor", action="store_true",
                help="DIAGNOSTIC: subtract the t=0 prediction so the path starts on the baseline")
ap.add_argument("--out", default="corrected_trajectory.csv")
a = ap.parse_args()

ref = pd.read_csv(a.reference)
pr = pd.read_csv(a.predictions)
df = ref.merge(pr, on="t", how="inner", suffixes=("", "_p"))
if len(df) < 2:
    raise SystemExit("no matching timestamps between reference and predictions")
print(f"{len(df)} aligned rows (reference {len(ref)}, predictions {len(pr)})")

if a.model == "pinn":
    suffix = "_pred"
elif a.model == "mlp":
    suffix = "_mlp_pred"
else:
    suffix = "_sindy_pred"
truth = df[["x", "y", "z"]].values
actual = df[["drx", "dry", "drz"]].values
pred = df[[f"dr{k}{suffix}" for k in "xyz"]].values
if a.anchor:
    pred = pred - pred[0]

baseline = truth - actual
corrected = baseline + pred

t = df["t"].values
vel = np.column_stack([np.gradient(corrected[:, i], t) for i in range(3)])
out = pd.DataFrame({"t": t, "x": corrected[:, 0], "y": corrected[:, 1], "z": corrected[:, 2],
                    "vx": vel[:, 0], "vy": vel[:, 1], "vz": vel[:, 2]})
out.to_csv(a.out, index=False)

def rms_km(v):
    return np.sqrt((v ** 2).sum(1).mean()) / 1000

print(f"baseline  vs Horizons: RMS {rms_km(baseline - truth):10.1f} km")
print(f"corrected vs Horizons: RMS {rms_km(corrected - truth):10.1f} km  (model={a.model}, anchor={a.anchor})")
print(f"residual fit error |pred - actual|: RMS {rms_km(pred - actual):10.1f} km")
print("wrote", a.out)