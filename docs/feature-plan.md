# Feature Engineering Plan

## Problem
`ml/pinn/run.py` on `data/residual_table.csv` uses raw `t` (seconds since epoch, ~1e8) as a feature.
The feature grid is fixed at `(-2, 2)` and basis functions blow up on unscaled/monotonic
inputs, producing NaN metrics. The residual table only contains `t` plus residual
deltas (`drx..dvz`), so there is no natural state feature set.

## Goal
Replace raw `t` with well-scaled, physically meaningful features derived from time,
so PINN/MLP/SINDy train stably and the held-out split still measures generalisation
rather than memorising the time index.

## Proposed features (derived from `t` and the row index)
1. `t_norm` — `(t - t_mean) / t_std` (z-scored time, O(1) scale).
2. `phase` — `2*pi*(t - t0)/T_orbit` wrapped to `[-pi, pi]` (lunar orbital phase).
3. `sin_phase`, `cos_phase` — circular encoding of phase (avoids the wrap discontinuity).
4. `t_scaled` — `t / 1e5` as a gentle linear proxy (keeps a monotonic trend if useful).

The target remains the residual vector (`drx, dry, drz, dvx, dvy, dvz`).

`masked_fraction`, burn masking, and `split_by_time` are unchanged.

## Scope
- `ml/pinn/features.py` (new): `engineer_features(df, time_col)` returns a DataFrame with
  the derived columns appended (or a feature list). Pure stdlib + pandas (already in data extra).
- `ml/pinn/dataset.py`: add `engineer_features` import path; `build_xy` unchanged.
- `ml/pinn/run.py`: add `--engineer` flag (default on when a time column exists and
  no `--features` given) that applies the engineering and reports which features were used.
  Keep `--features` override for manual control.
- Tests: `tests/test_features.py` covering determinism, scaling, and no-NaN.

## Agents
- Lane A (features.py + dataset wiring) — implement engineering.
- Lane B (run.py CLI integration) — flag + auto-apply + reporting.
- Lane C (tests) — unit tests for features.

Lanes A and B share run.py? Lane B touches run.py only; Lane A touches features.py + dataset.py.
Lane C touches tests only. No write-scope overlap.
EOF
echo written