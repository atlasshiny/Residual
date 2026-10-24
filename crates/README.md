# Rust core

| Crate | Role |
|---|---|
| `dynamics` | Pure-Rust RK4 propagator mirroring the CUDA physics (fallback + tests). |
| `frames` | Time and inertial-frame newtypes. |
| `ingest` | Reference telemetry CSV → validated `State`s with linear resampling. |
| `residual` | `reference - propagated` rows, burn masking, regular-grid resampling. |
| `posebuilder` | State + pointing law → camera pose and relative `ΔT` actions. |
| `evaluator` | ΔT integration, Sim(3)/Umeyama alignment, pose metrics. |
| `schema` | Mission, vehicle, burn-window, and solver JSON schemas. |
| `ipc` | Directory-backed Cosmos job queue and artifact manifests. |

## Quick check

```sh
cargo test --workspace
```

## End-to-end residual example

```sh
cargo run -p residual --example residual_pipeline -- ref.csv residuals.csv [mission.json]
```

`ref.csv` uses the canonical `t,x,y,z,vx,vy,vz` header. The example propagates
from the first reference state, resamples the trajectory at each reference
epoch, and writes `t,drx,dry,drz,dvx,dvy,dvz`. Burn windows from the optional
mission file are dropped from the output.

The generated residual CSV can be loaded directly by `ml.kan.dataset`:

```python
from ml.kan.dataset import load_residual_table, build_xy
df = load_residual_table("residuals.csv")
X, Y = build_xy(df, feature_cols="t,drx,dry,drz", target_cols="dvx,dvy,dvz")
```

## Cosmos pose/action pipeline

```sh
cargo run -p posebuilder --example pose_pipeline -- \
  ref.csv jobs/demo --job-id demo --mode forward_dynamics \
  --t0 0 --t1 5 --fps 24 --frames 120 --scale-k 1000 --pointing earth

python scripts/mock_cosmos_worker.py jobs/demo
```

The first command writes the §6.4 job manifest (`job.json`) and 9D action rows
(`actions.json`); the second writes `sample_outputs.json` and the ipc
`manifest.json`. The full schema and the two accepted job shapes are in
[`docs/cosmos-contract.md`](../docs/cosmos-contract.md). The ipc example
`enqueue_job` covers the native queue form:

```sh
cargo run -p ipc --example enqueue_job -- jobs demo inverse_dynamics clip.mp4
```
