# Data and Cosmos scripts

Run commands from the repository root. The scripts create the inputs used by
the orbital residual model and the Cosmos rollout pipeline. Most outputs go
under `data/`; preserve the raw files and their `.meta.json` sidecars so the
generated data can be traced back to its source.

## SPICE reference pipeline (offline)

SPICE kernels (e.g. `de440.bsp` + `naif0012.tls`) can replace the Horizons
download. The output feeds the same `convert_reference.py` (Moon mode) or a
generic body conversion:

```bash
export SPICE_KERNELS="/path/to/de440.bsp:/path/to/naif0012.tls"
uv sync --extra spice
uv run --extra spice python scripts/fetch_spice.py \
  --target MOON --center EARTH \
  --start "2022-11-16" --stop "2022-11-17" --step 600 \
  --out data/raw_spice.csv
uv run --extra data python scripts/convert_reference.py \
  --raw data/raw_spice.csv --mode moon --out data/reference.csv
```

`make fetch-spice SPICE_START=... SPICE_STOP=...` wraps both stages. Training on
Horizons vs. SPICE data is identical downstream: both produce the canonical
`data/reference.csv` (SI, Earth-Moon barycentric frame), so `make residual` and
`ml.pinn.run` need no changes. SPICE differs only in acquisition: offline,
kernel-version-dependent, and higher precision; it can also target spacecraft
(`--target -1024` etc.) with `--mode generic` (no barycentric split).

## Moon reference pipeline

The standard pipeline downloads a short Moon ephemeris arc, converts it into
the project’s canonical SI/barycentric frame, compares it with the numerical
propagator, and trains/evaluates the residual PINN:

```bash
uv run --extra data python scripts/fetch_horizons.py \
  --target 301 --start "2022-11-16 00:00" --stop "2022-11-17 00:00" \
  --step 10m --out data/raw_moon.csv
uv run --extra data python scripts/convert_reference.py \
  --raw data/raw_moon.csv --out data/reference.csv \
  --manifest data/manifest.json
cargo run -p residual --example residual_pipeline -- \
  data/reference.csv data/residual_table.csv
uv run --extra gpu --extra data python -m ml.pinn.run \
  --data data/residual_table.csv --out ml/pinn/pinn_result.json \
  --predictions-out ml/pinn/pinn_predictions.csv
```

`make fetch-data` runs the first two stages with the Makefile’s configured
arguments; `make residual` creates `data/residual_table.csv` after fetching a
reference if needed. The residual CSV contains `t`, position residuals
`drx,dry,drz`, and velocity residuals `dvx,dvy,dvz`. The PINN command writes
metrics to JSON and a row-aligned CSV containing actual residuals, predicted
position residuals (`drx_pred,dry_pred,drz_pred`), and the actual velocity
residuals needed to reconstruct the baseline state.

## Cosmos action-vector generation

PINN position residuals are not Cosmos action vectors. `pinn_to_cosmos.py`
uses those residual predictions to reconstruct a corrected trajectory, then
calls the Rust `posebuilder` example. Posebuilder converts camera poses into
the Cosmos camera-pose action domain: 9 values per step (translation plus
6D rotation). This keeps orbital units and Cosmos action semantics separate.

First generate the action/job files only:

```bash
uv run --extra data python scripts/pinn_to_cosmos.py \
  --reference data/reference.csv \
  --predictions ml/pinn/pinn_predictions.csv \
  --out-dir jobs/pinn-cosmos \
  --frames 121 --fps 24 --pointing earth
```

This writes `jobs/pinn-cosmos/corrected_trajectory.csv`, `actions.json`, and
`job.json`. Predictions must align exactly to reference timestamps and cover
the selected window. The adapter rejects missing timestamps rather than
silently interpolating across gaps. Use 17–401 poses (16–400 action rows) for a
Cosmos action rollout.

To submit to a running vLLM-Omni server, provide a starting image and the
server base URL (for example, `http://localhost:8000`). The endpoint is
`/v1/videos/sync`; set `COSMOS_API_KEY` if the server requires a bearer token.

```bash
uv run --extra data python scripts/pinn_to_cosmos.py \
  --reference data/reference.csv \
  --predictions ml/pinn/pinn_predictions.csv \
  --out-dir jobs/pinn-cosmos --frames 121 --fps 24 \
  --first-frame render/frames/frame_0000.png \
  --serve-url http://localhost:8000 \
  --output-video jobs/pinn-cosmos/cosmos_pinn.mp4
```

The request sends `extra_params` with `action_mode=forward_dynamics`,
`domain_name=camera_pose`, `raw_action_dim=9`, `action_chunk_size`, and the
action rows. Confirm the served model’s action-domain normalization and API
configuration match the posebuilder contract before interpreting a generated
video as a physical simulation. The initial image supplies visual context;
the actions specify camera motion.

## Script reference

| File | Purpose and outputs |
| --- | --- |
| `fetch_horizons.py` | Downloads a JPL Horizons vector ephemeris. Writes a CSV in km and km/s plus a `.meta.json` query/provenance sidecar. Supports Moon and spacecraft targets through `--target`; choose the center/window/step explicitly for reproducibility. |
| `convert_reference.py` | Converts raw geocentric Moon vectors into the canonical barycentric solver frame. Writes SI `t,x,y,z,vx,vy,vz` CSV and a manifest. By default normalizes separation to the solver’s nominal Earth–Moon distance; `--preserve-distance` keeps measured distance. |
| `convert_spacecraft.py` | Converts paired raw spacecraft and Moon ephemerides into a canonical spacecraft reference CSV. Input ephemerides are km and km/s; output is SI. The spacecraft and Moon files must cover the same mission arc and cadence. |
| `fetch_artemis.py` | Convenience pipeline for Artemis I or II: fetches spacecraft and Moon Horizons arcs, then runs `convert_spacecraft.py`. Writes `data/raw_artemisN.csv`, `data/raw_moon_artemisN.csv`, and `data/reference_artemisN.csv`. Override the mission date range with `--start` and `--stop`. |
| `mock_cosmos_worker.py` | Local mock worker for exercising the repository’s Cosmos job/action file contract. It does not load Cosmos or generate model videos; use it only for plumbing/demo checks. |
| `pinn_to_cosmos.py` | Reconstructs a PINN-corrected trajectory, runs Rust posebuilder, and optionally submits the 9D actions to vLLM-Omni. Writes corrected trajectory and Cosmos job/action JSON; with `--serve-url`, also writes the returned video. |

## Artemis data

For an Artemis trajectory, fetch and convert the mission arc first:

```bash
uv run --extra data python scripts/fetch_artemis.py --mission 1
cargo run -p residual --example residual_pipeline -- \
  data/reference_artemis1.csv data/residual_artemis1.csv
```

For Artemis II, use `--mission 2`. Its default mission dates may be future
relative to current Horizons coverage; supply a supported `--start`/`--stop`
window if the query is rejected. Train on the resulting residual file with
`--data data/residual_artemis1.csv` and write a mission-specific prediction
CSV using `--predictions-out`.

## Data quality notes

`pinn_result.json` contains scores, not per-sample predictions. The prediction
CSV is required by the Cosmos adapter. PINN scores should be reviewed before
using the output; a low or negative held-out R² means the trajectory is an
experimental visualization input, not a validated physical correction. The
Cosmos video is a generated visualization conditioned on actions and a first
frame, not an independent orbital solver.
