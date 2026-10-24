# Cosmos file contract

The boundary between the Rust core and the Python Cosmos workers is a directory
of plain files. There is no RPC and no shared database: the Rust side writes a
job, a worker writes its outputs plus a manifest, and every stage can be
inspected or replayed after a failure.

This document is the authoritative contract for that boundary. It implements
ARCHITECTURE.md §6 and §7; where the architecture's illustrative JSON shows a
placeholder, the concrete choice is recorded here.

## 1. Directory layout

```
<jobs-root>/
└─ <job-id>/                     # job-id: [A-Za-z0-9._-], not starting with '.'
   ├─ job.json                   # input: job request / pose manifest
   ├─ actions.json               # optional: 9D action rows (written by posebuilder)
   ├─ sample_outputs.json        # output: pose sequence (written by a worker)
   └─ manifest.json              # output: artifact manifest (written by a worker)
```

* A job id is also a directory name. The Rust `ipc::JobStore` rejects ids that
  could escape the root (`../`, absolute paths, leading `.`).
* Input paths inside `job.json` are relative to the job directory unless
  absolute. Workers must resolve relative paths against the **job directory**,
  not their own working directory.
* Outputs are written in the same job directory by default. The mock worker
  also accepts `--out <dir>` for tests.
* `job.json` is immutable once enqueued. A worker must not rewrite it.

## 2. Two accepted `job.json` shapes

The worker accepts a superset so the native queue and the pose pipeline can
share one layout. Both use the same `mode` vocabulary and the same output
files.

### 2.1 Native ipc job (`crates/ipc`)

```json
{
  "id": "flyby_w03",
  "mode": "inverse_dynamics",
  "inputs": ["clip.mp4"],
  "seed": 0,
  "params": null,
  "created_at_s": 1790000000
}
```

`mode` is one of `transfer`, `inverse_dynamics`, `forward_dynamics`.
Mode-specific values (`frames`, `fps`, `scale_k_km_per_m`, `actions`, ...) may
live in the free-form `params` object; top-level fields of the same name take
precedence. Enqueue one with:

```sh
cargo run -p ipc --example enqueue_job -- jobs flyby_w03 inverse_dynamics clip.mp4
```

### 2.2 Pose pipeline manifest (`crates/posebuilder`)

This is the ARCHITECTURE.md §6.4 job manifest, produced from a trajectory CSV:

```sh
cargo run -p posebuilder --example pose_pipeline -- \
  trajectory.csv jobs/flyby_w03 \
  --job-id flyby_w03 --mode forward_dynamics --model Cosmos3-Nano \
  --t0 1000 --t1 1005 --fps 24 --frames 120 \
  --scale-k 1000 --pointing earth --resolution 704x1280 --seed 0
```

The second positional argument is normally a job directory. If it ends in
`.json`, the manifest is written to that file and `actions.json` is written
next to it. Resulting `job.json`:

```json
{
  "schema": "cosmos-job/v1",
  "job_id": "flyby_w03",
  "id": "flyby_w03",
  "mode": "forward_dynamics",
  "model": "Cosmos3-Nano",
  "window": {
    "t0_tdb": 1000.0,
    "t1_tdb": 1005.0,
    "time_compression": 1.0084033613445378
  },
  "scale_k_km_per_m": 1000.0,
  "pointing_law": "earth",
  "fps": 24,
  "frames": 120,
  "resolution": [704, 1280],
  "inputs": {
    "control_video": null,
    "first_frame": null,
    "actions": "actions.json",
    "prompt_json": null
  },
  "seed": 0,
  "time_system": "TDB",
  "time_unit": "s",
  "frame_times_tdb_s": [1000.0, 1000.041666, "..."],
  "action_dim": 9,
  "action_encoding": "translation_xyz + rot6d_first_two_columns",
  "action_count": 119,
  "first_pose": {
    "rotation": [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    "translation_km": [7000.0, 0.0, 0.0]
  },
  "actions": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], "..."],
  "units": {
    "trajectory_position": "m",
    "pose_translation": "km",
    "action_translation": "cosmos_m",
    "rotation": "6d_zhuo2019_first_two_columns"
  }
}
```

`id` and `job_id` are both present, and `job_id` wins when both exist, so the
file is readable by both the native queue and a §6.4 consumer. `t0_tdb`,
`t1_tdb` and `frame_times_tdb_s` are numeric TDB **seconds**; the CSV `t` column
is the time base, and the manifest names the unit explicitly. The §6.4 template
shows quoted strings for readability only.

For the native ipc shape, `inputs` is a list of strings. For the pose manifest,
`inputs` is the §6.4 object. Workers must branch on the JSON type.

## 3. Inputs

### 3.1 Trajectory CSV

Canonical header, matched case-insensitively by `ingest::Telemetry`:

```csv
t,x,y,z,vx,vy,vz
0,7000000,0,0,0,7546.05,0
```

* `t`: seconds (TDB by convention), strictly increasing.
* `x,y,z`: metres, fixed inertial frame (the frame is not encoded in the CSV;
  pass it in mission metadata if it matters).
* `vx,vy,vz`: metres per second.

### 3.2 Window, sampling, and time compression

* `t0_tdb`/`t1_tdb` select the simulation window. They must lie inside the CSV
  range, with `t1 > t0`.
* Frame times are
  `t_i = t0 + (i / fps) * time_compression`, for `i = 0..frames-1`.
* When `--time-compression` is omitted, it is computed as
  `(t1 - t0) * fps / (frames - 1)`, so the final frame lands exactly on `t1`.
  When supplied, the clip may cover a sub-interval; the manifest still reports
  the requested window.
* States are linearly interpolated from the CSV at each frame time.
* `frames` is 2 or more. There are `frames - 1` relative actions because every
  action is a transition.

### 3.3 Pointing law and camera convention

| `pointing_law` | Meaning |
|---|---|
| `earth` | Camera at the spacecraft, optical axis toward the Earth centre. |
| `moon` | Camera at the spacecraft, optical axis toward the Moon centre. |
| `data` | Request measured attitude. The canonical CSV has none, so the Rust builder falls back to Earth pointing and records that in `pointing_note`. |

Positions come from `dynamics::earth_moon_positions(t, config)`; no SPICE
kernel is read by this pipeline. The camera uses the OpenCV convention Cosmos
expects: `+Z` forward (optical axis), `+X` right, `+Y` down. The pose rotation
matrix has those camera axes as its columns in the inertial frame. Every result
must carry its pointing law; a synthetic attitude must never be presented as
measured.

### 3.4 Scale `k`

`scale_k_km_per_m` is **kilometres per Cosmos-metre**. Pose positions are in
kilometres and actions are normalized as

```
translation_action = translation_km / k
```

so `k = 1000` maps 1 Cosmos-metre to 1000 km. The default is 1000; it is a tuned
parameter and must be logged with every clip. `k` is applied only to
translation. Rotations are not scaled.

### 3.5 Action encoding

Each action is exactly 9 numbers, in this order:

```
[tx, ty, tz, r00, r10, r20, r01, r11, r21]
```

* `tx,ty,tz`: relative translation `ΔT = T_{t-1}^{-1} T_t`, in Cosmos metres
  (kilometres divided by `k`).
* `r00,r10,r20`: first **column** of the relative rotation matrix.
* `r01,r11,r21`: second column of that matrix.

This is the Zhou et al. 2019 6D representation: the first two columns of `R`,
flattened column-major. It has no discontinuity at ±180° and can be converted
back to SO(3) by Gram–Schmidt or SVD. The architecture's rotation convention is
OpenCV camera axes.

`actions.json` has this shape:

```json
{
  "schema": "cosmos-actions/v1",
  "job_id": "flyby_w03",
  "action_dim": 9,
  "action_encoding": "translation_xyz + rot6d_first_two_columns",
  "translation_unit": "cosmos_metre",
  "scale_k_km_per_m": 1000.0,
  "count": 119,
  "actions": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], "..."]
}
```

The same array is inlined in `job.json` under `actions` for inspectability. If
both exist, `job.json` wins. `first_pose` is the absolute `T_0` (rotation in the
inertial frame, translation in km); integrating `actions` from it with
`T_t = T_{t-1} · ΔT_t` reproduces every frame. The first action is therefore the
transition from frame 0 to frame 1, not an identity-to-first-frame hop.

## 4. Outputs

### 4.1 `sample_outputs.json`

Written by a worker. The mock writes it without a GPU; a real inverse-dynamics
worker writes the same shape.

```json
{
  "schema": "cosmos-sample-outputs/v1",
  "job_id": "flyby_w03",
  "mode": "inverse_dynamics",
  "model": "Cosmos3-Nano",
  "action_dim": 9,
  "action_encoding": "translation_xyz + rot6d_first_two_columns",
  "translation_unit": "cosmos_metre",
  "scale_k_km_per_m": 1000.0,
  "pointing_law": "earth",
  "fps": 24,
  "frames": 120,
  "first_pose": {
    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "translation_km": [0.0, 0.0, 0.0]
  },
  "poses": [[0.02, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], "..."],
  "mock": true,
  "note": "mock inverse-dynamics output; each row is one 9D relative action"
}
```

`poses` is the recovered action sequence, one 9D row per frame transition. It is
named `poses` to match the Cosmos framework's output file; each row uses the
same encoding as the input `actions`. The evaluator integrates the rows and
then applies Sim(3)/Umeyama alignment before reporting RRE/RTE/ATE. Translation
in space is not metric-reliable, so rotation error is the primary metric.

For `transfer` jobs there is no pose sequence; `poses` is `[]`.

### 4.2 `manifest.json`

Matches `ipc::Manifest` exactly. `kind` is one of `video`, `pose_json`,
`metrics`, `image`, `log`, `other`.

```json
{
  "job_id": "flyby_w03",
  "mode": "inverse_dynamics",
  "artifacts": [
    {
      "name": "poses",
      "path": "sample_outputs.json",
      "kind": "pose_json",
      "bytes": 1234
    }
  ],
  "created_at_s": 1790000000,
  "metadata": {
    "mock": true,
    "worker": "scripts/mock_cosmos_worker.py",
    "pose_count": 119,
    "action_source": "job"
  }
}
```

* `artifacts[].path` is relative to the job directory.
* A manifest is the completion marker: `JobStore::pending()` is defined as jobs
  without a `manifest.json`. Write it last.
* A failed worker should still leave enough information to debug (return code
  and stderr in the scheduler log). It must not write a success manifest.

## 5. Status and ownership

| Stage | Owner | Writes |
|---|---|---|
| Pose/action job | Rust `posebuilder` example | `job.json`, `actions.json` |
| Native queue entry | Rust `ipc` example / orchestrator | `job.json` |
| Mock or real run | Python worker | `sample_outputs.json`, `manifest.json` |
| Evaluation | Rust `evaluator` | metrics JSON (outside this contract) |

The Rust side must not consume `sample_outputs.json` as if it were physics. It
is model output: it goes to the evaluator and the UI only.

## 6. Local workflow (no network, no GPU)

```sh
# 1. Build the pose/action job from a trajectory CSV.
cargo run -p posebuilder --example pose_pipeline -- \
  trajectory.csv jobs/demo --job-id demo --mode inverse_dynamics \
  --t0 0 --t1 5 --fps 24 --frames 120 --scale-k 1000 --pointing earth

# 2. Produce mock worker outputs.
python scripts/mock_cosmos_worker.py jobs/demo

# 3. Inspect.
cat jobs/demo/job.json            # §6.4 manifest + actions
cat jobs/demo/sample_outputs.json # poses array
cat jobs/demo/manifest.json       # ipc manifest
```

The mock worker is deterministic for a given `seed` and job id. It echoes
`actions` when the job supplies them and otherwise synthesizes a small,
model-shaped rotation sequence, so the evaluator and UI can be exercised before
the checkpoint is staged.

## 7. Invariants a worker may rely on

1. `job.json` exists and is valid JSON before the worker starts.
2. `job_id`/`id` is a safe directory name.
3. `mode` is one of the three names above.
4. `job.json` is not modified concurrently.
5. `frames >= 2` and `action_count == frames - 1` for pose jobs.
6. Every action row has exactly 9 finite numbers.
7. `resolution` is `[width, height]`.
8. Paths in `inputs` resolve relative to the job directory.

## 8. Known limits

* `pointing_law: data` is not implemented for the canonical CSV; it falls back
  to Earth pointing and says so.
* Times are seconds in the CSV's time base, not parsed TDB calendar strings.
* `actions.json` is JSON, not `.npy`. A real runner that requires `.npy` must
  convert it; the JSON stays the reviewable source of truth.
* No normalizer values are applied by the Rust pipeline. Step 5 of §6.2
  (checkpoint normalizers) belongs to the worker and must be logged.
