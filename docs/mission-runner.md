# Live mission runner

The viewer can propagate a mission from an initial state vector instead of
replaying the static `tauri-src/src/data/trajectory.json` bundle. This document
describes the contract, the frame handling, and the known approximations.

## Layers

| Layer | Responsibility |
| --- | --- |
| `crates/mission` | Pure Rust: validate a `MissionRequest`, run `dynamics::propagate_adaptive_detailed`, resample, return a `MissionTrajectory`. No Tauri or file I/O. |
| `tauri-src/src-tauri` | `propagate_mission` command; runs the pure crate on a blocking worker. |
| `tauri-src/src/main.js` | Mission form, `localStorage` persistence, presets, and the shared `applyTrajectoryBundle` renderer. |

Because the mission crate has no UI or I/O dependencies, its tests run headless
with `cargo test -p mission`.

## Request

`propagate_mission` accepts a single `request` argument:

```json
{
  "name": "LEO demo",
  "center": "earth",
  "position_km": [6778.137, 0.0, 0.0],
  "velocity_km_s": [0.0, 7.6686, 0.0],
  "duration_s": 7200.0,
  "samples": 1200,
  "start_epoch_s": 0.0,
  "burns": [
    { "t_s": 600.0, "dv_m_s": [0.0, 1000.0, 0.0] }
  ]
}
```

- `center` is `"earth"` (default) or `"barycentric"`.
- `samples` defaults to 1200, must be between 2 and 20,000.
- `duration_s` must be positive and at most 400 days.
- `start_epoch_s` defaults to 0.
- `burns` are relative to the start epoch; `t_s` must fall inside the mission.
  At most 8 burns are accepted (the `dynamics` limit), including any burns in
  the optional `solver` override.
- `solver` is an optional `SolverConfig`; omitted fields use the dynamics
  defaults.

## Response

The response deliberately reuses the static bundle's shape so one renderer
handles both:

```json
{
  "schema": "mission-trajectory/v1",
  "name": "LEO demo",
  "frame": "earth-centered-inertial",
  "units": "km",
  "points": [[6778.137, 0.0, 0.0], "..."],
  "times": [0.0, "..."],
  "times_absolute": [0.0, "..."],
  "count": 1200,
  "duration_s": 7200.0,
  "start_epoch_s": 0.0,
  "initial_state": { "position_km": [], "velocity_km_s": [], "radius_km": 0, "speed_km_s": 0 },
  "final_state": { "position_km": [], "velocity_km_s": [], "radius_km": 0, "speed_km_s": 0 },
  "stats": {
    "samples": 1200,
    "min_radius_km": 0,
    "max_radius_km": 0,
    "final_radius_km": 0,
    "min_speed_km_s": 0,
    "max_speed_km_s": 0,
    "accepted_steps": 0,
    "rejected_steps": 0
  }
}
```

`times` are seconds from the start of the mission (the scrubber's time base);
`times_absolute` are the solver epochs. `points` are kilometres in the requested
frame.

## Frames and the Earth offset

The solver is Earth–Moon barycentric. For `center: "earth"` the initial state
is shifted by the analytic Earth position at `start_epoch_s`; each output sample
is shifted back by the Earth position at that sample's absolute time. The first
sample therefore reproduces the requested state exactly, and Earth sits at the
origin in the viewer.

The conversion does **not** subtract the Earth's barycentric velocity (about
12 m/s in the analytic ephemeris). That is far below the circular-ephemeris
error and is left as a documented approximation.

For `center: "barycentric"` the state is used verbatim and the response frame is
`earth-moon-barycentric`; the viewer draws the corresponding barycentric Earth
and Moon markers.

## Sampling

The adaptive Dormand–Prince integrator uses `rtol = 1e-9` and its own step
sequence. The result is resampled onto a uniform grid with cubic Hermite
interpolation using position and velocity, which keeps the curve smooth even
across long coast steps.

## Viewer behaviour

- A live mission draws a cyan **Craft** marker at the current sample; the Moon
  marker is hidden because the moving body is now the spacecraft.
- The 2D guide uses the Earth-centred frame (Earth at the origin), computes the
  analytic Moon offset, and includes the Moon in the guide bounds only when it
  is within roughly 1.6 path radii.
- PINN/SINDy overlays and the Monte-Carlo ensemble are tied to the static arc,
  so their controls are disabled while a live mission is displayed.
- **Load bundle** returns to the bundled reference. If Tauri IPC is unavailable
  (plain browser), **Run mission** reports that the desktop backend is needed.

## Known limitations

- Analytic circular Earth–Moon ephemeris only; SPICE/Horizons kernels are not
  read from Rust.
- Impulsive burns only; no finite thrust, atmosphere, launch, or attitude model.
- Earth-centred inputs keep the Earth's barycentric velocity approximation
  described above.
- The result is a display trajectory, not a mission-design optimiser; it does
  not target or correct an arrival state.

## Tests

```bash
cargo test -p mission
cargo fmt --check
```

The mission tests cover the exact initial state, a circular LEO stay, a
prograde burn raising apogee, translunar distance, barycentric pass-through,
invalid inputs, and uniform resampling.
