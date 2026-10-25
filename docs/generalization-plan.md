# Generalization Plan

## Goal

Move the residual-correction model from fitting one 30-day Moon arc to producing
a correction that transfers across epochs, arcs, and mission-like trajectories.

"Generalized" here means:

- it works on held-out **future** samples, not just random interpolation points;
- it works on a **different trajectory arc** with the same physics;
- it works on a **different mission** (LEO, TLI, lunar flyby) without retraining;
- it reports when it is outside its training distribution instead of silently
  extrapolating.

The current model does not meet those tests. This document defines the work
needed to get there and the gates used to decide whether a change is real.

## Current bottlenecks

1. **One arc, one epoch.** Training data is a 30-day Moon arc with the Earth-Moon
   separation normalized to the circular model's radius.
2. **Time-based features.** The PINN consumes normalized time; the baselines use
   time-derived features. That makes the model epoch-specific.
3. **Random interpolation split.** The reported R² measures interpolation, not
   temporal or mission generalization. Temporal holdout has previously been
   negative.
4. **Position residual target.** Adding `dr(t)` after propagation does not
   guarantee stable motion; a dynamics residual is the physically meaningful
   target.
5. **Missing artifacts.** `data/` and the training checkpoints are not tracked,
   so results cannot currently be reproduced from a clean checkout.
6. **No uncertainty or OOD gate.** The Mission runner correctly disables the
   learned overlays, but there is no principled way to enable them for a new
   mission.

## Workstreams

### 1. Evaluation first

Before changing the model, make the score honest.

- Add a CLI split mode to `ml/pinn/run.py`:
  - `--split random` (current behavior, keep for comparison),
  - `--split time` (train on early samples, test on the tail),
  - `--split group --group-col arc_id` (hold out whole arcs/missions).
- Keep the last 20–30% of each arc out of training by default for the reported
  score.
- Run at least 3 seeds and report mean and spread.
- Save split metadata (train/test time ranges, arc IDs, row counts) next to the
  metrics.

Acceptance gate:

- Temporal holdout R² is positive for at least one model.
- Corrected trajectory RMS is lower than physics-only RMS on the temporal
  holdout, not just on random rows.

### 2. State-based features

Replace calendar time as the primary input.

Build a training table that joins residuals to the reference state. Proposed
columns:

- `r_earth`, `v_earth`: spacecraft state relative to Earth,
- `r_moon`, `v_moon`: state relative to the Moon,
- `r_sun`: Sun-relative position or unit direction,
- `radius_earth`, `radius_moon`, `speed`, `flight_path_angle`,
- `specific_energy`, `angular_momentum`,
- `moon_phase`, `sun_angle`,
- optional vehicle parameters (`mass`, `area`, `cr`) when available.

Recommended derived features:

- inverse distances and their powers (`1/r`, `1/r²`, `1/r³`) because gravity is
  nonlinear in distance,
- Earth/Moon/Sun direction cosines,
- normalized state within the arc,
- relative geometry (Earth-Moon-spacecraft angle).

Do **not** feed raw `t` unless it is needed to model a real epoch-dependent
effect such as the Sun's position. Prefer Sun-relative geometry instead.

Acceptance gate:

- A state-feature model beats the time-feature model on the temporal holdout
  with the same architecture and training budget.

### 3. Dynamics residual target

Move from "add a position offset" to "correct the acceleration."

- Extend the residual table with acceleration residuals:
  `arx, ary, arz = a_reference - a_propagated`.
- Train the model to predict the acceleration residual from the state.
- Integrate the corrected acceleration with the existing propagator (short
  sub-steps, same adaptive controller), rather than adding a post-hoc offset.
- Regularize correction magnitude and smoothness.
- Report energy and angular-momentum drift for physics-only and corrected paths.

Acceptance gate:

- Corrected propagation stays bounded over the full 30-day arc.
- Energy drift is not worse than physics-only by more than a small tolerance.
- Long-horizon RMS improves without a growing tail divergence.

Keep the position-residual model as a baseline; it is useful for comparison.

### 4. Diverse training data

Generate multiple arcs instead of one.

```bash
# Moon reference with measured separation and Sun
make fetch-data
make fetch-sun
make residual-tabulated-sun

# Artemis I/II once the fetcher is fixed
uv run --extra data python scripts/fetch_artemis.py --mission 1
cargo run -p residual --example residual_pipeline -- \
  data/reference_artemis1.csv data/residual_artemis1.csv

# Mission-like arcs from the live propagator / synthetic perturbations
# (add a small script that writes canonical reference CSVs)
```

Data requirements:

- at least 5–10 distinct arcs,
- different start epochs across the year,
- different cadences (e.g. 60 s, 600 s),
- at least one spacecraft/Artemis arc with real separation,
- burn and coast windows,
- perturbed initial conditions for robustness.

Each row needs an `arc_id` and `epoch_id` so grouped splits are possible.

Acceptance gate:

- Leave-one-arc-out evaluation is reported.
- A model trained on N−1 arcs beats physics-only on the held-out arc.

### 5. Model families and constraints

Baselines first:

1. linear regression,
2. SINDy,
3. MLP,
4. PINN,
5. residual MLP / neural ODE.

Then add:

- uncertainty estimates (deep ensemble or Gaussian head),
- rotation-equivariant features where physically appropriate,
- conservation penalties (energy, angular momentum),
- bounded correction output.

Do not make a large architecture the headline until the state-based MLP
baseline transfers. A Transformer or KAN without a temporal/mission holdout is
not evidence of generalization.

Acceptance gate:

- The best model is selected by held-out temporal/mission metrics, not training
  loss or random-split R².
- Uncertainty correlates with actual error on held-out arcs.

### 6. Out-of-distribution gating

The Mission runner must not apply a correction it cannot trust.

Implement a gate with at least:

- distance of the mission state from the training distribution (standardized
  feature distance or ensemble disagreement),
- correction magnitude relative to local acceleration,
- supported duration/epoch range.

Behavior:

- low risk: show physics + corrected path with an uncertainty band,
- medium risk: show correction dashed with a warning,
- high risk: physics only, explicitly labeled "correction unavailable."

Acceptance gate:

- New missions outside the training range are flagged.
- The gate does not suppress corrections on the benchmark missions.

### 7. Benchmark suite

Create `data/benchmarks/` or `benchmarks/generalization/` with canonical cases:

| Case | Purpose |
| --- | --- |
| LEO 400 km, 2 h | short-period sanity |
| LEO with prograde burn | burn handling |
| TLI, 3 days | long-horizon transfer |
| Lunar flyby | close approach / nonlinear regime |
| Moon arc, epoch A | in-distribution reference |
| Moon arc, epoch B | epoch transfer |
| Artemis I arc | spacecraft transfer |

For each case report:

- physics-only RMS / max / final error,
- corrected RMS / max / final error,
- energy and angular-momentum drift,
- correction magnitude,
- runtime,
- OOD score and gate decision.

Store results as JSON so the viewer or README can consume them.

### 8. Integration and presentation

- Add a **model selector** in the viewer: physics-only, PINN, MLP, best-by-holdout.
- Draw the corrected path only when the gate allows it.
- Show a small metric table: holdout R², RMS improvement, OOD status.
- Keep the static bundled demo stable; do not regenerate it during a live demo.
- Update the README to distinguish:
  - interpolation metrics,
  - temporal holdout,
  - mission holdout.

## Priority order

### P0 — before any deadline

1. Add temporal and grouped split modes.
2. Re-evaluate the existing models honestly.
3. Fix data provenance: keep the exact reference, residual table, predictions,
   weights, and split metadata for the demo.
4. Expose MLP and PINN corrected paths separately so the stronger model is not
   hidden behind the "PINN" label.

### P1 — next iteration

1. Build the state-joined training table.
2. Train state-based MLP and PINN.
3. Add multiple arcs and leave-one-arc-out evaluation.
4. Add uncertainty and a simple OOD gate.

### P2 — research stretch

1. Acceleration-residual model integrated by the propagator.
2. Neural ODE or residual dynamics model.
3. Conservation-constrained training.
4. Mission-runner correction with confidence bands.

## What not to claim

Until the P0/P1 gates pass:

- do not call the current model "generalized";
- do not quote random-split R² as trajectory accuracy;
- do not describe the PINN as better than the MLP without the holdout numbers;
- do not apply the Moon-arc correction to a new mission without a gate;
- do not present a mock Cosmos artifact as a model-generated video.

## Definition of done

A reasonable "generalized" milestone is:

- temporal holdout R² > 0 and corrected RMS < physics RMS;
- leave-one-arc-out corrected RMS < physics RMS;
- the OOD gate flags missions outside the training envelope;
- the Mission runner shows corrected and uncorrected paths with the gate state;
- all benchmark results are reproducible from committed data or a documented
  fetch command.
