# Impulsive Burn Support Plan

## Goal
Add impulsive (instantaneous delta-V) burns to the propagator so it can model
orbital transfers and correction maneuvers, making it useful for mission design.

## Design
- `MbsBurn { double t; double dvx, dvy, dvz; }` struct in multibody_solver.cuh.
- `MbsConfig` gains `MbsBurn burns[8]` + `int num_burns` (cap 8; 0 = none).
- Kernel: at the start of each step, for each burn with `|t - burn.t| < 0.5*dt`,
  add `dv` to every particle's velocity (uniform burn; per-particle burns are
  out of scope).
- Rust `Config` mirrors the field; `propagate` applies burns identically.
- `mbs_default_config` zeroes burns/num_burns; `valid_config` checks num_burns in [0,8].

## Scope
- CUDA: .cuh (struct + config), .cu (default, valid, kernel).
- Rust: lib.rs (Config, propagate, acceleration unaffected).
- No changes to benchmarks or UI.

## Agents
- Single lane: implement in .cuh, .cu, lib.rs. Small, well-defined.
