# ml/kan

Minimal, dependency-free KAN (Kolmogorov-Arnold Network) plus the baselines
required by ARCHITECTURE.md section 5.

Files:
- `kan.py`      - KAN model, B-spline edges, training, pruning, symbolic extraction
- `baselines.py`- Plain MLP and SINDy (STLSQ) baselines
- `demo.py`     - Synthetic residual demo comparing all three
- `__init__.py` - package exports

Run:
    cd ml/kan
    python3 demo.py

The demo also writes `kan_result.json` with the trained metrics and the fitted
edge polynomials.

Notes:
- Inputs are z-scored inside `KAN` when `standardize=True` (the default), so
  `grid_range` is expressed in standardized units.  Set
  `KAN(..., standardize=False, grid_range=(-1, 1))` to work in raw units.
- `l1` adds an L1 penalty on spline and linear edge weights.
- `prune(threshold)` is magnitude pruning.  Pass `inplace=False` to inspect the
  mask without modifying the model; pruning is destructive, but zeroed weights
  are ordinary parameters and can be relearned by training again.
- `symbolic_edge` returns a `PolynomialFit` (`coeffs`, `r2`, `rmse`) fitted on
  the layer's `grid_range`.

All code is pure Python (standard library only). For the real residual data,
replace the synthetic `X, Y` in `demo.py` with rows from `residual_table.parquet`.
