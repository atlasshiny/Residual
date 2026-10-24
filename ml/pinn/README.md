# ml/kan

Minimal, dependency-free KAN (Kolmogorov-Arnold Network) plus the baselines
required by ARCHITECTURE.md section 5.

Files:
- `kan.py`      - KAN model, B-spline edges, training, pruning, symbolic extraction
- `baselines.py`- Plain MLP and SINDy (STLSQ) baselines
- `demo.py`     - Synthetic residual demo comparing all three (thin wrapper over run.py)
- `dataset.py`  - Lazy pandas/pyarrow helpers for residual tables
- `run.py`      - CLI/core entry point for training on synthetic or residual data
- `__init__.py` - package exports

Run:
    python -m ml.kan.demo
    python -m ml.kan.run
    python -m ml.kan.run --data path/to/residual_table.parquet \
        --features x0,x1 --targets y --epochs 100 --out ml/kan/kan_result.json

Dataset helpers:
    from ml.kan.dataset import load_residual_table, build_xy, split_xy
    df = load_residual_table("residual_table.parquet")
    X, Y = build_xy(df, feature_cols=["x0", "x1"], target_cols=["y"])
    Xtr, Xte, Ytr, Yte = split_xy(X, Y, test_fraction=0.2, seed=0)

The run writes `kan_result.json` with trained metrics and fitted edge
polynomials.

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

Core model code is pure Python (standard library only). `dataset.py` and
`run.py` may import pandas/pyarrow lazily for residual data.