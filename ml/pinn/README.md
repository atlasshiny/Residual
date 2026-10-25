# ml/pinn

PINN-based symbolic-correction pipeline plus the baselines required by
ARCHITECTURE.md section 5.

The pipeline (`run.py`) trains a **Physics-Informed Neural Network (PINN)** to
model the propagator's position residuals and compares it honestly against a
plain **MLP**, a **SINDy** (STLSQ) sparse symbolic fit, and a **linear
regression** baseline.

Files:
- `pinn.py`     - PINN (PyTorch): predicts `delta_r = f(t)` and constrains
                  `d(delta_r)/dt` to match `delta_v` via autograd.
- `baselines.py`- Plain MLP, SINDy (STLSQ), and linear-regression baselines.
- `features.py` - Time-derived feature engineering (`t_norm`, `phase`,
                  `sin_phase`, `cos_phase`, `t_scaled`).
- `dataset.py`  - Lazy pandas/pyarrow helpers for residual tables (loading,
                  `build_xy`, time-based splits, burn masking).
- `run.py`      - CLI/core entry point for training on residual data.
- `demo.py`     - Thin wrapper over `run.py` for a synthetic demo.
- `config.py`   - Pydantic-verified configuration loaded from YAML.
- `__init__.py` - package exports.

Run:
    python -m ml.pinn.run
    python -m ml.pinn.demo
    python -m ml.pinn.run --data path/to/residual_table.parquet \
        --features x0,x1 --targets drx,dry,drz --test-fraction 0.2 \
        --out ml/pinn/pinn_result.json

Dataset helpers:
    from ml.pinn.dataset import load_residual_table, build_xy, split_by_time
    df = load_residual_table("residual_table.parquet")
    X, Y = build_xy(df, feature_cols=[...], target_cols=["drx", "dry", "drz"])
    Xtr, Xte, Ytr, Yte = split_by_time(X, Y, test_fraction=0.2)

The run writes `pinn_result.json` with per-model train/test MSE and R² for the
PINN, MLP, SINDy, and linear-regression models, plus the SINDy symbolic
expression.

Notes:
- The PINN requires PyTorch (`uv sync --extra gpu` or `pip install .[gpu]`).
  `pinn.py` raises a clear `ImportError` when torch is unavailable.
- `engineer_features(df, time_col)` replaces raw `t` with scaled, physically
  meaningful features so every model trains stably on unscaled residuals
  (see `docs/feature-plan.md`).
- The PINN consumes normalized time only (`t_norm`); the MLP, SINDy, and linear
  baselines consume the engineered feature matrix.
- Scheduled impulsive burns can be masked out with `--mission`.
