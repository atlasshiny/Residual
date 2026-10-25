"""End-to-end integration tests for the PINN / SINDy pipeline.

These exercise the whole ``ml.pinn.run`` path: synthetic data -> training ->
output JSON -> verifying that all four models (pinn, mlp, sindy, linreg)
report finite train/test MSE and R^2 metrics.

Tests that need PyTorch are skipped when it is not installed, matching the
pattern used in ``tests/test_pinn_run.py``.
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

from ml.pinn.run import build_parser, run

_HAS_TORCH = importlib.util.find_spec("torch") is not None

requires_torch = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="PyTorch not installed (needs --extra gpu)",
)

#: Models the pipeline must report metrics for.
MODELS = ("pinn", "mlp", "sindy", "linreg")

#: Metrics every model must report.
METRIC_KEYS = ("train_mse", "train_r2", "test_mse", "test_r2")


def _parse(out, *extra, epochs=2):
    """Parse args for an isolated run.

    Redirect every output path next to ``out``.  Only setting ``--out`` leaves
    the remaining defaults pointing at ``ml/pinn``, so a test run would
    overwrite the canonical predictions/weights the UI bundle is built from.
    """
    out = Path(out)
    out_dir = out.parent
    return build_parser().parse_args(
        [
            "--out",
            str(out),
            "--weights-out",
            str(out_dir / "pinn_models.pt"),
            "--mlp-weights-out",
            str(out_dir / "mlp_model.pt"),
            "--predictions-out",
            str(out_dir / "pinn_predictions.json"),
            "--mlp-predictions-out",
            str(out_dir / "mlp_predictions.json"),
            "--mlp-predictions-csv-out",
            str(out_dir / "mlp_predictions.csv"),
            "--predictions-csv-out",
            str(out_dir / "pinn_predictions.csv"),
            "--sindy-predictions-csv-out",
            str(out_dir / "sindy_predictions.csv"),
            "--epochs",
            str(epochs),
            *extra,
        ]
    )


def _load(path):
    return json.loads(Path(path).read_text())


def _seed_torch():
    """Seed PyTorch so the PINN's random init is reproducible.

    ``ml.pinn.run`` seeds :mod:`random` but not PyTorch, so the PINN train R^2
    would otherwise vary run-to-run.  Seeding here keeps the ``r2 > 0``
    assertion deterministic without touching production code.
    """
    if _HAS_TORCH:
        import torch

        torch.manual_seed(0)


def _write_residual_csv(path, n, period, reverse=False):
    """Write a synthetic residual table with a smooth position/velocity signal.

    The target ``y`` is a smooth sinusoid of time and ``dvx`` is its analytic
    time-derivative, giving the PINN's physics term a learnable, consistent
    signal.  ``reverse=True`` emits the rows in descending time order to mirror
    the held-out CSV test in ``tests/test_pinn_run.py``.
    """
    rows = ["t,x0,x1,y,dvx,dvy,dvz"]
    order = range(n - 1, -1, -1) if reverse else range(n)
    for i in order:
        t = period * i / (n - 1)
        y = 0.5 * math.sin(2.0 * math.pi * t / period)
        dy = 0.5 * (2.0 * math.pi / period) * math.cos(2.0 * math.pi * t / period)
        rows.append(f"{t},{i / n},{(i * i) / (n * n)},{y},{dy},0.0,0.0")
    Path(path).write_text("\n".join(rows) + "\n")
    return path


def _assert_all_models_reported(data):
    """Assert every model reports finite train/test MSE and R^2."""
    for model in MODELS:
        assert model in data, f"missing model: {model}"
        for key in METRIC_KEYS:
            assert key in data[model], f"missing {model}.{key}"
            assert math.isfinite(data[model][key]), f"{model}.{key} is not finite"


# ---------------------------------------------------------------------------
# 1. Full pipeline on synthetic data
# ---------------------------------------------------------------------------


@requires_torch
def test_full_pipeline_on_synthetic_data(tmp_path):
    _seed_torch()
    data_file = _write_residual_csv(tmp_path / "synthetic.csv", n=100, period=1000.0)
    out = tmp_path / "pinn_result.json"

    results, out_path = run(_parse(out, "--data", str(data_file), "--targets", "y"))

    # Output JSON is written and loadable.
    assert out_path == str(out)
    assert out.exists()
    data = _load(out)

    # All four models report finite train/test MSE and R^2.
    _assert_all_models_reported(data)

    # The PINN learned the smooth synthetic signal (well above the zero-R^2
    # baseline of a constant-mean predictor).
    assert data["pinn"]["train_r2"] > 0.0

    # The in-memory result mirrors what was persisted.
    assert results["masked_fraction"] == 0.0


# ---------------------------------------------------------------------------
# 2. CSV data file (no mission -> no masking)
# ---------------------------------------------------------------------------


@requires_torch
def test_csv_data_file_reports_zero_masked_fraction(tmp_path):
    pytest.importorskip("pandas")
    _seed_torch()
    table = tmp_path / "residual.csv"
    # 50 samples in reversed time order; the pipeline reads the CSV as-is.
    rows = ["t,x0,x1,y,dvx,dvy,dvz"]
    n = 50
    for i in reversed(range(n)):
        rows.append(f"{float(i)},{i / n},{(i * i) / (n * n)},{i / n},0.0,0.0,0.0")
    table.write_text("\n".join(rows) + "\n")
    out = tmp_path / "pinn_result.json"

    results, _ = run(
        _parse(
            out,
            "--data",
            str(table),
            "--features",
            "x0,x1",
            "--targets",
            "y",
        )
    )

    assert out.exists()
    # No mission configured, so nothing is masked.
    assert results["masked_fraction"] == 0.0

    data = _load(out)
    _assert_all_models_reported(data)


# ---------------------------------------------------------------------------
# 3. Mission file -> burn masking
# ---------------------------------------------------------------------------


@requires_torch
def test_mission_file_reports_masked_fraction(tmp_path):
    pytest.importorskip("pandas")
    _seed_torch()
    data_file = _write_residual_csv(tmp_path / "residual.csv", n=60, period=600.0)
    mission = tmp_path / "mission.json"
    mission.write_text(
        json.dumps(
            {
                "name": "Artemis test",
                "burns": [{"label": "TLI", "start_s": 100.0, "end_s": 300.0}],
            }
        )
    )
    out = tmp_path / "pinn_result.json"

    results, _ = run(
        _parse(
            out,
            "--data",
            str(data_file),
            "--targets",
            "y",
            "--mission",
            str(mission),
        )
    )

    assert out.exists()

    # Burn masking is reported: 20 of the 60 uniformly-spaced samples fall in
    # the half-open window [100, 300) s.
    assert results["masked_fraction"] > 0.0
    assert results["masked_fraction"] == pytest.approx(1.0 / 3.0, abs=0.05)
    assert results["n_samples"] == 40

    data = _load(out)
    assert data["masked_fraction"] == pytest.approx(1.0 / 3.0, abs=0.05)
    _assert_all_models_reported(data)
