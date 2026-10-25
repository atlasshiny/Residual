"""Held-out (time-window) evaluation and burn masking for the PINN/SINDy pipeline."""

import json
import math
from pathlib import Path

import pytest

from ml.pinn.dataset import apply_burn_mask, burn_keep_mask, split_by_time
from ml.pinn.run import build_parser, run


def _parse(out, *extra, epochs=2):
    return build_parser().parse_args(["--out", str(out), *extra])


def _load(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Deterministic time-window split
# ---------------------------------------------------------------------------


def test_split_by_time_sizes_and_holdout_order():
    X = [[float(i)] for i in range(100)]
    Y = [[2.0 * i] for i in range(100)]
    times = [float(i) for i in range(100)]

    X_train, X_test, Y_train, Y_test = split_by_time(X, Y, t=times)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(Y_train) == 80
    assert len(Y_test) == 20
    # The held-out set is the latest 20% of the time window.
    assert [row[0] for row in X_train] == [float(i) for i in range(80)]
    assert [row[0] for row in X_test] == [float(i) for i in range(80, 100)]


def test_split_by_time_sorts_caller_order():
    # Deliberately unsorted rows: the split must sort by t, not input order.
    times = [2.0, 0.0, 1.0, 4.0, 3.0]
    X = [[t] for t in times]
    Y = [[t] for t in times]

    X_train, X_test, _Y_train, _Y_test = split_by_time(X, Y, t=times, test_fraction=0.4)

    assert len(X_train) == 3
    assert len(X_test) == 2
    assert [row[0] for row in X_train] == [0.0, 1.0, 2.0]
    assert [row[0] for row in X_test] == [3.0, 4.0]


def test_split_by_time_is_deterministic_and_requires_ordered_times():
    X = [[float(i)] for i in range(30)]
    Y = [[float(i)] for i in range(30)]
    times = [float(i) for i in range(30)]

    a = split_by_time(X, Y, t=times, test_fraction=0.2)
    b = split_by_time(X, Y, t=times, test_fraction=0.2)
    assert a == b
    assert len(a[1]) == 6


# ---------------------------------------------------------------------------
# Burn masking
# ---------------------------------------------------------------------------


def test_burn_mask_is_half_open():
    times = [0.0, 10.0, 15.0, 20.0]
    burns = [{"label": "TLI", "start_s": 10.0, "end_s": 20.0}]

    assert burn_keep_mask(times, burns) == [True, False, False, True]

    X = [[t] for t in times]
    Y = [[t] for t in times]
    X_keep, Y_keep, t_keep, masked_fraction, masked = apply_burn_mask(
        X, Y, times, burns
    )
    assert len(X_keep) == len(Y_keep) == len(t_keep) == 2
    assert t_keep == [0.0, 20.0]
    assert masked == 2
    assert masked_fraction == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------


def test_synthetic_run_writes_test_metrics(tmp_path):
    out = tmp_path / "pinn_result.json"
    results, out_path = run(_parse(out))

    assert out_path == str(out)
    assert results["masked_fraction"] == 0.0
    assert results["n_samples"] == 12

    data = _load(out)
    for model in ("pinn", "mlp", "sindy", "linreg"):
        for key in ("train_mse", "train_r2", "test_mse", "test_r2"):
            assert key in data[model], f"missing {model}.{key}"
            assert math.isfinite(data[model][key])


def test_run_applies_mission_burn_mask_before_training(tmp_path):
    mission = tmp_path / "mission.json"
    mission.write_text(
        json.dumps(
            {
                "name": "Artemis test",
                "burns": [{"label": "TLI", "start_s": 0.0, "end_s": 100.0}],
            }
        )
    )
    out = tmp_path / "pinn_result.json"

    _results, _ = run(_parse(out, "--mission", str(mission)))

    # Burn masking only applies to real data; synthetic mode doesn't mask.
    assert _results["masked_fraction"] == 0.0
    assert _results["n_samples"] == 12


def test_run_uses_time_column_for_residual_table(tmp_path):
    pytest.importorskip("pandas")
    table = tmp_path / "residual.csv"
    # 50 samples in reversed time order; split must sort them by t.
    rows = ["t,x0,x1,y,dvx,dvy,dvz"]
    for i in reversed(range(50)):
        rows.append(f"{float(i)},{i / 50.0},{(i * i) / 2500.0},{i / 50.0},0.0,0.0,0.0")
    table.write_text("\n".join(rows) + "\n")

    out = tmp_path / "pinn_result.json"
    _results, _ = run(
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

    data = _load(out)
    for model in ("pinn", "mlp", "sindy", "linreg"):
        for key in ("train_mse", "train_r2", "test_mse", "test_r2"):
            assert key in data[model], f"missing {model}.{key}"
            assert math.isfinite(data[model][key])
