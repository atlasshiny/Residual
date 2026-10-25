"""Held-out (time-window) evaluation and burn masking for the PINN/SINDy pipeline."""

import json
import math
from pathlib import Path

import pytest

from ml.pinn.dataset import apply_burn_mask, burn_keep_mask, split_by_time
from ml.pinn.run import _select_split, build_parser, run


def _parse(out, *extra, epochs=2):
    """Parse args for an isolated run.

    Every output path is redirected next to ``out`` so a test can never
    overwrite the committed ``ml/pinn`` artifacts (this used to happen because
    only ``--out`` was set and the remaining paths kept their defaults).
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
# Split modes
# ---------------------------------------------------------------------------


def _split_frame():
    pandas = pytest.importorskip("pandas")
    return pandas.DataFrame(
        {
            "t": [float(i) for i in range(12)][::-1],
            "arc_id": ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3,
            "x0": [float(i) for i in range(12)],
            "y": [float(i * i) for i in range(12)],
        }
    )


def test_select_split_time_holds_out_latest_samples():
    df = _split_frame()
    args = build_parser().parse_args(["--split", "time", "--test-fraction", "0.25"])
    train, test, meta = _select_split(df, args, "t")

    assert len(train) == 9
    assert len(test) == 3
    assert meta["mode"] == "time"
    assert meta["split_unit"] == "time"
    assert meta["n_train"] == 9
    assert meta["n_test"] == 3
    assert max(train["t"]) < min(test["t"])
    assert meta["train_time_range_s"] == [
        float(min(train["t"])),
        float(max(train["t"])),
    ]
    assert meta["test_time_range_s"] == [float(min(test["t"])), float(max(test["t"]))]


def test_select_split_group_holds_out_whole_arcs():
    df = _split_frame()
    args = build_parser().parse_args(
        ["--split", "group", "--group-col", "arc_id", "--test-fraction", "0.25"]
    )
    train, test, meta = _select_split(df, args, "t")

    assert meta["mode"] == "group"
    assert meta["split_unit"] == "group"
    assert len(meta["test_groups"]) == 1
    assert set(train["arc_id"]).isdisjoint(set(test["arc_id"]))
    assert set(test["arc_id"]) == set(meta["test_groups"])
    assert set(train["arc_id"]) == set(meta["train_groups"])
    assert len(train) + len(test) == len(df)


def test_select_split_random_is_seeded():
    df = _split_frame()
    args_a = build_parser().parse_args(["--split", "random", "--seed", "7"])
    args_b = build_parser().parse_args(["--split", "random", "--seed", "7"])
    train_a, test_a, meta_a = _select_split(df, args_a, "t")
    train_b, test_b, _meta_b = _select_split(df, args_b, "t")

    assert train_a["x0"].tolist() == train_b["x0"].tolist()
    assert test_a["x0"].tolist() == test_b["x0"].tolist()
    assert meta_a["mode"] == "random"
    assert meta_a["seed"] == 7


def test_select_split_group_requires_group_column():
    df = _split_frame()
    args = build_parser().parse_args(
        ["--split", "group", "--group-col", "missing_column"]
    )
    with pytest.raises(ValueError, match="group column"):
        _select_split(df, args, "t")


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


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="PyTorch not installed (needs --extra gpu)",
)
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


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="PyTorch not installed (needs --extra gpu)",
)
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


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="PyTorch not installed (needs --extra gpu)",
)
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
