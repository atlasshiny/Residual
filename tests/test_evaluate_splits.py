"""Tests for the multi-split/multi-seed evaluation helper."""

from __future__ import annotations

from pathlib import Path

from ml.pinn.evaluate_splits import MODELS, aggregate_runs, build_run_args


def _result(test_r2, test_mse):
    return {
        model: {
            "train_mse": 1.0,
            "train_r2": 1.0,
            "test_mse": test_mse,
            "test_r2": test_r2,
        }
        for model in MODELS
    }


def test_aggregate_reports_mean_and_spread():
    runs = [
        {"split": "random", "seed": 0, "results": _result(0.50, 4.0)},
        {"split": "random", "seed": 1, "results": _result(0.70, 2.0)},
    ]

    aggregate = aggregate_runs(runs)

    pinn = aggregate["pinn"]["test_r2"]
    assert pinn["n"] == 2
    assert pinn["mean"] == 0.60
    assert pinn["min"] == 0.50
    assert pinn["max"] == 0.70
    assert abs(pinn["std"] - 0.10) < 1e-12
    assert aggregate["pinn"]["test_mse"]["mean"] == 3.0


def test_aggregate_skips_failed_and_non_finite_runs():
    runs = [
        {"split": "time", "seed": 0, "results": _result(0.25, 9.0)},
        {"split": "time", "seed": 1, "error": "not enough groups"},
        {
            "split": "time",
            "seed": 2,
            "results": _result(float("nan"), float("inf")),
        },
    ]

    aggregate = aggregate_runs(runs)

    assert aggregate["pinn"]["test_r2"]["n"] == 1
    assert aggregate["pinn"]["test_r2"]["mean"] == 0.25
    assert aggregate["pinn"]["test_mse"]["n"] == 1
    assert aggregate["pinn"]["test_mse"]["mean"] == 9.0


def test_build_run_args_isolates_each_run(tmp_path):
    run_dir = tmp_path / "random_seed3"
    args = build_run_args(
        "data/residual_table.csv",
        "random",
        3,
        run_dir,
        test_fraction=0.3,
        epochs=5,
        lr=0.01,
        features="t_norm,sin_phase",
        targets="drx,dry,drz",
        engineer=True,
    )

    assert args.split == "random"
    assert args.seed == 3
    assert args.test_fraction == 0.3
    assert args.epochs == 5
    assert args.lr == 0.01
    assert args.features == "t_norm,sin_phase"
    assert args.targets == "drx,dry,drz"
    assert args.engineer is True
    assert Path(args.out) == run_dir / "pinn_result.json"
    assert Path(args.weights_out) == run_dir / "pinn_models.pt"
    assert Path(args.predictions_out) == run_dir / "pinn_predictions.json"


def test_build_run_args_passes_group_column():
    args = build_run_args(
        "data/multi_arc.csv",
        "group",
        0,
        "results/run",
        group_col="mission_id",
    )

    assert args.split == "group"
    assert args.group_col == "mission_id"


def test_main_writes_aggregate_report(tmp_path, monkeypatch):
    from ml.pinn import evaluate_splits

    calls = []

    def fake_run(args):
        calls.append((args.split, args.seed))
        out = Path(args.out)
        out.write_text("{}")
        payload = _result(0.5 + 0.1 * args.seed, 4.0)
        payload["split"] = {"mode": args.split, "seed": args.seed}
        return payload, str(out)

    monkeypatch.setattr(evaluate_splits, "run", fake_run)
    report, out_path = evaluate_splits.main(
        [
            "--data",
            "data/table.csv",
            "--splits",
            "random,time",
            "--seeds",
            "0,1",
            "--out",
            str(tmp_path / "eval.json"),
            "--run-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert calls == [("random", 0), ("random", 1), ("time", 0), ("time", 1)]
    assert len(report["runs"]) == 4
    assert report["aggregate"]["pinn"]["test_r2"]["n"] == 4
    assert out_path.exists()
    assert (tmp_path / "runs" / "time_seed1" / "pinn_result.json").exists()

