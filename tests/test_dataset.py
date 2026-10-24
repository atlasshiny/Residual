import pytest

pd = pytest.importorskip("pandas")

from ml.pinn.dataset import build_xy, load_residual_table, split_xy


def test_build_xy_shapes():
    df = pd.DataFrame(
        {"x0": [1.0, 2.0, 3.0], "x1": [4.0, 5.0, 6.0], "y": [7.0, 8.0, 9.0]}
    )
    X, Y = build_xy(df, feature_cols=["x0", "x1"], target_cols=["y"])
    assert len(X) == 3
    assert len(X[0]) == 2
    assert len(Y) == 3
    assert len(Y[0]) == 1

    # auto-detection path
    X2, Y2 = build_xy(df)
    assert len(X2) == 3 and len(X2[0]) == 2
    assert len(Y2) == 3 and len(Y2[0]) == 1


def test_parquet_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    df = pd.DataFrame(
        {"x0": [1.0, 2.0, 3.0], "x1": [4.0, 5.0, 6.0], "y": [7.0, 8.0, 9.0]}
    )
    path = tmp_path / "table.parquet"
    df.to_parquet(path, index=False)
    loaded = load_residual_table(path)
    X, Y = build_xy(loaded, feature_cols=["x0", "x1"], target_cols=["y"])
    assert len(X) == 3 and len(X[0]) == 2
    assert len(Y) == 3 and len(Y[0]) == 1


def test_split_sizes():
    X = [[float(i)] for i in range(100)]
    Y = [[float(i)] for i in range(100)]
    Xtr, Xte, Ytr, Yte = split_xy(X, Y, test_fraction=0.2, seed=0)
    assert len(Xtr) == 80
    assert len(Xte) == 20
    assert len(Ytr) == 80
    assert len(Yte) == 20


def test_split_deterministic():
    X = [[float(i)] for i in range(50)]
    Y = [[float(i)] for i in range(50)]
    a = split_xy(X, Y, test_fraction=0.2, seed=1)
    b = split_xy(X, Y, test_fraction=0.2, seed=1)
    assert a == b