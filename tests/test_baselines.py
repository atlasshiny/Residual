"""Unit tests for MLP and SINDy baselines."""

import random

import pytest

from ml.pinn.baselines import (
    MLP,
    _poly_features,
    mse,
    poly_terms,
    poly_transform,
    sindy,
    sindy_predict,
)


@pytest.fixture(autouse=True)
def _seed():
    random.seed(999)


def test_poly_terms_structure():
    terms = poly_terms(2, 2)
    # x0, x1, x0^2, x0*x1, x1^2, 1
    assert len(terms) == 6
    names = [name for _, name in terms]
    assert "x0" in names
    assert "x0*x1" in names
    assert "1" in names[-1]


def test_poly_transform():
    terms = poly_terms(2, 2)
    row = [2.0, 3.0]
    vals = poly_transform(row, terms)
    assert vals[0] == pytest.approx(2.0)  # x0
    assert vals[1] == pytest.approx(3.0)  # x1


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="PyTorch not installed (needs --extra gpu)",
)
def test_mlp_forward_shape_and_training():
    random.seed(1)
    mlp = MLP([2, 4, 1])
    X = [[0.1, 0.2], [0.3, 0.4]]
    out = mlp.forward(X)
    assert len(out) == 2
    assert len(out[0]) == 1

    # training reduces loss on a tiny linear target
    random.seed(2)
    X = [[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)] for _ in range(60)]
    Y = [[2.0 * x[0] - 1.0 * x[1]] for x in X]
    mlp = MLP([2, 8, 1])
    before = mse(mlp.forward(X), Y)
    mlp.train(X, Y, epochs=50, lr=1e-2, verbose=False)
    after = mse(mlp.forward(X), Y)
    assert after < before


def test_sindy_recovers_quadratic():
    random.seed(3)
    X = [[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)] for _ in range(200)]
    Y = [[0.5 * x[0] * x[0] - 0.3 * x[1] + 0.1 * x[0] * x[1]] for x in X]
    Xi, names, exprs = sindy(X, Y, threshold=0.05, degree=2)
    pred = sindy_predict(X, Xi, degree=2)
    assert mse(pred, Y) < 1e-3
    assert "x0^2" in names
    assert len(exprs) == 1


def test_poly_features_degree():
    X = [[1.0, 2.0], [3.0, 4.0]]
    feats, names = _poly_features(X, degree=2)
    assert len(feats) == 2
    # 2 linear + 3 quadratic + 1 bias = 6
    assert len(feats[0]) == 6
    assert len(names) == 6