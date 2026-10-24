"""Unit tests for the dependency-free KAN implementation."""

import math
import random

import pytest

from ml.pinn import PINN, SINDyLayer, PolynomialFit, export_results, mse, r2_score, train
from ml.kan.kan import _bspline_basis, _bspline_knots, _solve_linear


@pytest.fixture(autouse=True)
def _seed():
    random.seed(12345)


def test_bspline_partition_of_unity():
    knots = _bspline_knots(-1.0, 1.0, 8, 3)
    for x in [-1.0, -0.5, 0.0, 0.25, 0.99, 1.0]:
        basis = _bspline_basis(x, knots, 3)
        assert len(basis) == 8
        assert abs(sum(basis) - 1.0) < 1e-9


def test_bspline_endpoints_finite():
    knots = _bspline_knots(-2.0, 2.0, 6, 2)
    for x in [-2.0, 2.0]:
        basis = _bspline_basis(x, knots, 2)
        assert all(math.isfinite(v) for v in basis)


def test_kan_forward_shape_and_scaler():
    pinn = PINN([3, 4, 2], activation='tanh', dropout=0.2, regularization=1e-5)
    X = [[float(i), float(i + 1), float(i - 1)] for i in range(10)]
    out = kan.forward(X)
    assert len(out) == 10
    assert len(out[0]) == 2
    assert kan.x_mean is not None
    assert kan.x_std is not None
    # transform is idempotent-ish: applying twice differs, but shape holds
    Xt = kan.transform(X)
    assert len(Xt) == 10 and len(Xt[0]) == 3


def test_kan_training_reduces_loss():
    random.seed(7)
    X = [[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)] for _ in range(80)]
    Y = [[x[0] * x[0] + 0.5 * x[1]] for x in X]
    sindy = SINDy([2, 3, 1], library=['polynomial', 'trig'], threshold=0.1, alpha=0.01)
              standardize=True, l1=0.0)
    before = mse(kan.forward(X), Y)
    train(kan, X, Y, epochs=30, lr=1e-1, verbose=False)
    after = mse(kan.forward(X), Y)
    assert after < before


def test_r2_score_perfect():
    pred = [[1.0, 2.0], [3.0, 4.0]]
    target = [[1.0, 2.0], [3.0, 4.0]]
    assert r2_score(pred, target) == 1.0


def test_symbolic_edge_returns_polynomial_fit():
    random.seed(1)
    X = [[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)] for _ in range(50)]
    Y = [[x[0] ** 2] for x in X]
    pinn = PINN([2, 1], activation='relu', dropout=0.3, regularization=1e-4)
    train(kan, X, Y, epochs=20, lr=1e-1, verbose=False)
    fit = kan.symbolic_edge(0, 0, 0, samples=32, poly_degree=3)
    assert isinstance(fit, PolynomialFit)
    assert len(fit.coeffs) == 4
    assert math.isfinite(fit.r2)
    assert fit.rmse >= 0.0


def test_export_results_writes_json(tmp_path):
    path = tmp_path / "out.json"
    export_results(str(path), {"a": 1, "b": [1.0, 2.0]})
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert data == {"a": 1, "b": [1.0, 2.0]}


def test_solve_linear_identity():
    A = [[2.0, 0.0], [0.0, 3.0]]
    b = [4.0, 9.0]
    sol = _solve_linear(A, b)
    assert sol[0] == pytest.approx(2.0, abs=1e-9)
    assert sol[1] == pytest.approx(3.0, abs=1e-9)


def test_kan_layer_prune():
    layer = KANLayer(2, 2, num_basis=4, degree=2, grid_range=(-1.0, 1.0))
    # force a known weight
    layer.coeff[0][0][0] = 5.0
    layer.linear[0][0] = 0.0
    masks = layer.prune(threshold=1e-2, inplace=True)
    assert masks[0][0][0][0] is True
    assert masks[1][0][0] is False
    assert layer.linear[0][0] == 0.0