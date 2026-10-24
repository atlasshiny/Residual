"""Tests for the Horizons/SPICE comparison helper."""


import pytest

from ml.validation.horizons import (
    compare,
    load_library,
    load_reference,
    make_synthetic_reference,
)


@pytest.fixture(scope="module")
def lib():
    try:
        return load_library()
    except (FileNotFoundError, RuntimeError) as e:
        message = str(e)
        if (
            "skipped in CI" in message
            or "not built" in message
            or "Could not find" in message
        ):
            pytest.skip(f"CUDA shared library not available: {e}")
        raise


def test_make_synthetic_reference_shape():
    try:
        df = make_synthetic_reference(steps=10, dt=60.0)
        assert len(df) == 11
        assert {"t", "x", "y", "z", "vx", "vy", "vz"}.issubset(df.columns)
    except (FileNotFoundError, RuntimeError) as e:
        message = str(e)
        if (
            "skipped in CI" in message
            or "not built" in message
            or "Could not find" in message
        ):
            pytest.skip("Skipping CUDA-dependent synthetic reference generation in CI")
        raise


def test_compare_identical_trajectory(lib):
    ref = make_synthetic_reference(steps=20, dt=60.0)
    metrics = compare(ref, lib=lib)
    assert metrics["position_rmse_m"] < 1e-6
    assert metrics["velocity_rmse_mps"] < 1e-9


def test_compare_detects_perturbation(lib):
    ref = make_synthetic_reference(steps=20, dt=60.0)
    ref = ref.copy()
    ref["x"] = ref["x"] + 100.0
    metrics = compare(ref, lib=lib)
    assert metrics["position_rmse_m"] > 50.0


def test_load_reference_csv(tmp_path):
    try:
        df = make_synthetic_reference(steps=5, dt=60.0)
        path = tmp_path / "ref.csv"
        df.to_csv(path, index=False)
        loaded = load_reference(path)
        assert len(loaded) == 6
        assert list(loaded.columns)[:2] == ["t", "x"]
    except (FileNotFoundError, RuntimeError) as e:
        message = str(e)
        if (
            "skipped in CI" in message
            or "not built" in message
            or "Could not find" in message
        ):
            pytest.skip("Skipping CUDA-dependent synthetic reference generation in CI")
        raise
