"""Tests for ml.kan.features.engineer_features."""

from __future__ import annotations

import math

import pytest

pd = pytest.importorskip("pandas")

from ml.pinn.features import engineer_features


def _df():
    import pandas as pd

    return pd.DataFrame(
        {"t": [float(i * 10) for i in range(20)], "y": [float(i) for i in range(20)]}
    )


def test_engineer_adds_columns():
    df = _df()
    out, cols = engineer_features(df)
    assert len(cols) == 5
    for c in cols:
        assert c in out.columns


def test_deterministic():
    df = _df()
    out1, _ = engineer_features(df)
    out2, _ = engineer_features(df)
    for c in ["t_norm", "phase", "sin_phase", "cos_phase", "t_scaled"]:
        assert out1[c].tolist() == out2[c].tolist()


def test_scaled_range():
    df = _df()
    out, _ = engineer_features(df)
    assert abs(out["t_norm"].mean()) < 1e-9
    assert abs(out["t_norm"].std(ddof=0) - 1.0) < 1e-9


def test_no_nan():
    df = _df()
    out, cols = engineer_features(df)
    for c in cols:
        assert not out[c].isna().any()


def test_does_not_mutate_input():
    df = _df()
    original_cols = list(df.columns)
    engineer_features(df)
    assert list(df.columns) == original_cols
