"""Lazy dataset helpers for residual tables (pandas/pyarrow optional)."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

__all__ = ["build_xy", "load_residual_table", "split_xy"]


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "pandas is required for ml.kan.dataset. "
            "Install the 'data' extra: `uv sync --extra data` "
            "or `pip install pandas pyarrow`."
        ) from exc
    return pd


def _import_pyarrow():
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "pyarrow is required for parquet support in ml.kan.dataset. "
            "Install the 'data' extra: `uv sync --extra data` "
            "or `pip install pyarrow`."
        ) from exc
    return pa


def _is_numeric(series) -> bool:
    pd = _import_pandas()
    return bool(pd.api.types.is_numeric_dtype(series))


def _to_cols(cols):
    if cols is None:
        return None
    if isinstance(cols, str):
        return [c.strip() for c in cols.split(",") if c.strip()]
    return [str(c) for c in cols]


def load_residual_table(path: Any):
    """Read a parquet or CSV file into a pandas DataFrame."""
    pd = _import_pandas()
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        _import_pyarrow()
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            return table.to_pandas()
        except Exception:  # noqa: BLE001
            return pd.read_parquet(path)
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    # Fall back: try parquet, then CSV.
    try:
        _import_pyarrow()
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pandas()
    except Exception:  # noqa: BLE001
        try:
            return pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Unsupported or unreadable table: {path}") from exc


def _default_target_cols(df, feature_cols):
    known = ("y", "target", "targets", "label", "labels", "output", "outputs")
    for name in known:
        if name in df.columns:
            return [name]
    numeric = [c for c in df.columns if _is_numeric(df[c])]
    if feature_cols is not None:
        candidates = [c for c in numeric if c not in feature_cols]
        if not candidates:
            raise ValueError("could not infer target columns")
        return list(candidates)
    candidates = list(numeric)
    if not candidates:
        raise ValueError("could not infer target columns")
    if len(candidates) == 1:
        return [candidates[0]]
    return [candidates[-1]]


def _default_feature_cols(df, target_cols):
    # Prefer columns named x0, x1, ...
    xcols = [c for c in df.columns if re.fullmatch(r"x\d+", str(c))]
    if xcols:
        xcols = sorted(xcols, key=lambda c: int(str(c)[1:]))
        return xcols
    numeric = [c for c in df.columns if _is_numeric(df[c])]
    if target_cols is not None:
        return [c for c in numeric if c not in target_cols]
    if len(numeric) > 1:
        return numeric[:-1]
    return list(numeric)


def build_xy(df, feature_cols=None, target_cols=None):
    """Return (X, Y) as lists of rows.

    ``feature_cols`` / ``target_cols`` may be lists or comma-separated strings.
    """
    _import_pandas()

    feature_cols = _to_cols(feature_cols)
    target_cols = _to_cols(target_cols)

    if target_cols is None:
        # If features are given, infer targets from remaining numeric cols.
        if feature_cols is not None:
            target_cols = _default_target_cols(df, feature_cols)
        else:
            target_cols = _default_target_cols(df, None)

    if feature_cols is None:
        feature_cols = _default_feature_cols(df, target_cols)

    if not feature_cols:
        raise ValueError("no feature columns found")
    if not target_cols:
        raise ValueError("no target columns found")

    for c in feature_cols + target_cols:
        if c not in df.columns:
            raise KeyError(f"column not found in DataFrame: {c}")

    n = len(df)
    feat_data = [df[c].tolist() for c in feature_cols]
    tgt_data = [df[c].tolist() for c in target_cols]
    X = [[float(feat_data[k][i]) for k in range(len(feature_cols))]
         for i in range(n)]
    Y = [[float(tgt_data[k][i]) for k in range(len(target_cols))]
         for i in range(n)]
    return X, Y


def split_xy(X, Y, test_fraction=0.2, seed=0):
    """Deterministically split (X, Y) into train/test.

    Returns (X_train, X_test, Y_train, Y_test).
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
    n = len(X)
    if n == 0:
        return [], [], [], []
    n_test = int(n * test_fraction)
    if n_test < 0 or n_test > n:
        raise ValueError("test_fraction out of range")
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    Y_train = [Y[i] for i in train_idx]
    Y_test = [Y[i] for i in test_idx]
    return X_train, X_test, Y_train, Y_test