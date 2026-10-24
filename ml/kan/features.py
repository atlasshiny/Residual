"""Time-derived feature engineering for residual tables.

Replaces the raw (and badly scaled) sample time with a small set of
well-scaled, physically meaningful features so that KAN / MLP / SINDy train
stably.  See ``docs/kan-feature-plan.md``.

The module is intentionally dependency-light: pandas is imported lazily (like
:mod:`ml.kan.dataset`) and numpy is used for the trigonometry when available,
falling back to the stdlib :mod:`math` module otherwise.
"""

from __future__ import annotations

import math
from typing import Any

from .dataset import _import_pandas, resolve_time_column

__all__ = [
    "FEATURE_COLUMNS",
    "LUNAR_ORBIT_PERIOD_S",
    "engineer_features",
]

#: Sidereal lunar orbital period in seconds (27.321661 days).
LUNAR_ORBIT_PERIOD_S = 2.3605915e6

#: Names of the engineered features, in the order they are returned.
FEATURE_COLUMNS = ("t_norm", "phase", "sin_phase", "cos_phase", "t_scaled")

#: Divisor used for the gentle linear time proxy (``t_scaled``).
_SCALE_DIVISOR = 1e5

_TWO_PI = 2.0 * math.pi


def _import_numpy():
    """Return the ``numpy`` module, or ``None`` when it is unavailable."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - depends on env
        return None
    return np


def _wrap_to_pi(phase):
    """Wrap ``phase`` into the half-open interval ``[-pi, pi)``."""
    return (phase + math.pi) % _TWO_PI - math.pi


def engineer_features(df: Any, time_col: str | None = None):
    """Append time-derived features to ``df`` and return them.

    ``time_col`` names the sample-time column; when ``None`` it is auto-detected
    with :func:`ml.kan.dataset.resolve_time_column`.  A :class:`ValueError` is
    raised when no time column can be found or the frame is empty.

    The input frame is **not** mutated.  Returns ``(df_out, feature_cols)`` where
    ``df_out`` is a copy carrying the new columns and ``feature_cols`` is the
    ordered list ``['t_norm', 'phase', 'sin_phase', 'cos_phase', 't_scaled']``.
    """
    pd = _import_pandas()

    col = resolve_time_column(df, time_col)
    if col is None:
        raise ValueError(
            "no time column found; pass time_col explicitly "
            "(known names: t, t_tdb, time, time_s, t_s, epoch_s, timestamp)"
        )
    if len(df) == 0:
        raise ValueError("cannot engineer features from an empty DataFrame")

    out = df.copy()
    t = pd.to_numeric(out[col], errors="coerce").astype("float64")

    t_mean = float(t.mean())
    # Population standard deviation; guard a degenerate (zero/NaN) spread.
    t_std = float(t.std(ddof=0))
    if not math.isfinite(t_std) or t_std == 0.0:
        t_std = 1.0

    t0 = float(t.iloc[0])
    period = LUNAR_ORBIT_PERIOD_S

    np = _import_numpy()
    if np is not None:
        arr = t.to_numpy(dtype="float64")
        t_norm = (arr - t_mean) / t_std
        phase = _wrap_to_pi(_TWO_PI * (arr - t0) / period)
        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)
        t_scaled = arr / _SCALE_DIVISOR
    else:  # pragma: no cover - exercised only without numpy
        vals = [float(v) for v in t.tolist()]
        t_norm = [(v - t_mean) / t_std for v in vals]
        phase = [_wrap_to_pi(_TWO_PI * (v - t0) / period) for v in vals]
        sin_phase = [math.sin(p) for p in phase]
        cos_phase = [math.cos(p) for p in phase]
        t_scaled = [v / _SCALE_DIVISOR for v in vals]

    out["t_norm"] = t_norm
    out["phase"] = phase
    out["sin_phase"] = sin_phase
    out["cos_phase"] = cos_phase
    out["t_scaled"] = t_scaled

    return out, list(FEATURE_COLUMNS)
