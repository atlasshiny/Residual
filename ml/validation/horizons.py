"""Compare the CUDA multibody propagator to a reference trajectory.

The reference data is expected in metres and seconds, using the same
barycentric convention as ``mbs_propagate``.  A typical Horizons export must be
converted to SI units and to the barycentric frame before calling
:func:`compare`.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
from ctypes import POINTER, Structure, c_double, c_int
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class MbsVec3(Structure):
    _fields_ = [("x", c_double), ("y", c_double), ("z", c_double)]


class MbsState(Structure):
    _fields_ = [("r", MbsVec3), ("v", MbsVec3)]


class MbsConfig(Structure):
    _fields_ = [
        ("dt", c_double),
        ("epoch", c_double),
        ("mu_earth", c_double),
        ("mu_moon", c_double),
        ("earth_radius", c_double),
        ("j2", c_double),
        ("moon_distance", c_double),
        ("moon_period", c_double),
        ("moon_phase", c_double),
        ("softening", c_double),
        ("perturbation", c_int),
    ]


def _library_candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "build" / "src" / "kernels" / "libmultibody_solver_shared.so",
        root / "build" / "src" / "kernels" / "libmultibody_solver.so",
    ]
    env = os.environ.get("MBS_LIB")
    if env:
        candidates.insert(0, Path(env))
    return candidates


def load_library(path: str | os.PathLike[str] | None = None):
    """Load the CUDA shared library."""
    if path is not None:
        lib = ctypes.CDLL(str(path))
    else:
        for candidate in _library_candidates():
            if candidate.exists():
                lib = ctypes.CDLL(str(candidate))
                break
        else:
            raise FileNotFoundError(
                "Could not find libmultibody_solver_shared.so. "
                "Build with `cmake --build build` first."
            )

    lib.mbs_default_config.argtypes = [POINTER(MbsConfig)]
    lib.mbs_default_config.restype = None
    lib.mbs_propagate.argtypes = [
        POINTER(MbsState),
        c_int,
        c_int,
        POINTER(MbsConfig),
        POINTER(MbsState),
        POINTER(MbsState),
    ]
    lib.mbs_propagate.restype = c_int
    return lib


def default_config(lib, dt: float = 60.0, perturbation: int = 0, moon_distance: float = 0.0):
    config = MbsConfig()
    lib.mbs_default_config(ctypes.byref(config))
    config.dt = float(dt)
    config.perturbation = int(perturbation)
    config.moon_distance = float(moon_distance)
    return config


def propagate(
    lib,
    initial_r: tuple[float, float, float],
    initial_v: tuple[float, float, float],
    steps: int,
    dt: float,
    config: MbsConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate one body and return (positions, velocities) arrays."""
    if config is None:
        config = default_config(lib, dt=dt)
    else:
        config.dt = float(dt)

    initial = MbsState()
    initial.r.x, initial.r.y, initial.r.z = initial_r
    initial.v.x, initial.v.y, initial.v.z = initial_v

    n_frames = steps + 1
    trajectory = (MbsState * n_frames)()
    final = MbsState()

    rc = lib.mbs_propagate(
        ctypes.byref(initial),
        1,
        int(steps),
        ctypes.byref(config),
        ctypes.byref(final),
        trajectory,
    )
    if rc != 0:
        raise RuntimeError(f"mbs_propagate failed with code {rc}")

    positions = np.array([(s.r.x, s.r.y, s.r.z) for s in trajectory], dtype=float)
    velocities = np.array([(s.v.x, s.v.y, s.v.z) for s in trajectory], dtype=float)
    return positions, velocities


def load_reference(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load a reference trajectory CSV/parquet."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = {"t", "x", "y", "z", "vx", "vy", "vz"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Reference data missing columns: {sorted(missing)}")
    return df.sort_values("t").reset_index(drop=True)


def compare(
    reference: pd.DataFrame | str | os.PathLike[str],
    lib=None,
    config: MbsConfig | None = None,
) -> dict[str, Any]:
    """Compare propagated trajectory to a reference trajectory.

    Returns a metrics dictionary with position/velocity RMSE and max errors.
    """
    if lib is None:
        lib = load_library()
    if isinstance(reference, (str, os.PathLike)):
        reference = load_reference(reference)

    times = reference["t"].to_numpy(dtype=float)
    if len(times) < 2:
        raise ValueError("Reference trajectory must contain at least two samples")
    dts = np.diff(times)
    if not np.allclose(dts, dts[0], rtol=1e-9, atol=1e-9):
        raise ValueError("Reference samples must be uniformly spaced in time")
    dt = float(dts[0])
    steps = len(times) - 1

    initial_r = tuple(float(v) for v in reference[["x", "y", "z"]].iloc[0])
    initial_v = tuple(float(v) for v in reference[["vx", "vy", "vz"]].iloc[0])

    if config is None:
        config = default_config(lib, dt=dt)

    positions, velocities = propagate(lib, initial_r, initial_v, steps, dt, config)

    ref_pos = reference[["x", "y", "z"]].to_numpy(dtype=float)
    ref_vel = reference[["vx", "vy", "vz"]].to_numpy(dtype=float)

    pos_err = np.linalg.norm(positions - ref_pos, axis=1)
    vel_err = np.linalg.norm(velocities - ref_vel, axis=1)

    metrics = {
        "n_samples": len(times),
        "dt_seconds": dt,
        "position_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
        "position_max_m": float(np.max(pos_err)),
        "position_final_m": float(pos_err[-1]),
        "velocity_rmse_mps": float(np.sqrt(np.mean(vel_err**2))),
        "velocity_max_mps": float(np.max(vel_err)),
        "velocity_final_mps": float(vel_err[-1]),
    }
    return metrics


def make_synthetic_reference(
    steps: int = 100,
    dt: float = 60.0,
    radius: float = 7.0e6,
    mu: float = 3.986004418e14,
    perturbation: float = 0.0,
) -> pd.DataFrame:
    """Generate a simple two-body reference trajectory for tests/demos.

    ``perturbation`` adds a small tangential velocity offset (m/s) to the
    initial state, which makes the comparison exercise meaningful.
    """
    v0 = math.sqrt(mu / radius) + perturbation
    initial_r = (radius, 0.0, 0.0)
    initial_v = (0.0, v0, 0.0)

    lib = load_library()
    config = default_config(lib, dt=dt, moon_distance=0.0)
    positions, velocities = propagate(lib, initial_r, initial_v, steps, dt, config)

    times = np.arange(steps + 1, dtype=float) * dt
    return pd.DataFrame(
        {
            "t": times,
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "vx": velocities[:, 0],
            "vy": velocities[:, 1],
            "vz": velocities[:, 2],
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare CUDA propagator to reference trajectory")
    parser.add_argument("reference", help="CSV or parquet file with t,x,y,z,vx,vy,vz")
    parser.add_argument("--moon-distance", type=float, default=0.0)
    parser.add_argument("--perturbation", type=int, default=0)
    parser.add_argument("--out", default="comparison.json")
    args = parser.parse_args()

    lib = load_library()
    cfg = default_config(lib, moon_distance=args.moon_distance, perturbation=args.perturbation)
    metrics = compare(args.reference, lib=lib, config=cfg)
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))