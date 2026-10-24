#!/usr/bin/env python3
"""Convert a raw Horizons Moon arc into the solver's reference frame.

Reads ``data/raw_moon.csv`` (kilometres, seconds) produced by
``scripts/fetch_horizons.py`` and writes ``data/reference.csv`` in SI units
(metres, seconds) with the canonical columns ``t,x,y,z,vx,vy,vz``.

The raw file contains the *geocentric* Moon vector (Moon relative to the
Earth body centre).  The CUDA/Rust propagators work in an Earth-Moon
barycentric frame, so each sample is split using the standard two-body mass
fractions::

    earth = -D * mu_moon / total_mu
    moon  =  D * mu_earth / total_mu

where ``D`` is the Earth-Moon separation vector and ``total_mu`` is
``mu_earth + mu_moon``.  By default the separation magnitude is normalised
to the propagator's nominal ``D = 3.844e8 m`` so the reference is directly
comparable to its circular ephemeris.  Pass ``--preserve-distance`` to keep
the instantaneous Horizons separation instead.

The result is then rotated into the solver's right-handed frame: the epoch
Earth -> Moon direction becomes ``+x`` and the orbital angular momentum
becomes ``+z``.  Pass ``--no-align`` to keep the Horizons J2000 ecliptic
orientation.

Only the Python standard library and pandas are used.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "data" / "raw_moon.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "reference.csv"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.json"

REQUIRED_COLUMNS = ["t", "x", "y", "z", "vx", "vy", "vz"]

# Same constants as crates/dynamics and src/kernels/multibody_solver.cuh.
MU_EARTH = 3.986004418e14  # m^3 / s^2
MU_MOON = 4.9048695e12  # m^3 / s^2
TOTAL_MU = MU_EARTH + MU_MOON
EARTH_MOON_DISTANCE = 3.844e8  # m, nominal solver separation


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path)


def _display_path(path: Path) -> str:
    """Prefer a repo-relative path in the manifest."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _norm(vector: Iterable[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _dot(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _scale(vector: Iterable[float], factor: float) -> tuple[float, float, float]:
    values = tuple(float(component) * factor for component in vector)
    return values[0], values[1], values[2]


def _unit(vector: Iterable[float], label: str) -> tuple[float, float, float]:
    magnitude = _norm(vector)
    if magnitude == 0.0 or not math.isfinite(magnitude):
        raise RuntimeError(f"cannot normalise degenerate {label}: {tuple(vector)!r}")
    return _scale(vector, 1.0 / magnitude)


def load_raw(path: Path) -> tuple[pd.DataFrame, float]:
    """Read and validate a raw geocentric Moon CSV."""
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"raw file not found: {path}; run `make fetch-data` first"
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise RuntimeError(f"raw file is empty: {path}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(
            f"raw file {path} is missing columns {missing}; expected "
            f"{REQUIRED_COLUMNS}"
        )

    numeric = frame[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid = numeric.isna().any(axis=1)
    if invalid.any():
        raise RuntimeError(
            f"raw file has {int(invalid.sum())} row(s) with non-numeric values"
        )

    frame = (
        numeric.sort_values("t")
        .drop_duplicates(subset="t", keep="first")
        .reset_index(drop=True)
    )
    if len(frame) < 2:
        raise RuntimeError("raw file must contain at least two samples")

    times = [float(value) for value in frame["t"]]
    if any(not math.isfinite(value) for value in times):
        raise RuntimeError("raw time column contains non-finite values")
    differences = [
        later - earlier for earlier, later in itertools.pairwise(times)
    ]
    if any(difference <= 0.0 for difference in differences):
        raise RuntimeError("raw samples are not strictly increasing in time")

    step = statistics.median(differences)
    tolerance = max(1.0e-3, abs(step) * 1.0e-9)
    if any(abs(difference - step) > tolerance for difference in differences):
        raise RuntimeError(
            "raw samples are not uniformly spaced; the fixed-step propagator "
            "cannot consume this reference"
        )

    return frame, float(step)


def to_solver_frame(
    raw: pd.DataFrame, preserve_distance: bool, align: bool
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Convert km / km-s geocentric vectors to barycentric SI states."""
    fraction_moon = MU_MOON / TOTAL_MU
    fraction_earth = MU_EARTH / TOTAL_MU

    times: list[float] = []
    moon_positions: list[tuple[float, float, float]] = []
    moon_velocities: list[tuple[float, float, float]] = []
    raw_separations: list[float] = []

    for row in raw.itertuples(index=False):
        position_km = (float(row.x), float(row.y), float(row.z))
        velocity_km_s = (float(row.vx), float(row.vy), float(row.vz))

        # km -> m and km/s -> m/s.
        position = _scale(position_km, 1000.0)
        velocity = _scale(velocity_km_s, 1000.0)

        separation = _norm(position)
        if separation == 0.0 or not math.isfinite(separation):
            raise RuntimeError("raw position vector has zero or non-finite length")
        raw_separations.append(separation)

        if not preserve_distance:
            # Scale the whole state so the reference matches the propagator's
            # nominal circular separation.  Scaling r and v by the same factor
            # preserves the osculating orbit shape.
            factor = EARTH_MOON_DISTANCE / separation
            position = _scale(position, factor)
            velocity = _scale(velocity, factor)

        # Barycentric split: earth = -D * mu_moon / total_mu,
        # moon = +D * mu_earth / total_mu, with D the separation vector.
        earth_position = _scale(position, -fraction_moon)
        moon_position = _scale(position, fraction_earth)
        moon_velocity = _scale(velocity, fraction_earth)

        # Mass-weighted centre of mass must stay at the barycentre.  The raw
        # positions do not sum to zero because the Earth and Moon have very
        # different masses.
        weighted = tuple(
            MU_EARTH * earth_position[axis] + MU_MOON * moon_position[axis]
            for axis in range(3)
        )
        if _norm(weighted) > 1.0e-9 * TOTAL_MU * _norm(position):
            raise RuntimeError(
                "barycentric split did not cancel to the origin; check the "
                "gravitational parameters"
            )

        times.append(float(row.t))
        moon_positions.append(moon_position)
        moon_velocities.append(moon_velocity)

    notes: dict[str, object] = {
        "distance_scaling": (
            "preserved instantaneous Horizons separation"
            if preserve_distance
            else f"normalised to nominal D={EARTH_MOON_DISTANCE:g} m"
        )
    }

    if align:
        # Right-handed solver basis:
        #   e1 = epoch Earth -> Moon direction (becomes +x)
        #   e3 = orbital angular momentum (becomes +z)
        #   e2 = e3 x e1 (direction of motion at epoch, +y)
        e1 = _unit(moon_positions[0], "epoch Moon position")
        angular_momentum = _cross(moon_positions[0], moon_velocities[0])
        e3 = _unit(angular_momentum, "epoch orbital angular momentum")
        e2 = _unit(_cross(e3, e1), "epoch transverse axis")

        def rotate(
            vector: tuple[float, float, float]
        ) -> tuple[float, float, float]:
            return (_dot(vector, e1), _dot(vector, e2), _dot(vector, e3))

        moon_positions = [rotate(vector) for vector in moon_positions]
        moon_velocities = [rotate(vector) for vector in moon_velocities]
        notes["orientation"] = (
            "rotated so the epoch Moon vector is +x and the Moon's orbital "
            "angular momentum is +z"
        )
    else:
        notes["orientation"] = "kept Horizons J2000 ecliptic orientation"

    converted = pd.DataFrame(
        {
            "t": times,
            "x": [value[0] for value in moon_positions],
            "y": [value[1] for value in moon_positions],
            "z": [value[2] for value in moon_positions],
            "vx": [value[0] for value in moon_velocities],
            "vy": [value[1] for value in moon_velocities],
            "vz": [value[2] for value in moon_velocities],
        }
    )
    checks = {
        "samples": len(converted),
        "raw_separation_m": {
            "min": min(raw_separations),
            "max": max(raw_separations),
            "mean": sum(raw_separations) / len(raw_separations),
        },
        "aligned": bool(align),
    }
    return converted, {**notes, **checks}


def load_sidecar_meta(raw_path: Path) -> dict[str, object]:
    meta_path = raw_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"convert_reference: ignoring unreadable metadata {meta_path}: {exc}",
            file=sys.stderr,
        )
        return {}
    return data if isinstance(data, dict) else {}


def build_manifest(
    raw_path: Path,
    reference_path: Path,
    step_seconds: float,
    conversion: dict[str, object],
    meta: dict[str, object],
) -> dict[str, object]:
    source = {
        "provider": "JPL Horizons (NASA/JPL Solar System Dynamics)",
        "target": meta.get("target", "301 (Moon)"),
        "center": meta.get("center", "500@399 (Earth body centre)"),
        "api_url": meta.get("api_url"),
        "query": meta.get("query"),
        "retrieved_utc": meta.get("retrieved_utc"),
    }
    epoch = {
        "time_scale": "TDB",
        "t_zero": "first sample in data/reference.csv",
        "first_jd_tdb": meta.get("first_jd_tdb"),
        "last_jd_tdb": meta.get("last_jd_tdb"),
        "start_tdb": meta.get("start_tdb"),
    }
    cadence = {
        "step_seconds": float(step_seconds),
        "step": meta.get("step"),
        "samples": int(conversion["samples"]),
        "uniform": True,
    }
    manifest = {
        "schema_version": 1,
        "dataset": "earth-moon-barycentric-moon-reference",
        "generated_by": "scripts/convert_reference.py",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "frame": {
            "name": "earth-moon-barycentric",
            "origin": "Earth-Moon barycentre",
            "orientation": conversion.get("orientation", "solver frame"),
            "handedness": "right-handed",
            "propagator_model": (
                "circular Earth-Moon two-body ephemeris used by the CUDA/Rust "
                "solvers"
            ),
            "nominal_earth_moon_distance_m": EARTH_MOON_DISTANCE,
        },
        "epoch": epoch,
        "cadence": cadence,
        "units": {
            "t": "s",
            "x": "m",
            "y": "m",
            "z": "m",
            "vx": "m/s",
            "vy": "m/s",
            "vz": "m/s",
        },
        "constants": {
            "mu_earth_m3_s2": MU_EARTH,
            "mu_moon_m3_s2": MU_MOON,
            "total_mu_m3_s2": TOTAL_MU,
            "earth_moon_distance_m": EARTH_MOON_DISTANCE,
        },
        "conversion": {
            "input": "geocentric Moon vector (Moon relative to Earth body centre)",
            "output": "Moon state in the Earth-Moon barycentric solver frame",
            "position_units": "km -> m (x1000)",
            "velocity_units": "km/s -> m/s (x1000)",
            "barycentric_split": (
                "earth = -D*mu_moon/total_mu * u; moon = "
                "D*mu_earth/total_mu * u, where u is the unit Earth->Moon "
                "vector and D is the separation"
            ),
            "distance_scaling": conversion.get("distance_scaling"),
            "orientation": conversion.get("orientation"),
            "checks": conversion.get("raw_separation_m"),
        },
        "files": {
            "raw": _display_path(raw_path),
            "reference": _display_path(reference_path),
            "metadata": _display_path(raw_path.with_suffix(".meta.json")),
        },
    }
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw Horizons Moon CSV to the solver's SI frame."
    )
    parser.add_argument(
        "--raw", default=str(DEFAULT_RAW), help="input raw CSV (default: data/raw_moon.csv)"
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="output reference CSV (default: data/reference.csv)",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="output manifest JSON (default: data/manifest.json)",
    )
    parser.add_argument(
        "--preserve-distance",
        action="store_true",
        help="keep the instantaneous Earth-Moon distance instead of D=3.844e8 m",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="do not rotate the arc into the solver's +x/+z orientation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_path = _resolve(args.raw)
    out_path = _resolve(args.out)
    manifest_path = _resolve(args.manifest)

    try:
        raw, step_seconds = load_raw(raw_path)
        reference, conversion = to_solver_frame(
            raw,
            preserve_distance=args.preserve_distance,
            align=not args.no_align,
        )
        meta = load_sidecar_meta(raw_path)
        manifest = build_manifest(
            raw_path, out_path, step_seconds, conversion, meta
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        reference.to_csv(out_path, index=False, float_format="%.12g")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"convert_reference: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote {len(reference)} samples to {out_path} "
        f"(dt={step_seconds:.0f}s, frame={manifest['frame']['name']})"
    )
    print(f"wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
