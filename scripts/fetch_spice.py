#!/usr/bin/env python3
"""Fetch an ephemeris arc from SPICE kernels and write a raw CSV.

This is the offline counterpart to ``scripts/fetch_horizons.py``: instead of
querying JPL Horizons over the network it reads one or more SPICE kernels
(.bsp / .tls / .tpc) with ``spiceypy`` and samples a target body relative to a
centre over a uniform time grid.

Output ``data/raw_spice.csv`` (or ``--out``) has the canonical columns::

    t,x,y,z,vx,vy,vz

where ``t`` is seconds from the first sample, positions are kilometres and
velocities are kilometres per second -- the exact contract consumed by
``scripts/convert_reference.py``.  A sidecar ``*.meta.json`` records the
kernels, body IDs, time range and frame so provenance lands in
``data/manifest.json``.

Requires the optional dependency group::

    uv sync --extra spice

and at least one SPICE kernel (e.g. ``de440.bsp`` plus ``naif0012.tls``).
Point at kernels with ``--kernel`` (repeatable) or the ``SPICE_KERNELS``
environment variable (colon/semicolon separated).

Only the Python standard library, pandas and spiceypy are used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "raw_spice.csv"
REQUIRED_COLUMNS = ["t", "x", "y", "z", "vx", "vy", "vz"]


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path)


def _parse_time(value: str) -> datetime:
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    raise ValueError(
        f"could not parse time {value!r}; use ISO-8601 (YYYY-MM-DD or "
        "YYYY-MM-DD HH:MM[:SS])"
    )


def _kernel_list(args_kernels: list[str]) -> list[Path]:
    raw: list[str] = list(args_kernels)
    env = os.environ.get("SPICE_KERNELS", "")
    if env:
        raw.extend(part for part in env.replace(";", ":").split(":") if part.strip())
    paths = [_resolve(p) for p in raw]
    if not paths:
        raise RuntimeError(
            "no SPICE kernels given; pass --kernel (repeatable) or set "
            "SPICE_KERNELS (colon/semicolon separated)"
        )
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(
            "SPICE kernel(s) not found: " + ", ".join(str(p) for p in missing)
        )
    return paths


def build_samples(
    kernels: list[Path],
    target: str,
    center: str,
    frame: str,
    start: datetime,
    stop: datetime,
    step_seconds: float,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    try:
        import spiceypy as spice
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "spiceypy is not installed; run `uv sync --extra spice`"
        ) from exc

    for kernel in kernels:
        spice.furnsh(str(kernel))

    try:
        # Convert UTC bounds to ephemeris time (ET ~ TDB).
        start_et = spice.utc2et(start.strftime("%Y-%m-%dT%H:%M:%S"))
        stop_et = spice.utc2et(stop.strftime("%Y-%m-%dT%H:%M:%S"))
    except Exception as exc:  # pragma: no cover - depends on kernel
        raise RuntimeError(f"could not convert UTC bounds to ET: {exc}") from exc

    if stop_et <= start_et:
        raise ValueError("stop must be after start")
    if step_seconds <= 0:
        raise ValueError("step must be positive")

    samples: list[dict[str, float]] = []
    ets: list[float] = []
    et = start_et
    # Sample on a uniform grid; the last point may fall slightly short of stop.
    while et <= stop_et + 1e-9:
        try:
            state, _lt = spice.spkezr(target, et, frame, "NONE", center)
        except Exception as exc:  # pragma: no cover - depends on kernel
            raise RuntimeError(
                f"spkezr({target!r}, {center!r}) failed at ET={et}: {exc}"
            ) from exc
        # state = [x, y, z, vx, vy, vz] in km / km/s.
        samples.append(
            {
                "x": float(state[0]),
                "y": float(state[1]),
                "z": float(state[2]),
                "vx": float(state[3]),
                "vy": float(state[4]),
                "vz": float(state[5]),
            }
        )
        ets.append(float(et))
        et += step_seconds

    if len(samples) < 2:
        raise RuntimeError("time range produced fewer than two samples")

    # Convert ET -> seconds from the first sample using the (monotonic) ET grid.
    t0 = ets[0]
    for sample, et_value in zip(samples, ets):
        sample["t"] = et_value - t0

    meta = {
        "provider": "NAIF SPICE (offline ephemeris)",
        "target": target,
        "center": center,
        "frame": frame,
        "kernels": [str(p) for p in kernels],
        "start_utc": start.isoformat(),
        "stop_utc": stop.isoformat(),
        "start_et": float(start_et),
        "stop_et": float(stop_et),
        "step_seconds": float(step_seconds),
        "samples": len(samples),
    }
    return samples, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch an ephemeris arc from SPICE kernels."
    )
    parser.add_argument(
        "--kernel",
        action="append",
        default=[],
        help="SPICE kernel path (repeatable); also SPICE_KERNELS env var",
    )
    parser.add_argument("--target", default="MOON", help="target body name or NAIF id (default: MOON)")
    parser.add_argument("--center", default="EARTH", help="centre body name or NAIF id (default: EARTH)")
    parser.add_argument("--frame", default="J2000", help="reference frame (default: J2000)")
    parser.add_argument("--start", required=True, help="start UTC (ISO-8601)")
    parser.add_argument(
        "--stop",
        default=None,
        help="stop UTC (ISO-8601); default: start + 30 days",
    )
    parser.add_argument("--step", default="600", help="step in seconds (default: 600)")
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT), help="output raw CSV (default: data/raw_spice.csv)"
    )
    parser.add_argument(
        "--meta", default=None, help="output metadata JSON (default: <out>.meta.json)"
    )
    args = parser.parse_args(argv)

    out_path = _resolve(args.out)
    meta_path = _resolve(args.meta) if args.meta else out_path.with_suffix(".meta.json")

    try:
        kernels = _kernel_list(args.kernel)
        start = _parse_time(args.start)
        if args.stop:
            stop = _parse_time(args.stop)
        else:
            stop = start + timedelta(days=30)
        step_seconds = float(args.step)

        samples, meta = build_samples(
            kernels, args.target, args.center, args.frame, start, stop, step_seconds
        )
        frame = pd.DataFrame(samples, columns=REQUIRED_COLUMNS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False, float_format="%.12g")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"fetch_spice: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote {len(samples)} samples to {out_path} "
        f"(dt={step_seconds:.0f}s, target={args.target}, center={args.center})"
    )
    print(f"wrote metadata to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
