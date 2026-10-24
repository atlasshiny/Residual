#!/usr/bin/env python3
"""Fetch a short Moon arc from JPL Horizons and write a raw CSV.

The query is deliberately Earth-centred: Horizons is asked for the Moon
(``301``) relative to the Earth *body centre* (``500@399``) over a short
window.  Bare ``399`` is not used because Horizons resolves it to a default
topocentric site; the ``500@`` prefix is required for body-centre vectors.

Output ``data/raw_moon.csv`` has the canonical columns::

    t,x,y,z,vx,vy,vz

where ``t`` is seconds from the first sample, positions are kilometres and
velocities are kilometres per second.  A sidecar ``*.meta.json`` records the
exact query so ``scripts/convert_reference.py`` can copy provenance into
``data/manifest.json``.

Only the Python standard library and pandas are used.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
import sys
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "raw_moon.csv"
DEFAULT_TARGET = "301"  # Moon
# ``500@399`` = body centre of Earth.  Plain ``399`` selects a topocentric
# default site (Kushiro at the time of writing), which would silently add a
# ~6400 km Earth-radius offset and violate the Earth-centred contract.
DEFAULT_CENTER = "500@399"
USER_AGENT = "gatorhacks-v-data/0.1 (+https://github.com/gatorhacksv)"
REQUIRED_COLUMNS = ["t", "x", "y", "z", "vx", "vy", "vz"]


def _resolve(path_value: str | Path) -> Path:
    """Resolve a CLI path against the current directory (Makefile root)."""
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path)


def _default_stop(start: str) -> str:
    """Return ``start + 1 day``, preserving the caller's format."""
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(start, fmt)
        except ValueError:
            continue
        return (parsed + timedelta(days=1)).strftime(fmt)
    raise ValueError(
        f"could not parse start time {start!r}; use YYYY-MM-DD or "
        "YYYY-MM-DD HH:MM"
    )


def build_query(
    start: str,
    stop: str,
    step: str,
    target: str = DEFAULT_TARGET,
    center: str = DEFAULT_CENTER,
) -> dict[str, str]:
    """Build the Horizons API query dictionary (quoted values included)."""
    return {
        "format": "text",
        "COMMAND": f"'{target}'",
        "CENTER": f"'{center}'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step}'",
        "VEC_TABLE": "'2'",  # position + velocity
        "REF_PLANE": "'ECLIPTIC'",  # J2000 ecliptic: close to the solver's xy plane
        "REF_SYSTEM": "'ICRF'",
        "OUT_UNITS": "'KM-S'",
        "CSV_FORMAT": "'YES'",
        "CAL_FORMAT": "'JD'",  # JDTDB in column 1; calendar date in column 2
        "TIME_DIGITS": "'SECONDS'",
        "VEC_CORR": "'NONE'",  # geometric states, no light-time correction
        "OBJ_DATA": "'NO'",
    }


def build_url(query: dict[str, str]) -> str:
    return f"{HORIZONS_API}?{urllib.parse.urlencode(query)}"


def fetch_text(url: str, timeout: float = 60.0, retries: int = 3) -> str:
    """GET ``url`` with a small retry loop for transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "text/plain"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            # 4xx means the query itself is wrong; retrying will not help.
            if 400 <= exc.code < 500:
                raise RuntimeError(
                    f"Horizons rejected the query (HTTP {exc.code}): {body}"
                ) from exc
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt < retries:
            _time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"could not reach JPL Horizons at {url}: {last_error}")


def _as_float(token: str) -> float | None:
    """Parse a Horizons numeric token, returning ``None`` for placeholders."""
    token = token.strip().replace("D", "E").replace("d", "e")
    if not token or token.lower() in {"n.a.", "na", "nan", "---"}:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def extract_ephemeris_block(text: str) -> str:
    """Return the text between the ``$$SOE``/``$$EOE`` sentinels."""
    if "$$SOE" not in text or "$$EOE" not in text:
        # Horizons often returns a helpful plain-text error with HTTP 200.
        snippet = " ".join(text.split())[:600]
        raise RuntimeError(f"Horizons response has no ephemeris block: {snippet}")
    return text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]


def parse_block(block: str) -> tuple[list[float], list[list[float]]]:
    """Parse JDTDB + six state components from a CSV-format Horizons block.

    Rows have the shape ``JDTDB, Calendar Date, X, Y, Z, VX, VY, VZ,``.
    The calendar-date field is intentionally skipped: it can contain spaces
    and is not needed once we have JDTDB.
    """
    jd_values: list[float] = []
    states: list[list[float]] = []

    for line_number, fields in enumerate(csv.reader(io.StringIO(block)), 1):
        if not fields or not any(field.strip() for field in fields):
            continue
        if len(fields) < 8:
            raise RuntimeError(
                f"unexpected Horizons row {line_number} (expected >=8 fields): "
                f"{','.join(fields)!r}"
            )
        jd = _as_float(fields[0])
        state = [_as_float(field) for field in fields[2:8]]
        if jd is None or any(value is None for value in state):
            # Header/units lines and placeholder rows are not data.
            continue
        jd_values.append(jd)
        states.append([float(value) for value in state])

    if len(jd_values) < 2:
        raise RuntimeError(
            f"Horizons returned {len(jd_values)} usable vector(s); "
            "expected at least two"
        )
    return jd_values, states


def _step_seconds(step: str) -> float | None:
    """Parse a Horizons step such as ``10m`` or ``1d`` into seconds."""
    token = step.strip().lower()
    units = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    try:
        if token and token[-1] in units:
            return float(token[:-1]) * units[token[-1]]
        # Horizons treats a bare number as days.
        return float(token) * 86400.0
    except ValueError:
        return None


def build_frame(
    jd_values: list[float],
    states: list[list[float]],
    expected_step: str | None = None,
) -> pd.DataFrame:
    """Convert JDTDB states to seconds-from-start and validate the cadence."""
    if len(jd_values) < 2:
        raise RuntimeError("need at least two JDTDB samples")

    # Horizons prints JDTDB with finite decimal precision, which introduces
    # tens-of-microseconds jitter into per-row differences.  Derive one nominal
    # step from the full span and emit exact multiples so downstream fixed-step
    # propagators see a clean cadence.
    span_seconds = (jd_values[-1] - jd_values[0]) * 86400.0
    step_seconds = span_seconds / (len(jd_values) - 1)
    if not math.isfinite(step_seconds) or step_seconds <= 0.0:
        raise RuntimeError("Horizons returned a non-positive time step")

    requested_step = _step_seconds(expected_step) if expected_step else None
    # Snap away the ~microsecond JDTDB print rounding when the caller's step
    # is an exact unit expression ("10m" -> 600 s).
    if (
        requested_step is not None
        and requested_step > 0.0
        and abs(step_seconds - requested_step) <= max(0.01, requested_step * 1e-6)
    ):
        step_seconds = requested_step

    measured = [
        (later - earlier) * 86400.0
        for earlier, later in itertools.pairwise(jd_values)
    ]
    if any(abs(diff - step_seconds) > 0.5 for diff in measured):
        raise RuntimeError(
            "Horizons returned a non-uniform time step; refusing to build a "
            "reference that the fixed-step propagator cannot consume"
        )
    times = [index * step_seconds for index in range(len(jd_values))]

    frame = pd.DataFrame(times, columns=["t"])
    frame[["x", "y", "z", "vx", "vy", "vz"]] = pd.DataFrame(
        states, columns=["x", "y", "z", "vx", "vy", "vz"]
    )
    return frame[REQUIRED_COLUMNS]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a short Moon arc from JPL Horizons (Earth-centred)."
    )
    parser.add_argument(
        "--start",
        default="2025-01-01",
        help="start epoch in TDB, e.g. '2025-01-01' or '2025-01-01 00:00'",
    )
    parser.add_argument(
        "--stop",
        default=None,
        help="stop epoch in TDB (default: start + 1 day)",
    )
    parser.add_argument(
        "--step", default="10m", help="Horizons step size (default: 10m)"
    )
    parser.add_argument(
        "--target", default=DEFAULT_TARGET, help="Horizons target id (default: 301 Moon)"
    )
    parser.add_argument(
        "--center",
        default=DEFAULT_CENTER,
        help="Horizons centre id (default: 500@399, Earth body centre)",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="raw CSV path (default: data/raw_moon.csv)",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="HTTP timeout in seconds"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stop = args.stop or _default_stop(args.start)
        query = build_query(
            start=args.start,
            stop=stop,
            step=args.step,
            target=args.target,
            center=args.center,
        )
        url = build_url(query)
        print(f"querying JPL Horizons for target {args.target} / centre {args.center} ...")
        text = fetch_text(url, timeout=args.timeout)
        block = extract_ephemeris_block(text)
        jd_values, states = parse_block(block)
        frame = build_frame(jd_values, states, expected_step=args.step)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"fetch_horizons: {exc}", file=sys.stderr)
        return 1

    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False, float_format="%.12g")

    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "source": "JPL Horizons API",
        "api_url": url,
        "target": f"{args.target} (Moon)",
        "center": f"{args.center} (Earth body centre)",
        "query": {key: value.strip("'") for key, value in query.items()},
        "start_tdb": args.start,
        "stop_tdb": stop,
        "step": args.step,
        "first_jd_tdb": jd_values[0],
        "last_jd_tdb": jd_values[-1],
        "samples": len(frame),
        "step_seconds": float(frame["t"].iloc[1] - frame["t"].iloc[0]),
        "units": {
            "t": "s from first sample",
            "x": "km",
            "y": "km",
            "z": "km",
            "vx": "km/s",
            "vy": "km/s",
            "vz": "km/s",
        },
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(
        f"wrote {len(frame)} samples to {out_path} "
        f"(dt={meta['step_seconds']:.0f}s, metadata: {meta_path.name})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
