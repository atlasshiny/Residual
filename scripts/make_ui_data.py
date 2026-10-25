#!/usr/bin/env python3
"""Build the Tauri UI data bundle under tauri-src/src/data/."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tauri-src" / "src" / "data"
MAX_POINTS = 1500


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _warn(msg: str) -> None:
    print(f"WARN: {msg}", file=sys.stderr)


def _copy_verbatim(src: Path, dst: Path) -> bool:
    if not src.exists():
        _warn(f"missing source, skipping {dst.name}: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"wrote {dst} ({src.stat().st_size} bytes)")
    return True


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _component(columns: list[str], axis: str) -> str | None:
    """Map a model target column to a residual component (x/y/z)."""
    for candidate in (f"dr{axis}", axis, f"r{axis}"):
        if candidate in columns:
            return candidate
    return None


def _load_sun_track(
    times: list[float], path: Path | None = None
) -> tuple[list[list[float]], list[float]] | None:
    """Align ``data/reference_sun.csv`` to the trajectory samples.

    ``scripts/convert_reference.py --sun-raw`` writes the Sun's position in the
    same solver frame (and with the same time tags) as the Moon arc, so the
    viewer can show the true Earth->Sun direction at every epoch instead of a
    fixed placeholder.  Returns unit directions and ranges in kilometres, or
    ``None`` when the file is missing or does not line up with the arc.
    """
    path = path or ROOT / "data" / "reference_sun.csv"
    if not path.exists():
        # The canonical raw Moon and this Sun ephemeris cover the same 2025
        # training/reference interval. Rotate the geocentric Sun vectors into
        # the canonical solver frame with the canonical raw Moon basis.
        raw_sun = ROOT / "data" / "raw_sun_moon_training.csv"
        raw_moon = ROOT / "data" / "raw_moon.csv"
        if raw_sun.exists() and raw_moon.exists():
            moon_rows = _read_csv(raw_moon)
            sun_rows = _read_csv(raw_sun)
            if moon_rows and sun_rows:
                p0 = [float(moon_rows[0][k]) for k in ("x", "y", "z")]
                v0 = [float(moon_rows[0][k]) for k in ("vx", "vy", "vz")]
                cross = [
                    p0[1] * v0[2] - p0[2] * v0[1],
                    p0[2] * v0[0] - p0[0] * v0[2],
                    p0[0] * v0[1] - p0[1] * v0[0],
                ]
                norm = lambda v: math.sqrt(sum(x * x for x in v))
                e1 = [v / norm(p0) for v in p0]
                e3 = [v / norm(cross) for v in cross]
                e2 = [
                    e3[1] * e1[2] - e3[2] * e1[1],
                    e3[2] * e1[0] - e3[0] * e1[2],
                    e3[0] * e1[1] - e3[1] * e1[0],
                ]
                basis = [e1, e2, e3]
                sun_times = [float(row["t"]) for row in sun_rows]
                if min(times) < sun_times[0] or max(times) > sun_times[-1]:
                    _warn(
                        "raw Sun ephemeris does not cover the trajectory interval; "
                        "skipping the Sun marker"
                    )
                    return None
                directions, ranges_km = [], []
                missing = 0
                for t in times:
                    vector = [
                        _interp_track(
                            sun_times,
                            [float(row[axis]) for row in sun_rows],
                            t,
                        )
                        for axis in ("x", "y", "z")
                    ]
                    rotated = [sum(basis[i][j] * vector[j] for j in range(3)) for i in range(3)]
                    distance = norm(rotated)
                    if not math.isfinite(distance) or distance == 0:
                        missing += 1
                        directions.append(directions[-1] if directions else [0.0, 0.0, 0.0])
                        ranges_km.append(ranges_km[-1] if ranges_km else 0.0)
                    else:
                        directions.append([v / distance for v in rotated])
                        ranges_km.append(distance)
                if missing <= max(1, len(times) // 100):
                    if missing:
                        _warn(f"{missing} raw Sun samples were filled from the previous epoch")
                    return directions, ranges_km
        _warn(f"missing source, no Sun in the scene: {path}")
        return None
    rows = _read_csv(path)
    if not rows:
        _warn(f"empty source, no Sun in the scene: {path}")
        return None
    by_time = {round(float(r["t"]), 6): r for r in rows}

    directions: list[list[float]] = []
    ranges_km: list[float] = []
    missing = 0
    for t in times:
        row = by_time.get(round(float(t), 6))
        vector = None
        if row is not None:
            try:
                vector = tuple(float(row[k]) for k in ("x", "y", "z"))
            except (KeyError, ValueError):
                vector = None
        if vector is None:
            missing += 1
            directions.append(directions[-1] if directions else [0.0, 0.0, 0.0])
            ranges_km.append(ranges_km[-1] if ranges_km else 0.0)
            continue
        distance = math.sqrt(sum(v * v for v in vector))
        if not math.isfinite(distance) or distance == 0.0:
            missing += 1
            directions.append(directions[-1] if directions else [0.0, 0.0, 0.0])
            ranges_km.append(ranges_km[-1] if ranges_km else 0.0)
            continue
        directions.append([v / distance for v in vector])
        ranges_km.append(distance / 1000.0)

    if missing > max(1, len(times) // 100):
        _warn(
            f"{missing}/{len(times)} Sun samples do not align with the arc; "
            "skipping the Sun in the scene"
        )
        return None
    if missing:
        _warn(f"{missing}/{len(times)} Sun samples were filled from the previous epoch")
    return directions, ranges_km


def _interp_track(times: list[float], values: list[float], t: float) -> float:
    """Linear interpolation matching numpy.interp without requiring NumPy."""
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    lo, hi = 0, len(times) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= t:
            lo = mid
        else:
            hi = mid
    fraction = (t - times[lo]) / (times[hi] - times[lo])
    return values[lo] + fraction * (values[hi] - values[lo])


def build_trajectory() -> None:
    src = ROOT / "data" / "reference.csv"
    dst = OUT_DIR / "trajectory.json"
    if not src.exists():
        _warn(f"missing source, skipping {dst.name}: {src}")
        return
    rows = _read_csv(src)
    n = len(rows)
    if n == 0:
        _warn(f"empty source, skipping {dst.name}: {src}")
        return
    if n <= MAX_POINTS:
        stride = 1
    else:
        stride = (n + MAX_POINTS - 1) // MAX_POINTS
    selected = rows[::stride][:MAX_POINTS]

    # residual = reference - propagated  (see crates/residual), so the
    # propagated arc is reference - dr. Units are metres in the pipeline and
    # kilometres in the UI scene.
    residual_path = ROOT / "data" / "residual_table.csv"
    residual: dict[float, dict[str, str]] = {}
    if residual_path.exists():
        for r in _read_csv(residual_path):
            residual[round(float(r["t"]), 6)] = r
    else:
        _warn(f"missing source, no propagated overlay: {residual_path}")

    # PINN-corrected = propagated + predicted residual. Only possible when the
    # prediction file covers all three position-residual components.
    predictions_path = ROOT / "ml" / "pinn" / "pinn_predictions.json"
    predicted: dict[float, dict[str, float]] = {}
    predicted_cols: tuple[str, str, str] | None = None
    if predictions_path.exists():
        payload = json.loads(predictions_path.read_text())
        model_rows = payload.get("rows") or []
        cols = payload.get("target_columns") or []
        mapping = tuple(_component(cols, axis) for axis in "xyz")
        if all(mapping):
            predicted_cols = mapping  # type: ignore[assignment]
            for r in model_rows:
                key = round(float(r["t"]), 6)
                values = r.get("pinn") or {}
                if all(c in values for c in mapping):
                    predicted[key] = {c: float(values[c]) for c in mapping}
        else:
            _warn(
                "predictions cover "
                f"{cols or 'no columns'}; corrected overlay needs drx/dry/drz"
            )
    else:
        _warn(f"missing source, no corrected overlay: {predictions_path}")

    points, times = [], []
    propagated, corrected = [], []
    have_prop = bool(residual)
    have_corr = bool(predicted_cols) and bool(predicted)
    for r in selected:
        t = float(r["t"])
        key = round(t, 6)
        ref = [float(r["x"]), float(r["y"]), float(r["z"])]
        points.append([v / 1000.0 for v in ref])
        times.append(t)
        base = None
        res_row = residual.get(key)
        if res_row is not None and all(f"dr{a}" in res_row for a in "xyz"):
            base = [
                ref[i] - float(res_row[f"dr{a}"]) for i, a in enumerate("xyz")
            ]
            propagated.append([v / 1000.0 for v in base])
        else:
            have_prop = False
        if base is not None and predicted_cols and key in predicted:
            pred = predicted[key]
            corrected.append(
                [
                    (base[i] + pred[col]) / 1000.0
                    for i, col in enumerate(predicted_cols)
                ]
            )
        else:
            have_corr = False

    sun = _load_sun_track(times)
    payload = {
        "points": points,
        "times": times,
        "units": "km",
        "count": len(points),
        "generated_utc": _now_iso(),
        "residual_convention": "reference - propagated",
    }
    if sun is not None:
        payload["sun_directions"] = sun[0]
        payload["sun_range_km"] = sun[1]
    if have_prop and len(propagated) == len(points):
        payload["points_propagated"] = propagated
        payload["overlay_sources"] = {"points_propagated": "data/residual_table.csv"}
    if have_corr and len(corrected) == len(points):
        payload["points_corrected"] = corrected
        payload.setdefault("overlay_sources", {})["points_corrected"] = (
            "propagated + PINN-predicted residual"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {dst} ({len(points)} points, stride={stride}, "
        f"prop={len(propagated) if have_prop else 0}, "
        f"corr={len(corrected) if have_corr else 0}, "
        f"sun={len(sun[0]) if sun is not None else 0})"
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = _now_iso()

    build_trajectory()

    _copy_verbatim(ROOT / "ml" / "pinn" / "pinn_result.json", OUT_DIR / "pinn_result.json")
    _copy_verbatim(ROOT / "ml" / "pinn" / "pinn_predictions.json", OUT_DIR / "pinn_predictions.json")
    _copy_verbatim(ROOT / "data" / "manifest.json", OUT_DIR / "manifest.json")

    # ensemble (optional)
    ens_src = ROOT / "tauri-src" / "src" / "data" / "ensemble.json"
    if ens_src.is_file():
        if ens_src.resolve() != (OUT_DIR / "ensemble.json").resolve():
            _copy_verbatim(ens_src, OUT_DIR / "ensemble.json")
        else:
            print(f"wrote {OUT_DIR / 'ensemble.json'} (already in place)")

    files = ["trajectory.json", "pinn_result.json", "pinn_predictions.json", "manifest.json"]
    sources = {
        "reference": "data/reference.csv",
        "pinn_result": "ml/pinn/pinn_result.json",
        "pinn_predictions": "ml/pinn/pinn_predictions.json",
        "manifest": "data/manifest.json",
    }
    if (OUT_DIR / "ensemble.json").is_file():
        files.append("ensemble.json")
        sources["ensemble"] = "tauri-src/src/data/ensemble.json"
    if (ROOT / "data" / "reference_sun.csv").exists():
        # The Sun track is embedded in trajectory.json rather than shipped as
        # its own bundle file.
        sources["sun"] = "data/reference_sun.csv (embedded in trajectory.json)"
    elif (
        (ROOT / "data" / "raw_sun_moon_training.csv").exists()
        and (ROOT / "data" / "raw_moon.csv").exists()
    ):
        sources["sun"] = (
            "data/raw_sun_moon_training.csv rotated with data/raw_moon.csv "
            "(embedded in trajectory.json)"
        )

    ui_manifest = {
        "schema": "ui-data/v1",
        "generated_utc": generated,
        "sources": sources,
        "files": files,
    }
    dst = OUT_DIR / "ui_manifest.json"
    dst.write_text(json.dumps(ui_manifest, indent=2) + "\n")
    print(f"wrote {dst}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
