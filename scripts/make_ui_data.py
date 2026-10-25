#!/usr/bin/env python3
"""Build the Tauri UI data bundle under tauri-src/src/data/."""

from __future__ import annotations

import csv
import json
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

    payload = {
        "points": points,
        "times": times,
        "units": "km",
        "count": len(points),
        "generated_utc": _now_iso(),
        "residual_convention": "reference - propagated",
    }
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
        f"corr={len(corrected) if have_corr else 0})"
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
