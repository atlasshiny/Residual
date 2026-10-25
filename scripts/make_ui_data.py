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


def build_trajectory() -> None:
    src = ROOT / "data" / "reference.csv"
    dst = OUT_DIR / "trajectory.json"
    if not src.exists():
        _warn(f"missing source, skipping {dst.name}: {src}")
        return
    rows = []
    with src.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    n = len(rows)
    if n == 0:
        _warn(f"empty source, skipping {dst.name}: {src}")
        return
    if n <= MAX_POINTS:
        stride = 1
    else:
        stride = (n + MAX_POINTS - 1) // MAX_POINTS
    points = []
    times = []
    for r in rows[::stride]:
        x = float(r["x"])
        y = float(r["y"])
        z = float(r["z"])
        points.append([x, y, z])
        times.append(float(r["t"]))
    if len(points) > MAX_POINTS:
        points = points[:MAX_POINTS]
        times = times[:MAX_POINTS]
    payload = {
        "points": points,
        "times": times,
        "units": "km",
        "count": len(points),
        "generated_utc": _now_iso(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {dst} ({len(points)} points, stride={stride})")


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
