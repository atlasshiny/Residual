#!/usr/bin/env python3
"""Fetch + convert an Artemis mission arc into the solver's reference frame.

Wraps fetch_horizons.py (Moon + spacecraft) and convert_spacecraft.py so a
single command produces data/reference_artemis<N>.csv ready for the residual
pipeline.

Usage:
    uv run python scripts/fetch_artemis.py --mission 1
    uv run python scripts/fetch_artemis.py --mission 2 --start "2026-04-02 02:00" --stop "2026-04-10"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MISSIONS = {
    1: {"target": -1023, "start": "2022-11-16 09:00", "stop": "2022-12-11"},
    2: {"target": -1024, "start": "2026-04-02 02:00", "stop": "2026-04-10 23:54"},
}


def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", type=int, choices=(1, 2), required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--stop", default=None)
    ap.add_argument("--step", default="10m")
    args = ap.parse_args()

    m = MISSIONS[args.mission]
    start = args.start or m["start"]
    stop = args.stop or m["stop"]
    target = m["target"]
    tag = f"artemis{args.mission}"

    raw_sc = REPO / "data" / f"raw_{tag}.csv"
    raw_moon = REPO / "data" / f"raw_moon_{tag}.csv"
    ref = REPO / "data" / f"reference_{tag}.csv"

    run([sys.executable, str(REPO / "scripts" / "fetch_horizons.py"),
         "--target", str(target), "--start", start, "--stop", stop,
         "--step", args.step, "--out", str(raw_sc)])
    run([sys.executable, str(REPO / "scripts" / "fetch_horizons.py"),
         "--target", "301", "--start", start, "--stop", stop,
         "--step", args.step, "--out", str(raw_moon)])
    run([sys.executable, str(REPO / "scripts" / "convert_spacecraft.py"),
         str(raw_sc), str(raw_moon), str(ref)])

    print(f"\nDone. Next: cargo run -p residual --example residual_pipeline -- {ref} data/residual_{tag}.csv")


if __name__ == "__main__":
    main()
