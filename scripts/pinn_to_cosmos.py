#!/usr/bin/env python3
"""Reconstruct a PINN-corrected trajectory and turn it into a Cosmos job.

Reads a canonical reference CSV (``t,x,y,z,vx,vy,vz`` in SI units) and a
prediction CSV produced by ``ml.pinn.run`` (``--predictions-csv-out``), aligns
them by timestamp, and reconstructs the corrected trajectory::

    corrected = reference + (predicted_residual - actual_residual)

The corrected trajectory is written as ``corrected_trajectory.csv`` and then
handed to the Rust ``posebuilder`` example, which produces ``actions.json`` and
``job.json`` in the Cosmos file contract (``docs/cosmos-contract.md``).

Optionally submits the 9D actions to a vLLM-Omni server (``--serve-url``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_REF = ["t", "x", "y", "z", "vx", "vy", "vz"]
POS_COLS = ["drx", "dry", "drz"]
POS_PRED_COLS = ["drx_pred", "dry_pred", "drz_pred"]


def _resolve(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path)


def _load_and_align(reference_path: Path, predictions_path: Path) -> pd.DataFrame:
    ref = pd.read_csv(reference_path)
    pred = pd.read_csv(predictions_path)

    for col in REQUIRED_REF:
        if col not in ref.columns:
            raise ValueError(f"reference missing column {col!r}")
    for col in POS_COLS + POS_PRED_COLS:
        if col not in pred.columns:
            raise ValueError(f"predictions missing column {col!r}")
    if "t" not in pred.columns:
        raise ValueError("predictions missing column 't'")

    # Strict alignment: every reference timestamp must have a prediction.
    merged = ref.merge(pred[["t"] + POS_COLS + POS_PRED_COLS], on="t", how="left", validate="one_to_one")
    missing = merged[POS_PRED_COLS[0]].isna()
    if missing.any():
        bad = merged.loc[missing, "t"].tolist()
        raise ValueError(
            f"predictions do not cover {int(missing.sum())} reference timestamp(s); "
            f"first missing: {bad[:5]}"
        )

    return merged


def _reconstruct(merged: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["t"] = merged["t"]
    for axis, col in zip("xyz", ("x", "y", "z")):
        res_col = f"dr{axis}"
        pred_col = f"dr{axis}_pred"
        out[col] = merged[col] + (merged[pred_col] - merged[res_col])
    # Velocities: keep reference velocities (position-only correction).
    for col in ("vx", "vy", "vz"):
        out[col] = merged[col]
    return out


def _run_posebuilder(
    corrected_path: Path,
    out_dir: Path,
    frames: int,
    fps: int,
    pointing: str,
    scale_k: float,
    seed: int,
    resolution: str,
    t0: float | None,
    t1: float | None,
    mode: str,
    model: str,
    control_video: str | None,
    first_frame: str | None,
    prompt_json: str | None,
) -> None:
    cmd = [
        "cargo",
        "run",
        "-p",
        "posebuilder",
        "--example",
        "pose_pipeline",
        "--",
        str(corrected_path),
        str(out_dir),
        "--frames",
        str(frames),
        "--fps",
        str(fps),
        "--pointing",
        pointing,
        "--scale-k",
        str(scale_k),
        "--seed",
        str(seed),
        "--resolution",
        resolution,
        "--mode",
        mode,
        "--model",
        model,
    ]
    if t0 is not None:
        cmd += ["--t0", str(t0)]
    if t1 is not None:
        cmd += ["--t1", str(t1)]
    if control_video:
        cmd += ["--control-video", control_video]
    if first_frame:
        cmd += ["--first-frame", first_frame]
    if prompt_json:
        cmd += ["--prompt-json", prompt_json]

    print("Running posebuilder:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _submit_to_server(
    out_dir: Path,
    serve_url: str,
    first_frame: str | None,
    output_video: str | None,
    frames: int,
    fps: int,
) -> None:
    import urllib.request

    job = json.loads((out_dir / "job.json").read_text())
    actions = json.loads((out_dir / "actions.json").read_text())

    payload = {
        "model": job.get("model", "Cosmos3-Nano"),
        "prompt": job.get("prompt", "orbital trajectory"),
        "extra_params": {
            "action_mode": "forward_dynamics",
            "domain_name": "camera_pose",
            "raw_action_dim": 9,
            "action_chunk_size": len(actions.get("actions", [])),
        },
        "actions": actions.get("actions", []),
    }
    if first_frame:
        payload["first_frame"] = first_frame

    data = json.dumps(payload).encode("utf-8")
    url = serve_url.rstrip("/") + "/v1/videos/sync"
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    api_key = os.environ.get("COSMOS_API_KEY")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    print(f"Submitting to {url} ...")
    with urllib.request.urlopen(req) as resp:
        body = resp.read()
    out_path = Path(output_video) if output_video else (out_dir / "cosmos_pinn.mp4")
    out_path.write_bytes(body)
    print(f"Wrote video to {out_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Reconstruct a PINN-corrected trajectory and build a Cosmos job.")
    p.add_argument("--reference", required=True, help="reference CSV (t,x,y,z,vx,vy,vz)")
    p.add_argument("--predictions", required=True, help="predictions CSV from ml.pinn.run")
    p.add_argument("--out-dir", required=True, help="output directory for corrected trajectory + job files")
    p.add_argument("--frames", type=int, default=121, help="number of frames")
    p.add_argument("--fps", type=int, default=24, help="frames per second")
    p.add_argument("--pointing", default="earth", help="pointing law")
    p.add_argument("--scale-k", type=float, default=1000.0, help="km per Cosmos-metre")
    p.add_argument("--seed", type=int, default=0, help="seed")
    p.add_argument("--resolution", default="704x1280", help="WxH")
    p.add_argument("--mode", default="forward_dynamics", help="Cosmos mode")
    p.add_argument("--model", default="Cosmos3-Nano", help="Cosmos model")
    p.add_argument("--t0", type=float, default=None, help="window start (s)")
    p.add_argument("--t1", type=float, default=None, help="window end (s)")
    p.add_argument("--control-video", default=None, help="control video path")
    p.add_argument("--first-frame", default=None, help="first frame image path")
    p.add_argument("--prompt-json", default=None, help="prompt json path")
    p.add_argument("--serve-url", default=None, help="vLLM-Omni base URL")
    p.add_argument("--output-video", default=None, help="output video path when serving")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ref_path = _resolve(args.reference)
    pred_path = _resolve(args.predictions)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = _load_and_align(ref_path, pred_path)
    corrected = _reconstruct(merged)
    corrected_path = out_dir / "corrected_trajectory.csv"
    corrected.to_csv(corrected_path, index=False, float_format="%.12g")
    print(f"Wrote corrected trajectory: {corrected_path} ({len(corrected)} rows)")

    _run_posebuilder(
        corrected_path,
        out_dir,
        args.frames,
        args.fps,
        args.pointing,
        args.scale_k,
        args.seed,
        args.resolution,
        args.t0,
        args.t1,
        args.mode,
        args.model,
        args.control_video,
        args.first_frame,
        args.prompt_json,
    )

    if args.serve_url:
        _submit_to_server(out_dir, args.serve_url, args.first_frame, args.output_video, args.frames, args.fps)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
