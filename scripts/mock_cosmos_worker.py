#!/usr/bin/env python3
"""Mock Cosmos worker for the file-based job contract.

Reads an ipc job directory (``<job-dir>/job.json``) and writes:

* ``sample_outputs.json`` -- the ``poses`` array (9D action rows) that a real
  inverse-dynamics worker would produce, and
* ``manifest.json`` -- the ipc artifact manifest.

No GPU, no network, no third-party packages. This exists so the Rust pipeline,
evaluator, and UI can be developed before the Cosmos checkpoint is available.

Usage::

    python scripts/mock_cosmos_worker.py jobs/flyby_w03
    python scripts/mock_cosmos_worker.py jobs/flyby_w03 --out /tmp/mock-out

Both job shapes are accepted:

* the native :crate:`ipc` job: ``{id, mode, inputs: [...], seed, params}``
* the pose manifest written by ``posebuilder/examples/pose_pipeline.rs``:
  ``{job_id, mode, inputs: {...}, actions: [...], ...}``

See ``docs/cosmos-contract.md`` for the canonical file layout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ACTION_DIM = 9
ACTION_ENCODING = "translation_xyz + rot6d_first_two_columns"
TRANSFER_MODE = "transfer"


def _normalize_mode(value: Any) -> str:
    mode = str(value or "").strip().lower().replace("-", "_")
    if mode not in {"transfer", "inverse_dynamics", "forward_dynamics"}:
        raise ValueError(
            f"unknown mode {value!r}; expected transfer, inverse_dynamics, "
            "or forward_dynamics"
        )
    return mode


def _safe_job_id(job: dict[str, Any]) -> str:
    job_id = job.get("job_id") or job.get("id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("job.json has no job_id/id")
    if job_id.startswith(".") or any(
        not (c.isalnum() or c in "._-") for c in job_id
    ):
        raise ValueError(
            f"job id {job_id!r} must be non-empty and contain only [A-Za-z0-9._-]"
        )
    return job_id


def _params(job: dict[str, Any]) -> dict[str, Any]:
    params = job.get("params")
    return params if isinstance(params, dict) else {}


def _pick(
    job: dict[str, Any],
    params: dict[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
    """Top-level job fields win over ``params``; skip explicit nulls."""
    for name in names:
        if job.get(name) is not None:
            return job[name]
    for name in names:
        if params.get(name) is not None:
            return params[name]
    return default


def _inputs(job: dict[str, Any]) -> dict[str, Any]:
    value = job.get("inputs")
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        # Native ipc jobs list paths; recognise an action file if present.
        actions = next(
            (
                item
                for item in value
                if isinstance(item, str)
                and item.lower().endswith(("actions.json", "action.json"))
            ),
            None,
        )
        return {"files": value, **({"actions": actions} if actions else {})}
    return {}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_actions(value: Any) -> list[list[float]]:
    if isinstance(value, dict):
        value = value.get("actions", value.get("rows"))
    if not isinstance(value, list):
        raise TypeError("actions must be a list of 9-element rows")
    rows: list[list[float]] = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != ACTION_DIM:
            raise ValueError(
                f"action row {index} must have exactly {ACTION_DIM} numbers"
            )
        numbers = [float(item) for item in row]
        if not all(math.isfinite(item) for item in numbers):
            raise ValueError(f"action row {index} contains a non-finite value")
        rows.append(numbers)
    return rows


def _job_actions(
    job: dict[str, Any], params: dict[str, Any], job_dir: Path
) -> tuple[list[list[float]] | None, str]:
    """Return ``(actions, source)`` from inline JSON or the inputs path."""
    inline = job.get("actions")
    if inline is None:
        inline = params.get("actions")
    if isinstance(inline, list):
        return _coerce_actions(inline), "job"

    actions_path = _inputs(job).get("actions")
    if isinstance(actions_path, str) and actions_path:
        path = Path(actions_path)
        if not path.is_absolute():
            path = job_dir / path
        if path.is_file():
            return _coerce_actions(_load_json(path)), "job"
        print(
            f"warning: inputs.actions {actions_path!r} not found; "
            "using synthetic actions",
            file=sys.stderr,
        )
    return None, "synthetic"


def _synthetic_actions(count: int, seed: int) -> list[list[float]]:
    """Deterministic, model-shaped 9D rows.

    Translation stays small (a few centimetres in Cosmos units) and the 6D
    rotation describes a gentle rotation about the optical axis, which keeps
    the mock inside the range real actions are normalized to.
    """
    count = max(count, 1)
    phase = (seed % 360) * math.pi / 180.0
    rows: list[list[float]] = []
    for index in range(count):
        angle = phase + 2.0 * math.pi * index / count
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rows.append(
            [
                0.02 * cos_a,
                0.02 * sin_a,
                0.0,
                cos_a,
                sin_a,
                0.0,
                -sin_a,
                cos_a,
                0.0,
            ]
        )
    return rows


def _identity_pose() -> dict[str, Any]:
    return {
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "translation_km": [0.0, 0.0, 0.0],
    }


def _write_json(path: Path, value: Any) -> int:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def run(job_dir: Path, out_dir: Path | None = None) -> dict[str, Any]:
    if job_dir.is_file():
        # Also accept a combined manifest path (pose_pipeline <...> action.json).
        job_path = job_dir
        job_dir = job_path.parent if str(job_path.parent) else Path(".")
    else:
        job_path = job_dir / "job.json"
    if not job_path.is_file():
        raise FileNotFoundError(f"missing job file: {job_path}")

    job = _load_json(job_path)
    if not isinstance(job, dict):
        raise TypeError(f"{job_path} must contain a JSON object")

    params = _params(job)
    job_id = _safe_job_id(job)
    mode = _normalize_mode(_pick(job, params, "mode", default=""))
    model = _pick(job, params, "model", default="Cosmos3-Nano")
    seed = int(_pick(job, params, "seed", default=0))
    fps = _pick(job, params, "fps", default=24)
    scale_k = _pick(
        job, params, "scale_k_km_per_m", "scale_k", "k", default=1000.0
    )
    pointing_law = _pick(job, params, "pointing_law", default="earth")
    window = _pick(job, params, "window", default=None)
    frames = _pick(job, params, "frames", default=None)

    frame_count = int(frames) if frames is not None else None
    if frame_count is not None and frame_count < 2:
        raise ValueError("frames must be at least 2")

    actions, source = _job_actions(job, params, job_dir)
    if mode == TRANSFER_MODE:
        # Job A produces a video, not a pose sequence.
        actions = []
        source = "none"
        if frame_count is None:
            frame_count = int(fps) if str(fps).isdigit() else 1
    else:
        if actions is None:
            count = (frame_count - 1) if frame_count else 24
            actions = _synthetic_actions(count, seed)
        if frame_count is None:
            frame_count = len(actions) + 1

    if frame_count is None:
        frame_count = 1

    out_dir = out_dir or job_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_outputs = {
        "schema": "cosmos-sample-outputs/v1",
        "job_id": job_id,
        "mode": mode,
        "model": model,
        "action_dim": ACTION_DIM,
        "action_encoding": ACTION_ENCODING,
        "translation_unit": "cosmos_metre",
        "scale_k_km_per_m": scale_k,
        "pointing_law": pointing_law,
        "fps": fps,
        "frames": frame_count,
        "first_pose": job.get("first_pose") or _identity_pose(),
        "poses": actions,
        "mock": True,
        "note": (
            "transfer mode emits no poses"
            if mode == TRANSFER_MODE
            else "mock inverse-dynamics output; each row is one 9D relative action"
        ),
    }
    output_path = out_dir / "sample_outputs.json"
    size = _write_json(output_path, sample_outputs)

    manifest = {
        "job_id": job_id,
        "mode": mode,
        "artifacts": [
            {
                "name": "poses",
                "path": "sample_outputs.json",
                "kind": "pose_json",
                "bytes": size,
            }
        ],
        "created_at_s": int(time.time()),
        "metadata": {
            "mock": True,
            "worker": "scripts/mock_cosmos_worker.py",
            "seed": seed,
            "model": model,
            "fps": fps,
            "frames": frame_count,
            "pose_count": len(actions),
            "action_source": source,
            "scale_k_km_per_m": scale_k,
            "pointing_law": pointing_law,
            "window": window,
        },
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mock Cosmos worker over an ipc job directory."
    )
    parser.add_argument(
        "job_dir",
        type=Path,
        help="directory containing job.json (outputs are written here by default)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write sample_outputs.json and manifest.json here instead",
    )
    args = parser.parse_args(argv)

    try:
        manifest = run(args.job_dir, args.out)
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"wrote {manifest['metadata']['pose_count']} poses for "
        f"{manifest['job_id']} ({manifest['mode']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
