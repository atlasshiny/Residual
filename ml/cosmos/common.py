"""Shared helpers for the real Cosmos 3 workers.

Implements the file contract in ``docs/cosmos-contract.md``: read ``job.json``
from a job directory, write ``sample_outputs.json`` and (last) ``manifest.json``.
Every worker emits the same JSON shapes; only the model call differs.

Standard library only, so importing a worker never needs PyTorch, vLLM, or
``requests`` on a login node.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ACTION_DIM = 9
ACTION_ENCODING = "translation_xyz + rot6d_first_two_columns"
TRANSLATION_UNIT = "cosmos_metre"
SAMPLE_OUTPUTS_SCHEMA = "cosmos-sample-outputs/v1"

MODE_TRANSFER = "transfer"
MODE_INVERSE = "inverse_dynamics"
MODE_FORWARD = "forward_dynamics"
VALID_MODES = (MODE_TRANSFER, MODE_INVERSE, MODE_FORWARD)

# ARCHITECTURE.md section 7 checkpoint routing.
CHECKPOINTS = {
    MODE_TRANSFER: "Cosmos3-Nano-Sim-Transfer",
    MODE_INVERSE: "Cosmos3-Nano",
    MODE_FORWARD: "Cosmos3-Nano",
}
DEFAULT_MODEL = "Cosmos3-Nano"
DEFAULT_DOMAIN_NAME = "camera_pose"
DEFAULT_SCALE_K = 1000.0

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".ppm")
# Longest first so ``action_path.json`` is recognised before ``actions.json``.
ACTION_EXTENSIONS = ("action_path.json", "actions.json", "action.json")


class JobError(Exception):
    """The job is missing, malformed, or inconsistent with this worker."""


# ---------------------------------------------------------------------------
# Discovery / logging
# ---------------------------------------------------------------------------


def log(worker: str, message: str, **fields: Any) -> None:
    """Write a progress line to stderr (contract: progress goes to stderr)."""

    detail = " ".join(
        f"{key}={_format(value)}" for key, value in fields.items() if value is not None
    )
    line = f"[cosmos:{worker}] {message}"
    if detail:
        line = f"{line} {detail}"
    print(line, file=sys.stderr, flush=True)


def _format(value: Any) -> str:
    if isinstance(value, Path):
        return value.name
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def build_parser(worker: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "job_dir",
        type=Path,
        help="directory containing job.json (outputs written here unless --out)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write sample_outputs.json and manifest.json here instead",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "omni", "fallback"),
        default="auto",
        help=(
            "auto: use vLLM-Omni when reachable, else fall back; "
            "omni: require vLLM-Omni; fallback: force the synthetic path"
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Job loading
# ---------------------------------------------------------------------------


def load_job(job_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Return ``(job_dir, job)``.

    Accepts a job directory (containing ``job.json``) or a path straight to a
    ``job.json``.  Relative input paths are resolved against the job directory.
    """

    job_dir = Path(job_dir)
    if job_dir.is_file():
        job_path = job_dir
        parent = job_path.parent
        job_dir = parent if str(parent) else Path(".")
    else:
        job_path = job_dir / "job.json"
    if not job_path.is_file():
        raise JobError(f"missing job file: {job_path}")
    job = load_json(job_path)
    if not isinstance(job, dict):
        raise JobError(f"{job_path} must contain a JSON object")
    return job_dir, job


def normalize_mode(value: Any) -> str:
    mode = str(value or "").strip().lower().replace("-", "_")
    if mode not in VALID_MODES:
        raise JobError(
            f"unknown mode {value!r}; expected one of {', '.join(VALID_MODES)}"
        )
    return mode


def safe_job_id(job: dict[str, Any]) -> str:
    job_id = job.get("job_id") or job.get("id")
    if not isinstance(job_id, str) or not job_id:
        raise JobError("job.json has no job_id/id")
    if job_id.startswith(".") or any(
        not (c.isalnum() or c in "._-") for c in job_id
    ):
        raise JobError(
            f"job id {job_id!r} must be non-empty and contain only [A-Za-z0-9._-]"
        )
    return job_id


def params(job: dict[str, Any]) -> dict[str, Any]:
    value = job.get("params")
    return value if isinstance(value, dict) else {}


def pick(
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


def inputs_map(job: dict[str, Any]) -> dict[str, Any]:
    value = job.get("inputs")
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        action = next(
            (
                item
                for item in value
                if isinstance(item, str) and item.lower().endswith(ACTION_EXTENSIONS)
            ),
            None,
        )
        return {"files": value, **({"actions": action} if action else {})}
    return {}


def first_media(
    job: dict[str, Any], extensions: Sequence[str] = VIDEO_EXTENSIONS
) -> str | None:
    """First input path (list form) with one of ``extensions``."""

    value = job.get("inputs")
    if not isinstance(value, list):
        return None
    for item in value:
        if isinstance(item, str) and item.lower().endswith(tuple(extensions)):
            return item
    return None


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(job_dir: Path, value: Any) -> Path | None:
    """Resolve an input path relative to the job directory (contract section 1)."""

    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(job_dir) / path
    return path


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def coerce_actions(value: Any) -> list[list[float]]:
    if isinstance(value, dict):
        for key in ("actions", "poses", "rows"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list):
        raise JobError("actions must be a list of 9-element rows")
    rows: list[list[float]] = []
    for index, row in enumerate(value):
        if not isinstance(row, (list, tuple)) or len(row) != ACTION_DIM:
            raise JobError(
                f"action row {index} must have exactly {ACTION_DIM} numbers"
            )
        numbers = [float(item) for item in row]
        if not all(math.isfinite(item) for item in numbers):
            raise JobError(f"action row {index} contains a non-finite value")
        rows.append(numbers)
    return rows


def job_actions(
    job: dict[str, Any], params: dict[str, Any], job_dir: Path
) -> tuple[list[list[float]] | None, str, Path | None]:
    """Return ``(rows, source, path)`` from inline JSON or an input file.

    ``source`` is ``"job"`` when the job supplied actions and ``"synthetic"``
    when it did not.  ``path`` is the file the rows were read from, if any, so a
    forward-dynamics worker can hand it to the model unchanged.
    """

    inline = job.get("actions")
    if inline is None:
        inline = params.get("actions")
    if inline is not None:
        return coerce_actions(inline), "job", None

    inputs = inputs_map(job)
    for key in ("actions", "action_path", "action"):
        ref = inputs.get(key)
        path = resolve_path(job_dir, ref)
        if path is None and ref is not None:
            raise JobError(f"inputs.{key} must be a path string")
        if path is None:
            continue
        if not path.is_file():
            raise JobError(f"inputs.{key} {ref!r} not found")
        return coerce_actions(load_json(path)), "job", path
    return None, "synthetic", None


def identity_pose() -> dict[str, Any]:
    return {
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "translation_km": [0.0, 0.0, 0.0],
    }


def synthetic_actions(count: int, seed: int) -> list[list[float]]:
    """Deterministic, model-shaped 9D rows used by the offline fallback.

    Kept identical to ``scripts/mock_cosmos_worker.py`` so the fallback is a
    drop-in for the mock the pipeline was developed against.
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


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_json(path: Path, value: Any) -> int:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def format_size(resolution: Any) -> str | None:
    if resolution is None:
        return None
    if isinstance(resolution, str):
        return resolution
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        return f"{int(resolution[0])}x{int(resolution[1])}"
    return None


def format_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def write_outputs(
    out_dir: Path,
    *,
    worker: str,
    job_id: str,
    mode: str,
    model: str,
    poses: Sequence[Sequence[float]],
    frames: int,
    fps: Any,
    scale_k: float,
    pointing_law: str,
    first_pose: dict[str, Any] | None,
    note: str,
    backend: str,
    mock: bool,
    seed: int = 0,
    action_source: str = "synthetic",
    window: Any = None,
    video_path: Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``sample_outputs.json`` then ``manifest.json`` (the completion marker).

    Raises before writing the manifest if anything fails, so a failed run never
    leaves a success manifest behind (contract section 4.2).
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_rows = [list(row) for row in poses]

    sample_outputs = {
        "schema": SAMPLE_OUTPUTS_SCHEMA,
        "job_id": job_id,
        "mode": mode,
        "model": model,
        "action_dim": ACTION_DIM,
        "action_encoding": ACTION_ENCODING,
        "translation_unit": TRANSLATION_UNIT,
        "scale_k_km_per_m": scale_k,
        "pointing_law": pointing_law,
        "fps": fps,
        "frames": frames,
        "first_pose": first_pose or identity_pose(),
        "poses": pose_rows,
        "mock": mock,
        "note": note,
    }
    output_path = out_dir / "sample_outputs.json"
    size = write_json(output_path, sample_outputs)

    artifacts = [
        {
            "name": "poses",
            "path": output_path.name,
            "kind": "pose_json",
            "bytes": size,
        }
    ]
    if video_path is not None:
        video_path = Path(video_path)
        try:
            video_rel = video_path.resolve().relative_to(out_dir.resolve()).as_posix()
        except ValueError:
            video_rel = video_path.name
        artifacts.append(
            {
                "name": "video",
                "path": video_rel,
                "kind": "video",
                "bytes": video_path.stat().st_size,
            }
        )

    metadata: dict[str, Any] = {
        "worker": worker,
        "backend": backend,
        "mock": mock,
        "seed": seed,
        "model": model,
        "fps": fps,
        "frames": frames,
        "pose_count": len(pose_rows),
        "action_source": action_source,
        "scale_k_km_per_m": scale_k,
        "pointing_law": pointing_law,
        "window": window,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    manifest = {
        "job_id": job_id,
        "mode": mode,
        "artifacts": artifacts,
        "created_at_s": int(time.time()),
        "metadata": metadata,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest
