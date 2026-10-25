#!/usr/bin/env python3
"""Cosmos Job C -- forward dynamics: image/video + action path -> video.

Checkpoint: ``Cosmos3-Nano`` (ARCHITECTURE.md section 7C). One first frame (or a
video) plus the planned pose sequence are pushed through the vLLM-Omni
generator. The action modality is selected with ``domain_name: camera_pose``
(section 6.2 step 6); ``av`` can be requested in ``job.json`` for the shipped
examples.

The action rows come from ``actions.json`` (or an inline ``actions`` array). If
the job supplies none, the offline fallback synthesizes them, matching
``scripts/mock_cosmos_worker.py``. The written ``poses`` echo the commanded
sequence so the evaluator can compare it with job B's recovery.

Usage::

    python -m ml.cosmos.forward_worker jobs/flyby_w03
    python ml/cosmos/forward_worker.py jobs/flyby_w03 --out /tmp/out
    python -m ml.cosmos.forward_worker jobs/flyby_w03 --backend fallback
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common  # type: ignore
    from omni import OmniClient, OmniUnavailable  # type: ignore
else:
    from . import common
    from .omni import OmniClient, OmniUnavailable

WORKER = "forward"
MODE = common.MODE_FORWARD
VIDEO_NAME = "output.mp4"
DEFAULT_FALLBACK_COUNT = 24
ACTION_NAME = "forward_actions.json"


def _media_input(
    job: dict[str, Any], params: dict[str, Any], job_dir: Path
) -> Path | None:
    inputs = common.inputs_map(job)
    return common.resolve_path(
        job_dir,
        common.pick(
            job,
            params,
            "first_frame",
            "image",
            "video",
            default=inputs.get("first_frame")
            or inputs.get("image")
            or inputs.get("video")
            or inputs.get("control_video")
            or common.first_media(job, common.VIDEO_EXTENSIONS + common.IMAGE_EXTENSIONS),
        ),
    )


def _materialize_actions(
    out_dir: Path, job_id: str, actions: list[list[float]], scale_k: float
) -> Path:
    """Write an ``actions.json``-shaped file for the model's ``action_path``."""

    path = out_dir / ACTION_NAME
    common.write_json(
        path,
        {
            "schema": "cosmos-actions/v1",
            "job_id": job_id,
            "action_dim": common.ACTION_DIM,
            "action_encoding": common.ACTION_ENCODING,
            "translation_unit": common.TRANSLATION_UNIT,
            "scale_k_km_per_m": scale_k,
            "count": len(actions),
            "actions": [list(row) for row in actions],
        },
    )
    return path


def run(
    job_dir: Path, out_dir: Path | None = None, backend: str = "auto"
) -> dict[str, Any]:
    job_dir, job = common.load_job(Path(job_dir))
    params = common.params(job)
    job_id = common.safe_job_id(job)

    mode = common.normalize_mode(common.pick(job, params, "mode", default=MODE))
    if mode != MODE:
        raise common.JobError(f"forward worker cannot run mode {mode!r}")

    model = common.pick(job, params, "model", default=common.CHECKPOINTS[MODE])
    seed = int(common.pick(job, params, "seed", default=0))
    fps = common.pick(job, params, "fps", default=24)
    resolution = common.pick(job, params, "resolution", default=None)
    domain_name = common.pick(
        job, params, "domain_name", default=common.DEFAULT_DOMAIN_NAME
    )
    scale_k = float(
        common.pick(
            job,
            params,
            "scale_k_km_per_m",
            "scale_k",
            "k",
            default=common.DEFAULT_SCALE_K,
        )
    )
    pointing_law = common.pick(job, params, "pointing_law", default="earth")
    window = common.pick(job, params, "window", default=None)
    frames = common.pick(job, params, "frames", default=None)

    actions, source, action_path = common.job_actions(job, params, job_dir)
    if actions is None:
        count = (int(frames) - 1) if frames is not None else DEFAULT_FALLBACK_COUNT
        actions = common.synthetic_actions(max(count, 1), seed)
        source = "synthetic"
    frame_count = len(actions) + 1

    media = _media_input(job, params, job_dir)
    out_dir = Path(out_dir) if out_dir is not None else job_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    common.log(
        WORKER,
        "loaded job",
        job=job_id,
        model=model,
        actions=len(actions),
        source=source,
        media=media,
    )

    client = OmniClient(model=model)
    use_omni = backend == "omni" or (backend == "auto" and client.available())

    video_path: Path | None = None
    if use_omni:
        if media is None:
            raise common.JobError(
                "forward_dynamics requires inputs.first_frame or inputs.video"
            )
        if not media.is_file():
            raise common.JobError(f"input media not found: {media}")
        if source == "synthetic":
            raise common.JobError(
                "forward_dynamics requires an action sequence "
                "(actions.json or inline actions)"
            )
        if action_path is None:
            action_path = _materialize_actions(out_dir, job_id, actions, scale_k)
        fields = {
            "model": model,
            "action_path": str(action_path),
            "domain_name": str(domain_name),
            "num_frames": str(frame_count),
            "fps": common.format_number(fps),
            "seed": str(seed),
            "size": common.format_size(resolution),
        }
        if media.suffix.lower() in common.IMAGE_EXTENSIONS:
            fields["image"] = str(media)
        else:
            fields["video"] = str(media)
        common.log(
            WORKER,
            "generating video via vLLM-Omni",
            domain=domain_name,
            frames=frame_count,
        )
        video_path = out_dir / VIDEO_NAME
        client.generate_video_to(video_path, fields)
        backend_label = "vllm-omni"
        mock = False
        note = "commanded forward-dynamics actions; each row is one 9D relative action"
    else:
        if backend == "omni":
            raise OmniUnavailable("vLLM-Omni backend requested but not reachable")
        common.log(WORKER, "vLLM-Omni unavailable; writing contract outputs only")
        backend_label = "synthetic"
        mock = True
        note = (
            "synthetic forward-dynamics fallback; poses echo the commanded 9D "
            "actions and no video was generated"
        )

    manifest = common.write_outputs(
        out_dir,
        worker=WORKER,
        job_id=job_id,
        mode=mode,
        model=model,
        poses=actions,
        frames=frame_count,
        fps=fps,
        scale_k=scale_k,
        pointing_law=str(pointing_law),
        first_pose=job.get("first_pose"),
        note=note,
        backend=backend_label,
        mock=mock,
        seed=seed,
        action_source=source,
        window=window,
        video_path=video_path,
        extra_metadata={
            "checkpoint": common.CHECKPOINTS[MODE],
            "domain_name": domain_name,
            "media": media.name if media else None,
            "action_path": action_path.name if action_path else None,
            "resolution": resolution,
        },
    )
    common.log(
        WORKER,
        "complete",
        job=job_id,
        poses=len(actions),
        video=video_path is not None,
        backend=backend_label,
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = common.build_parser(WORKER, __doc__ or "")
    args = parser.parse_args(argv)
    try:
        manifest = run(args.job_dir, args.out, backend=args.backend)
    except (common.JobError, OmniUnavailable, OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {manifest['metadata']['pose_count']} poses for "
        f"{manifest['job_id']} ({manifest['mode']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
