#!/usr/bin/env python3
"""Cosmos Job B -- inverse dynamics: video clip -> 9D action sequence.

Checkpoint: ``Cosmos3-Nano`` (ARCHITECTURE.md section 7B). The model consumes a
video and returns one 9D relative action per frame transition; the worker writes
them to ``sample_outputs.json`` under ``poses`` for the evaluator to integrate
and Sim(3)-align.

The shipped inverse-dynamics examples cover the ``av`` and ``robot`` domains;
``camera_pose`` is attempted first and ``av`` is the documented fallback
(section 7B caveat 1).  Set ``domain_name`` in ``job.json`` to override.

Usage::

    python -m ml.cosmos.inverse_worker jobs/flyby_w03
    python ml/cosmos/inverse_worker.py jobs/flyby_w03 --out /tmp/out
    python -m ml.cosmos.inverse_worker jobs/flyby_w03 --backend fallback
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

WORKER = "inverse"
MODE = common.MODE_INVERSE
DEFAULT_FALLBACK_COUNT = 24


def _video_input(
    job: dict[str, Any], params: dict[str, Any], job_dir: Path
) -> Path | None:
    inputs = common.inputs_map(job)
    return common.resolve_path(
        job_dir,
        common.pick(
            job,
            params,
            "clip",
            "video",
            "control_video",
            default=inputs.get("video")
            or inputs.get("clip")
            or inputs.get("control_video")
            or common.first_media(job, common.VIDEO_EXTENSIONS),
        ),
    )


def run(
    job_dir: Path, out_dir: Path | None = None, backend: str = "auto"
) -> dict[str, Any]:
    job_dir, job = common.load_job(Path(job_dir))
    params = common.params(job)
    job_id = common.safe_job_id(job)

    mode = common.normalize_mode(common.pick(job, params, "mode", default=MODE))
    if mode != MODE:
        raise common.JobError(f"inverse worker cannot run mode {mode!r}")

    model = common.pick(job, params, "model", default=common.CHECKPOINTS[MODE])
    seed = int(common.pick(job, params, "seed", default=0))
    fps = common.pick(job, params, "fps", default=24)
    domain_name = common.pick(
        job, params, "domain_name", default=common.DEFAULT_DOMAIN_NAME
    )
    scale_k = common.pick(
        job, params, "scale_k_km_per_m", "scale_k", "k", default=common.DEFAULT_SCALE_K
    )
    pointing_law = common.pick(job, params, "pointing_law", default="earth")
    window = common.pick(job, params, "window", default=None)
    frames = common.pick(job, params, "frames", default=None)

    if frames is not None:
        frame_count = max(2, int(frames))
        fallback_count = frame_count - 1
    else:
        fallback_count = DEFAULT_FALLBACK_COUNT
        frame_count = fallback_count + 1

    video = _video_input(job, params, job_dir)
    out_dir = Path(out_dir) if out_dir is not None else job_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    common.log(WORKER, "loaded job", job=job_id, model=model, video=video)

    client = OmniClient(model=model)
    use_omni = backend == "omni" or (backend == "auto" and client.available())

    if use_omni:
        if video is None:
            raise common.JobError(
                "inverse_dynamics requires an input video to run vLLM-Omni"
            )
        if not video.is_file():
            raise common.JobError(f"input video not found: {video}")
        common.log(
            WORKER,
            "inferring actions via vLLM-Omni",
            domain=domain_name,
            fps=fps,
        )
        actions = client.infer_actions(
            video, model=model, domain_name=str(domain_name), fps=fps, frames=frame_count
        )
        if not actions:
            raise common.JobError("vLLM-Omni returned no action rows")
        backend_label = "vllm-omni"
        mock = False
        action_source = "vllm-omni"
        note = "real inverse-dynamics output; each row is one 9D relative action"
    else:
        if backend == "omni":
            raise OmniUnavailable("vLLM-Omni backend requested but not reachable")
        common.log(
            WORKER,
            "vLLM-Omni unavailable; synthesizing actions",
            count=fallback_count,
        )
        actions = common.synthetic_actions(fallback_count, seed)
        backend_label = "synthetic"
        mock = True
        action_source = "synthetic"
        note = (
            "synthetic inverse-dynamics fallback; each row is one 9D relative action"
        )

    frame_count = len(actions) + 1  # invariant: action_count == frames - 1

    manifest = common.write_outputs(
        out_dir,
        worker=WORKER,
        job_id=job_id,
        mode=mode,
        model=model,
        poses=actions,
        frames=frame_count,
        fps=fps,
        scale_k=float(scale_k),
        pointing_law=str(pointing_law),
        first_pose=job.get("first_pose"),
        note=note,
        backend=backend_label,
        mock=mock,
        seed=seed,
        action_source=action_source,
        window=window,
        extra_metadata={
            "checkpoint": common.CHECKPOINTS[MODE],
            "domain_name": domain_name,
            "video": video.name if video else None,
        },
    )
    common.log(
        WORKER,
        "complete",
        job=job_id,
        poses=len(actions),
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
