#!/usr/bin/env python3
"""Cosmos Job A -- transfer: control video + caption -> photoreal video.

Checkpoint: ``Cosmos3-Nano-Sim-Transfer`` (ARCHITECTURE.md section 7A). One
precomputed control video (depth by default) plus a caption are fed to the
vLLM-Omni generator, which emits the clip with a fixed four-step schedule.
A transfer job has no pose sequence, so ``poses`` is ``[]``
(``docs/cosmos-contract.md`` section 4.1).

Because ``job.json`` is immutable (contract section 1) this worker reads the
control video path and caption from the job and writes only ``output.mp4``
(when the model is reachable), ``sample_outputs.json`` and ``manifest.json``.

Usage::

    python -m ml.cosmos.transfer_worker jobs/flyby_w03
    python ml/cosmos/transfer_worker.py jobs/flyby_w03 --out /tmp/out
    python -m ml.cosmos.transfer_worker jobs/flyby_w03 --backend fallback
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

WORKER = "transfer"
MODE = common.MODE_TRANSFER
VIDEO_NAME = "output.mp4"


def _caption(job: dict[str, Any], params: dict[str, Any], job_dir: Path) -> str | None:
    """Inline ``prompt``/``caption`` wins over a ``prompt_json`` file."""

    text = common.pick(job, params, "prompt", "caption")
    if text:
        return str(text)
    ref = common.pick(
        job, params, "prompt_json", default=common.inputs_map(job).get("prompt_json")
    )
    path = common.resolve_path(job_dir, ref)
    if path is not None and path.is_file():
        data = common.load_json(path)
        if isinstance(data, dict):
            for key in ("prompt", "caption", "text"):
                if data.get(key):
                    return str(data[key])
        elif isinstance(data, str):
            return data
    return None


def run(
    job_dir: Path, out_dir: Path | None = None, backend: str = "auto"
) -> dict[str, Any]:
    job_dir, job = common.load_job(Path(job_dir))
    params = common.params(job)
    job_id = common.safe_job_id(job)

    mode = common.normalize_mode(common.pick(job, params, "mode", default=MODE))
    if mode != MODE:
        raise common.JobError(f"transfer worker cannot run mode {mode!r}")

    model = common.pick(job, params, "model", default=common.CHECKPOINTS[MODE])
    seed = int(common.pick(job, params, "seed", default=0))
    fps = common.pick(job, params, "fps", default=24)
    resolution = common.pick(job, params, "resolution", default=None)
    scale_k = common.pick(
        job, params, "scale_k_km_per_m", "scale_k", "k", default=common.DEFAULT_SCALE_K
    )
    pointing_law = common.pick(job, params, "pointing_law", default="earth")
    window = common.pick(job, params, "window", default=None)
    control_type = common.pick(job, params, "control_type", default="depth")

    frames = common.pick(job, params, "frames", default=None)
    if frames is not None:
        frame_count = int(frames)
    else:
        try:
            frame_count = int(float(fps))
        except (TypeError, ValueError):
            frame_count = 1
    if frame_count < 1:
        raise common.JobError("frames must be at least 1")

    inputs = common.inputs_map(job)
    control_video = common.resolve_path(
        job_dir,
        common.pick(
            job,
            params,
            "control_video",
            default=inputs.get("control_video") or common.first_media(job),
        ),
    )
    caption = _caption(job, params, job_dir)

    out_dir = Path(out_dir) if out_dir is not None else job_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    common.log(
        WORKER,
        "loaded job",
        job=job_id,
        model=model,
        frames=frame_count,
        control=control_video,
    )

    client = OmniClient(model=model)
    use_omni = backend == "omni" or (backend == "auto" and client.available())

    video_path: Path | None = None
    if use_omni:
        if control_video is None:
            raise common.JobError(
                "transfer requires inputs.control_video to run vLLM-Omni"
            )
        if not control_video.is_file():
            raise common.JobError(f"control video not found: {control_video}")
        fields = {
            "model": model,
            "prompt": caption or "",
            "num_frames": str(frame_count),
            "fps": common.format_number(fps),
            "seed": str(seed),
            "control_type": control_type,
            "control_video": str(control_video),
            "guidance": "1.0",
            "control_guidance": "1.0",
            "prompt_upsampling": "false",
            "size": common.format_size(resolution),
        }
        common.log(
            WORKER, "generating video via vLLM-Omni", frames=frame_count, fps=fps
        )
        video_path = out_dir / VIDEO_NAME
        client.generate_video_to(video_path, fields)
        backend_label = "vllm-omni"
        mock = False
        note = "transfer mode emits no poses"
    else:
        if backend == "omni":
            raise OmniUnavailable("vLLM-Omni backend requested but not reachable")
        common.log(WORKER, "vLLM-Omni unavailable; writing contract outputs only")
        backend_label = "synthetic"
        mock = True
        note = (
            "transfer mode emits no poses; vLLM-Omni unavailable so no video was "
            "generated (synthetic fallback)"
        )

    manifest = common.write_outputs(
        out_dir,
        worker=WORKER,
        job_id=job_id,
        mode=mode,
        model=model,
        poses=[],
        frames=frame_count,
        fps=fps,
        scale_k=float(scale_k),
        pointing_law=str(pointing_law),
        first_pose=job.get("first_pose"),
        note=note,
        backend=backend_label,
        mock=mock,
        seed=seed,
        action_source="none",
        window=window,
        video_path=video_path,
        extra_metadata={
            "checkpoint": common.CHECKPOINTS[MODE],
            "control_type": control_type,
            "control_video": control_video.name if control_video else None,
            "caption": caption,
            "resolution": resolution,
        },
    )
    common.log(
        WORKER,
        "complete",
        job=job_id,
        poses=0,
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
        f"wrote transfer outputs for {manifest['job_id']} "
        f"({len(manifest['artifacts'])} artifacts)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
