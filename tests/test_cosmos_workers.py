"""Contract tests for the Cosmos 3 workers in ``ml/cosmos``.

Every worker is a CLI tool that consumes a job directory and writes
``sample_outputs.json`` plus ``manifest.json`` (``docs/cosmos-contract.md``). The
integration tests below invoke them through :mod:`subprocess` exactly as the
Rust scheduler does -- the mock worker's ``job.json`` shapes are used as
fixtures -- while the shared helpers in :mod:`ml.cosmos.common` are unit-tested
directly.

The workers default to synthetic (fallback) outputs when no vLLM-Omni server is
configured, so these tests need no GPU, no network, and no third-party packages.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from ml.cosmos import common

REPO_ROOT = Path(__file__).resolve().parents[1]
COSMOS_DIR = REPO_ROOT / "ml" / "cosmos"

WORKERS = {
    "transfer": COSMOS_DIR / "transfer_worker.py",
    "inverse": COSMOS_DIR / "inverse_worker.py",
    "forward": COSMOS_DIR / "forward_worker.py",
}

# A valid 9D action row (Zhou et al. 2019 6D rotation, identity).
IDENTITY_ROW = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def run_worker(
    worker: str,
    job_dir: Path,
    out_dir: Path | None = None,
    backend: str = "fallback",
) -> subprocess.CompletedProcess:
    """Run a worker as a CLI tool. ``fallback`` keeps it offline/deterministic."""

    cmd = [sys.executable, str(WORKERS[worker]), str(job_dir)]
    if out_dir is not None:
        cmd += ["--out", str(out_dir)]
    cmd += ["--backend", backend]
    return subprocess.run(
        cmd, check=False, capture_output=True, text=True, cwd=REPO_ROOT
    )


def write_job(job_dir: Path, job: dict) -> Path:
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    return job_dir


def write_raw_job(job_dir: Path, text: str) -> Path:
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(text, encoding="utf-8")
    return job_dir


def write_actions_file(job_dir: Path, actions: list) -> Path:
    """An ``actions.json`` in the shape the contract (section 3.5) defines."""

    path = Path(job_dir) / "actions.json"
    path.write_text(
        json.dumps(
            {
                "schema": "cosmos-actions/v1",
                "job_id": "fixture",
                "action_dim": 9,
                "action_encoding": "translation_xyz + rot6d_first_two_columns",
                "translation_unit": "cosmos_metre",
                "scale_k_km_per_m": 1000.0,
                "count": len(actions),
                "actions": actions,
            }
        ),
        encoding="utf-8",
    )
    return path


def load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def touch(path: Path) -> Path:
    Path(path).write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return Path(path)


# ---------------------------------------------------------------------------
# Transfer worker (Job A)
# ---------------------------------------------------------------------------


def _transfer_job(frames: int = 12) -> dict:
    return {
        "schema": "cosmos-job/v1",
        "job_id": "flyby_tx",
        "id": "flyby_tx",
        "mode": "transfer",
        "model": "Cosmos3-Nano-Sim-Transfer",
        "fps": 24,
        "frames": frames,
        "seed": 3,
        "prompt": "a low flyby over the lunar terminator",
        "inputs": {"control_video": "control.mp4"},
    }


def test_transfer_writes_outputs_and_empty_poses(tmp_path):
    job_dir = write_job(tmp_path / "job", _transfer_job())
    touch(job_dir / "control.mp4")
    out = tmp_path / "out"

    result = run_worker("transfer", job_dir, out)
    assert result.returncode == 0, result.stderr

    assert (out / "sample_outputs.json").is_file()
    assert (out / "manifest.json").is_file()

    outputs = load(out / "sample_outputs.json")
    manifest = load(out / "manifest.json")

    # Transfer jobs emit a video, not a pose sequence (contract section 4.1).
    assert outputs["mode"] == "transfer"
    assert outputs["poses"] == []
    assert outputs["action_dim"] == 9
    assert outputs["frames"] == 12
    assert outputs["job_id"] == "flyby_tx"

    assert manifest["job_id"] == "flyby_tx"
    assert manifest["mode"] == "transfer"
    assert manifest["artifacts"][0]["name"] == "poses"
    assert manifest["artifacts"][0]["path"] == "sample_outputs.json"
    assert manifest["artifacts"][0]["kind"] == "pose_json"
    assert manifest["metadata"]["pose_count"] == 0


def test_transfer_missing_control_video_exits_2_without_manifest(tmp_path):
    job = _transfer_job()
    job["inputs"] = {}  # no control video anywhere in the job
    job_dir = write_job(tmp_path / "job", job)
    out = tmp_path / "out"

    # The control video is only validated on the vLLM-Omni path; force it so
    # the missing input surfaces instead of the synthetic fallback.
    result = run_worker("transfer", job_dir, out, backend="omni")

    assert result.returncode == 2
    assert not (out / "manifest.json").exists()
    assert not (out / "sample_outputs.json").exists()
    assert "control_video" in result.stderr


def test_transfer_wrong_mode_exits_2_without_manifest(tmp_path):
    job = _transfer_job()
    job["mode"] = "inverse_dynamics"
    job_dir = write_job(tmp_path / "job", job)
    out = tmp_path / "out"

    result = run_worker("transfer", job_dir, out)

    assert result.returncode == 2
    assert not (out / "manifest.json").exists()
    assert "cannot run mode" in result.stderr


# ---------------------------------------------------------------------------
# Inverse worker (Job B)
# ---------------------------------------------------------------------------


def _inverse_job(frames: int = 5) -> dict:
    return {
        "schema": "cosmos-job/v1",
        "job_id": "flyby_inv",
        "id": "flyby_inv",
        "mode": "inverse_dynamics",
        "model": "Cosmos3-Nano",
        "fps": 24,
        "frames": frames,
        "seed": 7,
        "domain_name": "camera_pose",
        "inputs": {"video": "clip.mp4"},
    }


def test_inverse_recovers_nine_dimensional_poses(tmp_path):
    job_dir = write_job(tmp_path / "job", _inverse_job())
    touch(job_dir / "clip.mp4")
    out = tmp_path / "out"

    result = run_worker("inverse", job_dir, out)
    assert result.returncode == 0, result.stderr

    outputs = load(out / "sample_outputs.json")
    manifest = load(out / "manifest.json")

    assert outputs["mode"] == "inverse_dynamics"
    assert outputs["action_dim"] == 9
    assert outputs["action_encoding"] == "translation_xyz + rot6d_first_two_columns"
    assert len(outputs["poses"]) == 4

    # Every recovered row is exactly 9 finite numbers (contract section 7.6).
    for row in outputs["poses"]:
        assert isinstance(row, list)
        assert len(row) == 9
        assert all(math.isfinite(value) for value in row)

    assert manifest["metadata"]["pose_count"] == 4
    assert manifest["artifacts"][0]["path"] == "sample_outputs.json"


@pytest.mark.parametrize("frames", [2, 5, 24, 120])
def test_inverse_pose_count_is_frames_minus_one(tmp_path, frames):
    job_dir = write_job(tmp_path / "job", _inverse_job(frames=frames))
    out = tmp_path / "out"

    result = run_worker("inverse", job_dir, out)
    assert result.returncode == 0, result.stderr

    outputs = load(out / "sample_outputs.json")
    manifest = load(out / "manifest.json")

    assert outputs["frames"] == frames
    assert len(outputs["poses"]) == frames - 1
    assert manifest["metadata"]["pose_count"] == frames - 1


def test_inverse_missing_clip_exits_2_without_manifest(tmp_path):
    job = _inverse_job()
    job["inputs"] = {}  # no clip/video input
    job_dir = write_job(tmp_path / "job", job)
    out = tmp_path / "out"

    # The clip is required on the vLLM-Omni path; the fallback would synthesize.
    result = run_worker("inverse", job_dir, out, backend="omni")

    assert result.returncode == 2
    assert not (out / "manifest.json").exists()
    assert not (out / "sample_outputs.json").exists()
    assert "video" in result.stderr


# ---------------------------------------------------------------------------
# Forward worker (Job C)
# ---------------------------------------------------------------------------


def _forward_job(actions: list | None = None) -> dict:
    job = {
        "schema": "cosmos-job/v1",
        "job_id": "flyby_fw",
        "id": "flyby_fw",
        "mode": "forward_dynamics",
        "model": "Cosmos3-Nano",
        "fps": 24,
        "seed": 1,
        "domain_name": "camera_pose",
        "inputs": {"first_frame": "frame.png"},
    }
    if actions is not None:
        job["actions"] = actions
    return job


def test_forward_echoes_inline_job_actions(tmp_path):
    actions = [
        IDENTITY_ROW,
        [0.02, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.02, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ]
    job_dir = write_job(tmp_path / "job", _forward_job(actions=actions))
    touch(job_dir / "frame.png")
    out = tmp_path / "out"

    result = run_worker("forward", job_dir, out)
    assert result.returncode == 0, result.stderr

    outputs = load(out / "sample_outputs.json")
    manifest = load(out / "manifest.json")

    assert outputs["mode"] == "forward_dynamics"
    # The commanded sequence is echoed for the evaluator to compare.
    assert outputs["poses"] == actions
    assert outputs["frames"] == len(actions) + 1
    assert manifest["metadata"]["pose_count"] == len(actions)
    assert manifest["metadata"]["action_source"] == "job"


def test_forward_reads_actions_file(tmp_path):
    actions = [
        IDENTITY_ROW,
        [0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ]
    job = _forward_job()
    job["inputs"]["actions"] = "actions.json"
    job_dir = write_job(tmp_path / "job", job)
    touch(job_dir / "frame.png")
    write_actions_file(job_dir, actions)
    out = tmp_path / "out"

    result = run_worker("forward", job_dir, out)
    assert result.returncode == 0, result.stderr

    outputs = load(out / "sample_outputs.json")
    assert outputs["poses"] == actions
    assert outputs["frames"] == len(actions) + 1


def test_forward_missing_action_file_exits_2_without_manifest(tmp_path):
    job = _forward_job()
    job["inputs"]["actions"] = "does_not_exist.json"
    job_dir = write_job(tmp_path / "job", job)
    out = tmp_path / "out"

    result = run_worker("forward", job_dir, out)

    assert result.returncode == 2
    assert not (out / "manifest.json").exists()
    assert not (out / "sample_outputs.json").exists()
    assert "not found" in result.stderr


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_forward_non_finite_actions_exit_2_without_manifest(tmp_path, bad):
    row = list(IDENTITY_ROW)
    row[3] = bad
    job = _forward_job(actions=[IDENTITY_ROW, row])
    job_dir = write_job(tmp_path / "job", job)
    out = tmp_path / "out"

    result = run_worker("forward", job_dir, out)

    assert result.returncode == 2
    assert not (out / "manifest.json").exists()
    assert "non-finite" in result.stderr


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def test_load_job_from_directory(tmp_path):
    job_dir = write_job(tmp_path / "job", {"job_id": "x", "mode": "transfer"})

    got_dir, job = common.load_job(job_dir)

    assert got_dir == job_dir
    assert job == {"job_id": "x", "mode": "transfer"}


def test_load_job_from_file_path(tmp_path):
    job_dir = write_job(tmp_path / "job", {"job_id": "x", "mode": "transfer"})

    got_dir, job = common.load_job(job_dir / "job.json")

    assert got_dir == job_dir
    assert job["job_id"] == "x"


def test_load_job_missing_directory_raises(tmp_path):
    with pytest.raises(common.JobError):
        common.load_job(tmp_path / "nope")


def test_load_job_directory_without_job_file_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(common.JobError):
        common.load_job(tmp_path / "empty")


def test_load_job_non_object_raises(tmp_path):
    job_dir = write_raw_job(tmp_path / "job", "[1, 2, 3]")
    with pytest.raises(common.JobError):
        common.load_job(job_dir)


def test_load_job_invalid_json_raises(tmp_path):
    job_dir = write_raw_job(tmp_path / "job", "{not json")
    with pytest.raises(json.JSONDecodeError):
        common.load_job(job_dir)


def test_coerce_actions_accepts_valid_rows():
    rows = [list(IDENTITY_ROW), [0.5] * 9]
    assert common.coerce_actions(rows) == rows


def test_coerce_actions_accepts_wrapped_object():
    wrapped = {"schema": "cosmos-actions/v1", "actions": [IDENTITY_ROW]}
    assert common.coerce_actions(wrapped) == [IDENTITY_ROW]


@pytest.mark.parametrize(
    "bad",
    [
        [[0.0] * 8],  # too short
        [[0.0] * 10],  # too long
        "not a list of rows",
        None,
        [IDENTITY_ROW, 0.0],  # a row that is not a sequence
    ],
)
def test_coerce_actions_rejects_bad_shape(bad):
    with pytest.raises(common.JobError):
        common.coerce_actions(bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_coerce_actions_rejects_non_finite(value):
    row = list(IDENTITY_ROW)
    row[3] = value
    with pytest.raises(common.JobError):
        common.coerce_actions([row])


# -- write_outputs ----------------------------------------------------------


def _write_outputs(out_dir: Path, **overrides) -> dict:
    kwargs = {
        "worker": "test",
        "job_id": "job",
        "mode": "inverse_dynamics",
        "model": "Cosmos3-Nano",
        "poses": [IDENTITY_ROW],
        "frames": 2,
        "fps": 24,
        "scale_k": 1000.0,
        "pointing_law": "earth",
        "first_pose": None,
        "note": "test run",
        "backend": "synthetic",
        "mock": True,
    }
    kwargs.update(overrides)
    return common.write_outputs(out_dir, **kwargs)


def test_write_outputs_creates_both_files(tmp_path):
    out = tmp_path / "nested" / "out"

    manifest = _write_outputs(out)

    assert (out / "sample_outputs.json").is_file()
    assert (out / "manifest.json").is_file()

    outputs = load(out / "sample_outputs.json")
    assert outputs["schema"] == "cosmos-sample-outputs/v1"
    assert outputs["job_id"] == "job"
    assert outputs["mode"] == "inverse_dynamics"
    assert outputs["poses"] == [IDENTITY_ROW]
    assert outputs["frames"] == 2
    assert outputs["first_pose"] == common.identity_pose()

    assert manifest["job_id"] == "job"
    assert manifest["artifacts"][0]["path"] == "sample_outputs.json"
    assert manifest["artifacts"][0]["kind"] == "pose_json"
    assert manifest["artifacts"][0]["bytes"] > 0
    assert manifest["metadata"]["pose_count"] == 1
    assert manifest["metadata"]["worker"] == "test"


def test_write_outputs_marks_video_artifact(tmp_path):
    out = tmp_path / "out"
    out.mkdir(parents=True)
    video = touch(out / "output.mp4")

    manifest = _write_outputs(out, video_path=video)

    kinds = {artifact["kind"] for artifact in manifest["artifacts"]}
    assert kinds == {"pose_json", "video"}
    video_artifact = next(a for a in manifest["artifacts"] if a["kind"] == "video")
    assert video_artifact["path"] == "output.mp4"
    assert video_artifact["bytes"] == video.stat().st_size


def test_write_outputs_does_not_write_manifest_on_failure(tmp_path):
    out = tmp_path / "out"
    bad_row = list(IDENTITY_ROW)
    bad_row[0] = float("nan")

    # allow_nan=False makes serialisation fail; the completion marker must not
    # be written (contract section 4.2: write manifest.json last).
    with pytest.raises(ValueError):
        _write_outputs(out, poses=[bad_row])

    assert not (out / "manifest.json").exists()
