"""Headless depth-control renderer for Cosmos transfer jobs.

Reads a trajectory CSV with the canonical shape ``t,x,y,z,...`` and writes one
grayscale PPM depth frame per output frame plus a ``manifest.json``.  The
projection is orthographic: two scene axes map to pixels and the third becomes
pixel intensity, so nearer samples are brighter.

No matplotlib, PIL, or ffmpeg is required.  Frames are deterministic for a
given input and argument set.

Usage::

    python render/render_control.py --trajectory data/traj.csv --out frames/depth
    python render/render_control.py --trajectory data/traj.csv --out frames/depth \
        --frames 121 --fps 24 --width 832 --height 480

The printed ffmpeg command encodes the frames to a video; running it is
optional.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import shlex
import shutil
import sys
from pathlib import Path

import numpy as np

TIME_ALIASES = ("t", "time", "epoch", "seconds")
FRAME_GLOB = "frame_*.ppm"
FRAME_PATTERN = "frame_%04d.ppm"

#: Depth-map grayscale endpoints.  Both are non-zero so the far end of the
#: trajectory is still visible against the black background.
NEAR_GRAY = 255
FAR_GRAY = 32


class TrajectoryError(ValueError):
    """The trajectory CSV cannot be parsed or rendered."""


def read_trajectory(path: str | Path) -> list[tuple[float, float, float, float]]:
    """Load ``(t, x, y, z)`` rows from a CSV file.

    Column names are matched case-insensitively; ``t`` also accepts ``time``,
    ``epoch``, and ``seconds``.  Unknown columns, blank lines, and ``#``
    comments are ignored.  Rows must be finite and strictly increasing in time.
    """

    path = Path(path)
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise TrajectoryError(f"cannot read {path}: {exc}") from exc

    with handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise TrajectoryError(f"{path}: empty trajectory") from None

        names = [name.strip().lower() for name in header]
        columns: dict[str, int] = {}
        for field in ("t", "x", "y", "z"):
            aliases = TIME_ALIASES if field == "t" else (field,)
            for index, name in enumerate(names):
                if name in aliases:
                    columns[field] = index
                    break
            else:
                raise TrajectoryError(
                    f"{path}: missing required column '{field}' "
                    f"(header: {', '.join(names)})"
                )

        required = max(columns.values())
        samples: list[tuple[float, float, float, float]] = []
        for line_no, row in enumerate(reader, start=2):
            if not row or not any(field.strip() for field in row):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            if len(row) <= required:
                raise TrajectoryError(
                    f"{path}:{line_no}: expected at least {required + 1} "
                    f"fields, found {len(row)}"
                )
            values = {}
            for field, index in columns.items():
                text = row[index].strip()
                try:
                    values[field] = float(text)
                except ValueError as exc:
                    raise TrajectoryError(
                        f"{path}:{line_no}: column '{field}' is not a number: "
                        f"{text!r}"
                    ) from exc
            if not all(math.isfinite(value) for value in values.values()):
                raise TrajectoryError(f"{path}:{line_no}: value is not finite")
            sample = (values["t"], values["x"], values["y"], values["z"])
            if samples and sample[0] <= samples[-1][0]:
                raise TrajectoryError(
                    f"{path}:{line_no}: time must increase "
                    f"(previous {samples[-1][0]}, current {sample[0]})"
                )
            samples.append(sample)

    if not samples:
        raise TrajectoryError(f"{path}: no data rows")
    return samples


def _camera_basis(view_dir) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(toward_viewer, right, up)`` unit vectors."""

    direction = np.asarray(view_dir, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise TrajectoryError("view direction must be three finite numbers")
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise TrajectoryError("view direction must be non-zero")
    forward = direction / norm

    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up))) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return forward, right, up


def _resample(
    samples: list[tuple[float, float, float, float]],
    times: list[float],
) -> np.ndarray:
    """Linearly interpolate positions at ``times`` (clamped to the data)."""

    sample_times = [sample[0] for sample in samples]
    positions = np.empty((len(times), 3), dtype=float)
    last = len(samples) - 1

    for row, t in enumerate(times):
        if len(samples) == 1:
            positions[row] = samples[0][1:4]
            continue
        index = bisect.bisect_left(sample_times, t)
        if index <= 0:
            positions[row] = samples[0][1:4]
        elif index > last:
            positions[row] = samples[last][1:4]
        elif sample_times[index] == t:
            positions[row] = samples[index][1:4]
        else:
            lo = samples[index - 1]
            hi = samples[index]
            alpha = (t - lo[0]) / (hi[0] - lo[0])
            positions[row] = [
                lo[axis] + (hi[axis] - lo[axis]) * alpha for axis in (1, 2, 3)
            ]
    return positions


def _frame_times(t0: float, t1: float, frame_count: int) -> list[float]:
    if frame_count <= 1 or t1 <= t0:
        return [t0]
    step = (t1 - t0) / (frame_count - 1)
    return [t0 + step * index for index in range(frame_count)]


def _auto_frame_count(
    samples: list[tuple[float, float, float, float]], start: float, end: float
) -> int:
    """One frame per sample interval in the window, at the input cadence."""

    if len(samples) < 2 or end <= start:
        return 1
    duration = samples[-1][0] - samples[0][0]
    if duration <= 0.0:
        return 1
    dt = duration / (len(samples) - 1)
    return max(1, int(math.floor((end - start) / dt + 0.5)) + 1)


def _project(
    positions: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return positions @ right, positions @ up, positions @ forward


def _pixel_transform(
    u: np.ndarray,
    v: np.ndarray,
    width: int,
    height: int,
    margin: int,
) -> tuple[float, float, float]:
    """Map scene coordinates to pixels, preserving aspect ratio.

    Returns ``(u_origin, v_origin, scale)`` such that
    ``column = (u - u_origin) * scale`` and
    ``row = height - 1 - (v - v_origin) * scale``.
    """

    usable_w = width - 1 - 2 * margin
    usable_h = height - 1 - 2 * margin
    if usable_w < 1 or usable_h < 1:
        raise TrajectoryError(
            f"resolution {width}x{height} is too small for margin {margin}"
        )

    u_min = float(np.min(u))
    v_min = float(np.min(v))
    span_u = float(np.max(u)) - u_min
    span_v = float(np.max(v)) - v_min

    scales = []
    if span_u > 0.0:
        scales.append(usable_w / span_u)
    if span_v > 0.0:
        scales.append(usable_h / span_v)
    scale = min(scales) if scales else 1.0

    pad_u = (usable_w - span_u * scale) / 2.0
    pad_v = (usable_h - span_v * scale) / 2.0
    u_origin = u_min - (margin + pad_u) / scale
    v_origin = v_min - (margin + pad_v) / scale
    return u_origin, v_origin, scale


def _depth_gray(depth: np.ndarray) -> np.ndarray:
    """Map depth to grayscale: larger (nearer) depth is brighter."""

    low = float(np.min(depth))
    span = float(np.max(depth)) - low
    if span <= 0.0:
        return np.full(depth.shape, (NEAR_GRAY + FAR_GRAY) // 2, dtype=float)
    normalized = (depth - low) / span
    return FAR_GRAY + (NEAR_GRAY - FAR_GRAY) * normalized


def _draw_sprite(
    image: np.ndarray,
    column: int,
    row: int,
    radius: int,
    value: int,
) -> None:
    height, width = image.shape
    top = max(0, row - radius)
    bottom = min(height, row + radius + 1)
    left = max(0, column - radius)
    right = min(width, column + radius + 1)
    if top < bottom and left < right:
        image[top:bottom, left:right] = value


def write_ppm(path: str | Path, image: np.ndarray) -> None:
    """Write an 8-bit grayscale array as a binary PPM (P6)."""

    if image.dtype != np.uint8:
        raise TypeError("PPM image must be uint8")
    if image.ndim != 2:
        raise ValueError("PPM image must be two-dimensional")
    height, width = image.shape
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    Path(path).write_bytes(header + image.tobytes())


def _default_caption(path: Path, frame_count: int, fps: float) -> str:
    name = path.stem
    return (
        f"Depth control for {name}: orthographic projection of the spacecraft "
        f"position, {frame_count} frames at {fps:g} fps."
    )


def render_trajectory(
    trajectory_path: str | Path,
    out_dir: str | Path,
    *,
    width: int = 832,
    height: int = 480,
    fps: float = 24.0,
    frames: int | None = None,
    t0: float | None = None,
    t1: float | None = None,
    view_dir=(0.0, 0.0, 1.0),
    margin: int = 16,
    radius: int = 2,
    caption: str | None = None,
) -> dict:
    """Render depth frames and a manifest; return the manifest dict."""

    if width <= 1 or height <= 1:
        raise TrajectoryError("width and height must be greater than one")
    if not math.isfinite(fps) or fps <= 0.0:
        raise TrajectoryError("fps must be a positive number")
    if margin < 0:
        raise TrajectoryError("margin must be non-negative")
    if radius < 0:
        raise TrajectoryError("radius must be non-negative")
    if frames is not None and frames <= 0:
        frames = None

    trajectory_path = Path(trajectory_path)
    samples = read_trajectory(trajectory_path)
    first_t = samples[0][0]
    last_t = samples[-1][0]

    start = first_t if t0 is None else min(max(float(t0), first_t), last_t)
    end = last_t if t1 is None else min(max(float(t1), first_t), last_t)
    if end < start:
        raise TrajectoryError(
            f"time window is empty: t0={start:g} is after t1={end:g}"
        )

    if frames is None:
        frame_count = _auto_frame_count(samples, start, end)
    else:
        frame_count = int(frames)

    times = _frame_times(start, end, frame_count)
    positions = _resample(samples, times)

    forward, right, up = _camera_basis(view_dir)
    u, v, depth = _project(positions, right, up, forward)

    # Fit the view using trajectory samples inside the window so relevant
    # motion is not dwarfed by portions of the path that are not rendered.
    window = [sample for sample in samples if start <= sample[0] <= end] or samples
    window_positions = np.asarray([sample[1:4] for sample in window], dtype=float)
    fit_u, fit_v, _ = _project(window_positions, right, up, forward)
    all_u = np.concatenate([u, fit_u])
    all_v = np.concatenate([v, fit_v])

    u_origin, v_origin, scale = _pixel_transform(
        all_u, all_v, width, height, margin
    )
    colors = _depth_gray(depth)

    out_dir = Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise TrajectoryError(f"output path is not a directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(FRAME_GLOB):
        stale.unlink()

    for index, (column, row, color) in enumerate(zip(u, v, colors)):
        pixel_column = int(math.floor((float(column) - u_origin) * scale + 0.5))
        pixel_row = int(
            math.floor(
                height - 1 - ((float(row) - v_origin) * scale) + 0.5
            )
        )
        image = np.zeros((height, width), dtype=np.uint8)
        gray = int(math.floor(float(color) + 0.5))
        _draw_sprite(image, pixel_column, pixel_row, radius, gray)
        write_ppm(out_dir / (FRAME_PATTERN % index), image)

    if not caption:
        caption = _default_caption(trajectory_path, frame_count, fps)

    manifest = {
        "control_type": "depth",
        "frame_count": frame_count,
        "fps": fps,
        "resolution": [width, height],
        "caption": caption,
        "frame_pattern": FRAME_PATTERN,
        "projection": "orthographic",
        "view_dir": [float(value) for value in view_dir],
        "time_range": [start, end],
        "trajectory": str(trajectory_path),
        "depth_range": [float(np.min(depth)), float(np.max(depth))],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"wrote {frame_count} depth frames to {out_dir} "
        f"({width}x{height} @ {fps:g} fps)"
    )
    print(f"wrote {manifest_path}")

    command = ffmpeg_command(out_dir, fps, frame_count)
    print("encode with:")
    print("  " + " ".join(shlex.quote(part) for part in command))
    if shutil.which("ffmpeg") is None:
        print("note: ffmpeg is not installed; encoding is optional")
    return manifest


def ffmpeg_command(out_dir: str | Path, fps: float, frame_count: int) -> list[str]:
    """Command that encodes the rendered frames into an H.264 MP4."""

    out_dir = Path(out_dir)
    return [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(out_dir / "frame_%04d.ppm"),
        "-frames:v",
        str(frame_count),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_dir / "depth.mp4"),
    ]


def _parse_view_dir(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    try:
        numbers = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return numbers  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render orthographic depth control frames from a trajectory CSV."
    )
    parser.add_argument("--trajectory", required=True, help="trajectory CSV (t,x,y,z,...)")
    parser.add_argument(
        "--out", default="results/control/depth", help="output directory"
    )
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="frame count (0 = one frame per sample in the window)",
    )
    parser.add_argument("--t0", type=float, default=None, help="window start time")
    parser.add_argument("--t1", type=float, default=None, help="window end time")
    parser.add_argument(
        "--view-dir",
        type=_parse_view_dir,
        default=(0.0, 0.0, 1.0),
        help="orthographic view direction (from scene toward viewer)",
    )
    parser.add_argument("--margin", type=int, default=16, help="pixel border")
    parser.add_argument("--radius", type=int, default=2, help="sprite half-width")
    parser.add_argument("--caption", default=None, help="manifest caption")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        render_trajectory(
            args.trajectory,
            args.out,
            width=args.width,
            height=args.height,
            fps=args.fps,
            frames=args.frames or None,
            t0=args.t0,
            t1=args.t1,
            view_dir=args.view_dir,
            margin=args.margin,
            radius=args.radius,
            caption=args.caption,
        )
    except (TrajectoryError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
