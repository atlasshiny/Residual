# render/

Headless control-video renderer for the Cosmos transfer job. The first
renderer is `render_control.py`: it turns a trajectory CSV into an
orthographic **depth** control sequence (grayscale PPM frames) plus a
`manifest.json`.

No matplotlib, PIL, or ffmpeg is required. Frame content is deterministic for a
given trajectory and argument set.

## Install

```bash
uv sync --extra render
```

The `render` extra contains only `numpy`. `render_control.py` also works with
the `data` extra, which already includes numpy.

## Run

```bash
python render/render_control.py --trajectory data/traj.csv --out results/control/depth
```

Or select a clip window and frame count:

```bash
python render/render_control.py \
    --trajectory data/traj.csv \
    --out results/control/tli_depth \
    --frames 121 --fps 24 --width 832 --height 480 \
    --t0 0 --t1 3600 \
    --caption "TLI outbound coast, depth control"
```

`python -m render.render_control ...` works from the repository root as well.

### Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--trajectory` | required | trajectory CSV |
| `--out` | `results/control/depth` | output directory (stale `frame_*.ppm` files are removed) |
| `--width`, `--height` | `832`, `480` | frame resolution |
| `--fps` | `24` | manifest/encoding frame rate |
| `--frames` | `0` | output frame count; `0` = follow the input cadence for the window |
| `--t0`, `--t1` | full range | time window in trajectory seconds (clamped to the data) |
| `--view-dir` | `0,0,1` | orthographic view direction, from the scene toward the viewer |
| `--margin` | `16` | pixel border kept free around the projected path |
| `--radius` | `2` | sprite half-width in pixels |
| `--caption` | generated | caption stored in the manifest |

## Input

Canonical trajectory shape (same as `crates/ingest`):

```text
t,x,y,z,vx,vy,vz
0,7000000,0,0,0,7546.05,0
60,6999962,452761,0,-488.1,7545.9,0
```

- `t` also accepts `time`, `epoch`, or `seconds`; names are case-insensitive.
- Only `t`, `x`, `y`, `z` are used. Extra columns are ignored.
- Blank lines and lines starting with `#` are skipped.
- Times must be finite and strictly increasing.
- Frames inside `[t0, t1]` are linearly interpolated in position.

## Output

```
results/control/depth/
├─ frame_0000.ppm
├─ frame_0001.ppm
├─ ...
└─ manifest.json
```

Each frame is a binary P6 grayscale PPM. The projection is orthographic:
`--view-dir` becomes intensity (nearer = brighter, range 32–255) while the two
perpendicular axes map to pixels with the aspect ratio preserved. The
trajectory window is normalized once, so the brightness and framing stay fixed
across the clip.

For a planar trajectory such as `z = 0`, the default view (`0,0,1`) gives a
constant depth. Tilt the view (for example `--view-dir 0.4,-0.6,0.7`) when the
depth channel should vary across the clip.

`manifest.json`:

```json
{
  "control_type": "depth",
  "frame_count": 121,
  "fps": 24.0,
  "resolution": [832, 480],
  "caption": "Depth control for traj: orthographic projection of the spacecraft position, 121 frames at 24 fps.",
  "frame_pattern": "frame_%04d.ppm",
  "projection": "orthographic",
  "view_dir": [0.0, 0.0, 1.0],
  "time_range": [0.0, 3600.0],
  "trajectory": "data/traj.csv",
  "depth_range": [-452761.0, 7000000.0]
}
```

The manifest always carries `control_type: "depth"`, `frame_count`, `fps`,
`resolution`, and `caption`, matching the control-video fields in
`ARCHITECTURE.md` §7A.

## Encode (optional)

The script prints the exact command, for example:

```bash
ffmpeg -y -framerate 24 -i results/control/depth/frame_%04d.ppm \
    -frames:v 121 -c:v libx264 -pix_fmt yuv420p results/control/depth/depth.mp4
```

It does not run ffmpeg and does not fail when ffmpeg is absent. If this command
is run on a machine without ffmpeg, copy the PPM frames and encode elsewhere.
