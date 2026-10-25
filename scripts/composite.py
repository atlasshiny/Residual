# scripts/composite.py
import argparse, numpy as np, pandas as pd, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFilter

ap = argparse.ArgumentParser()
ap.add_argument("--restyled", required=True)
ap.add_argument("--positions", required=True)
ap.add_argument("--out", default="out/final.mp4")
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--earth-r", type=int, default=45)
ap.add_argument("--moon-r", type=int, default=22)
ap.add_argument("--marker", default=None, help="path name for the Orion marker (default: first non-truth path)")
a = ap.parse_args()

W, H, SS = 1280, 720, 2
pos = pd.read_csv(a.positions)
names = [c[:-5] for c in pos.columns if c.endswith("_px_x") and c[:-5] not in ("earth", "moon")]
marker = a.marker or next((n for n in names if n != "truth"), "truth")
colors = {"truth": (255, 255, 255), "mlp": (0, 220, 255), "pinn": (255, 150, 40), "sindy": (230, 80, 230)}
fallback = [(120, 255, 120), (255, 255, 120)]
col = {n: colors.get(n, fallback[i % 2]) for i, n in enumerate(names)}

# One static star field, reused for every frame (stationary by construction)
rng = np.random.default_rng(7)
bg = np.zeros((H, W, 3), np.float32)
for _ in range(900):
    x, y = rng.integers(0, W), rng.integers(0, H)
    bg[y, x] = rng.uniform(70, 255)
big = Image.fromarray(bg.astype(np.uint8))
for _ in range(60):
    x, y = rng.integers(1, W - 2), rng.integers(1, H - 2)
    ImageDraw.Draw(big).ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255, 255, 255))
stars = big.filter(ImageFilter.GaussianBlur(0.6))

reader = imageio.get_reader(a.restyled)
n_vid = reader.count_frames()
n = min(n_vid, len(pos))
if n_vid != len(pos):
    print(f"warning: video has {n_vid} frames, positions {len(pos)}; using {n}")

writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=9)
for f in range(n):
    frame = Image.fromarray(reader.get_data(f)).convert("RGB").resize((W, H), Image.LANCZOS)
    r = pos.iloc[f]

    # Keep only the restyled pixels around Earth and the Moon; everything else is our stars
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    for key, rad in (("earth", a.earth_r), ("moon", a.moon_r)):
        cx, cy, rr = r[f"{key}_px_x"], r[f"{key}_px_y"], rad * 1.3
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(4))
    base = Image.composite(frame, stars, mask)

    # Trails + Orion, drawn from the data at 2x, then downsampled
    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    for n_ in names:
        pts = [(pos[f"{n_}_px_x"].iloc[i] * SS, pos[f"{n_}_px_y"].iloc[i] * SS) for i in range(f + 1)]
        if len(pts) > 1:
            od.line(pts, fill=col[n_] + (255,), width=3 if n_ == "truth" else 4)
    ox, oy = r[f"{marker}_px_x"] * SS, r[f"{marker}_px_y"] * SS
    for rad, al in ((16, 60), (11, 120)):
        od.ellipse([ox - rad * SS / 2, oy - rad * SS / 2, ox + rad * SS / 2, oy + rad * SS / 2], fill=(255, 150, 0, al))
    od.ellipse([ox - 5 * SS / 2, oy - 5 * SS / 2, ox + 5 * SS / 2, oy + 5 * SS / 2], fill=(255, 235, 200, 255))
    ov = ov.resize((W, H), Image.LANCZOS)
    base = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")

    dd = ImageDraw.Draw(base)
    dd.text((16, H - 24), "Bodies and spacecraft enlarged; not to scale", fill=(170, 170, 170))
    for i, n_ in enumerate(names):
        dd.text((16, 14 + 14 * i), n_, fill=col[n_])
    writer.append_data(np.asarray(base))
writer.close()
print("wrote", a.out)