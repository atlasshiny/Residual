# scripts/render_depth.py
import argparse, numpy as np, pandas as pd, imageio.v2 as imageio

ap = argparse.ArgumentParser()
ap.add_argument("--positions", default="out/positions.csv")
ap.add_argument("--marker", default="mlp")
ap.add_argument("--earth-r", type=int, default=45)
ap.add_argument("--moon-r", type=int, default=22)
ap.add_argument("--invert", action="store_true", help="flip near/far if results look inside-out")
ap.add_argument("--out", default="out/guide_depth.mp4")
a = ap.parse_args()

W, H = 1280, 720
pos = pd.read_csv(a.positions)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

def add_sphere(img, cx, cy, r, peak):
    d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)
    z = np.sqrt(np.clip(1 - d2, 0, 1)) * peak
    np.maximum(img, z, out=img)

w = imageio.get_writer(a.out, fps=24, codec="libx264", quality=10)
for _, r in pos.iterrows():
    img = np.full((H, W), 0.04, np.float32)      # distant background, not pure black
    add_sphere(img, r.earth_px_x, r.earth_px_y, a.earth_r, 0.9)
    add_sphere(img, r.moon_px_x, r.moon_px_y, a.moon_r, 0.6)
    add_sphere(img, r[f"{a.marker}_px_x"], r[f"{a.marker}_px_y"], 7, 0.5)
    if a.invert:
        img = 1 - img
    g = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    w.append_data(np.stack([g, g, g], -1))
w.close()
print("wrote", a.out, len(pos), "frames")