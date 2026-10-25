# scripts/composite_sprites.py
import argparse, numpy as np, pandas as pd, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFilter

ap = argparse.ArgumentParser()
ap.add_argument("--earth", required=True)
ap.add_argument("--moon", required=True)
ap.add_argument("--positions", default="out/positions.csv")
ap.add_argument("--marker", default="mlp")
ap.add_argument("--earth-r", type=int, default=45)
ap.add_argument("--moon-r", type=int, default=22)
ap.add_argument("--spin", type=float, default=0.0, help="Moon sprite degrees per frame")
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--out", default="out/final.mp4")
a = ap.parse_args()

W, H, SS = 1280, 720, 2
pos = pd.read_csv(a.positions)
names = [c[:-5] for c in pos.columns if c.endswith("_px_x") and c[:-5] not in ("earth", "moon")]
colors = {"truth": (255, 255, 255), "mlp": (0, 220, 255), "pinn": (255, 150, 40), "sindy": (230, 80, 230)}
col = {n: colors.get(n, (120, 255, 120)) for n in names}

def load_disc(path):
    im = Image.open(path).convert("RGB")
    lum = np.asarray(im.convert("L")).astype(np.float32)
    mask = lum > max(25, 0.15 * lum.max())
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy, s = (x0 + x1) // 2, (y0 + y1) // 2, max(x1 - x0, y1 - y0) // 2
    return im.crop((cx - s, cy - s, cx + s, cy + s))

def circle_sprite(im, r, angle=0.0):
    if angle:
        im = im.rotate(angle, resample=Image.BICUBIC)
    im = im.resize((2 * r, 2 * r), Image.LANCZOS)
    m = Image.new("L", (2 * r, 2 * r), 0)
    ImageDraw.Draw(m).ellipse([1, 1, 2 * r - 2, 2 * r - 2], fill=255)
    out = im.convert("RGBA"); out.putalpha(m.filter(ImageFilter.GaussianBlur(0.8)))
    return out

earth_src, moon_src = load_disc(a.earth), load_disc(a.moon)
earth = circle_sprite(earth_src, a.earth_r)

rng = np.random.default_rng(7)
bg = np.zeros((H, W, 3), np.uint8)
for _ in range(900):
    bg[rng.integers(0, H), rng.integers(0, W)] = int(rng.uniform(70, 255))
stars = Image.fromarray(bg)
sd = ImageDraw.Draw(stars)
for _ in range(60):
    x, y = int(rng.integers(1, W - 2)), int(rng.integers(1, H - 2))
    sd.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255, 255, 255))
stars = stars.filter(ImageFilter.GaussianBlur(0.6)).convert("RGBA")

writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=9)
for f in range(len(pos)):
    r = pos.iloc[f]
    base = stars.copy()
    base.alpha_composite(earth, (int(round(r.earth_px_x)) - a.earth_r, int(round(r.earth_px_y)) - a.earth_r))
    moon = circle_sprite(moon_src, a.moon_r, a.spin * f)
    base.alpha_composite(moon, (int(round(r.moon_px_x)) - a.moon_r, int(round(r.moon_px_y)) - a.moon_r))

    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    for n in names:
        pts = [(pos[f"{n}_px_x"].iloc[i] * SS, pos[f"{n}_px_y"].iloc[i] * SS) for i in range(f + 1)]
        if len(pts) > 1:
            od.line(pts, fill=col[n] + (255,), width=3 if n == "truth" else 4)
    ox, oy = r[f"{a.marker}_px_x"] * SS, r[f"{a.marker}_px_y"] * SS
    for rad, al in ((16, 60), (10, 130)):
        od.ellipse([ox - rad * SS / 2, oy - rad * SS / 2, ox + rad * SS / 2, oy + rad * SS / 2], fill=(255, 150, 0, al))
    od.ellipse([ox - 2.5 * SS, oy - 2.5 * SS, ox + 2.5 * SS, oy + 2.5 * SS], fill=(255, 235, 200, 255))
    base = Image.alpha_composite(base, ov.resize((W, H), Image.LANCZOS))

    dd = ImageDraw.Draw(base)
    dd.text((16, H - 24), "Bodies and spacecraft enlarged; not to scale", fill=(170, 170, 170, 255))
    for i, n in enumerate(names):
        dd.text((16, 14 + 14 * i), n, fill=col[n] + (255,))
    writer.append_data(np.asarray(base.convert("RGB")))
writer.close()
print("wrote", a.out, len(pos), "frames")