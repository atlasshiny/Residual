# scripts/track_error.py
import argparse, numpy as np, pandas as pd, imageio.v2 as imageio
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

W, H = 1280, 720
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, help="Cosmos clip (.mp4) or image (.png)")
ap.add_argument("--positions", required=True)
ap.add_argument("--static-row", type=int, default=None, help="use this positions row for every frame")
ap.add_argument("--earth-r", type=float, default=45)
ap.add_argument("--moon-r", type=float, default=22)
ap.add_argument("--annotate", default=None, help="output overlay (.mp4 or .png)")
ap.add_argument("--csv", default="out/track_error.csv")
a = ap.parse_args()

def largest(mask, min_area):
    lab, n = ndi.label(mask)
    if n == 0:
        return None
    areas = ndi.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(areas))
    return (lab == k + 1) if areas[k] >= min_area else None

def stats(m):
    ys, xs = np.nonzero(m)
    return xs.mean(), ys.mean(), float(np.sqrt(m.sum() / np.pi))

def detect(frame):
    f = frame.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    earth_m = ndi.binary_opening((b > r + 40) & (b > g) & (b > 60), np.ones((9, 9)))
    e = largest(earth_m, 400)
    earth = None
    if e is not None:
        e = ndi.binary_fill_holes(e)
        earth = stats(e)
    gray = ((f.max(2) - f.min(2)) < 40) & (f.mean(2) > 130)
    gray = ndi.binary_opening(gray, np.ones((7, 7)))
    if earth is not None:   # ignore white clouds inside/near Earth
        yy, xx = np.mgrid[0:H, 0:W]
        gray &= ((xx - earth[0]) ** 2 + (yy - earth[1]) ** 2) > (1.2 * earth[2]) ** 2
    m = largest(gray, 150)
    return earth, (stats(m) if m is not None else None)

pos = pd.read_csv(a.positions)
is_img = a.input.lower().endswith((".png", ".jpg", ".jpeg"))
frames = [np.asarray(Image.open(a.input).convert("RGB").resize((W, H)))] if is_img else \
         [np.asarray(Image.fromarray(x).convert("RGB").resize((W, H))) for x in imageio.get_reader(a.input)]
if not is_img and a.static_row is None and len(frames) != len(pos):
    print(f"warning: {len(frames)} video frames vs {len(pos)} position rows; using the shorter")

rows, overlays = [], []
for i, fr in enumerate(frames):
    pr = pos.iloc[a.static_row if a.static_row is not None else min(i, len(pos) - 1)]
    ex, ey, mx, my = pr.earth_px_x, pr.earth_px_y, pr.moon_px_x, pr.moon_px_y
    earth, moon = detect(fr)
    row = {"frame": i}
    if earth:
        row.update(earth_dx=earth[0] - ex, earth_dy=earth[1] - ey,
                   earth_err_px=np.hypot(earth[0] - ex, earth[1] - ey), earth_r_ratio=earth[2] / a.earth_r)
    if moon:
        row.update(moon_dx=moon[0] - mx, moon_dy=moon[1] - my,
                   moon_err_px=np.hypot(moon[0] - mx, moon[1] - my), moon_r_ratio=moon[2] / a.moon_r)
    if earth and moon:
        v_exp = np.array([mx - ex, my - ey]); v_det = np.array([moon[0] - earth[0], moon[1] - earth[1]])
        ang = np.degrees(np.arctan2(v_det[1], v_det[0]) - np.arctan2(v_exp[1], v_exp[0]))
        row["axis_angle_err_deg"] = (ang + 180) % 360 - 180
        row["separation_ratio"] = np.linalg.norm(v_det) / np.linalg.norm(v_exp)
    rows.append(row)
    if a.annotate:
        im = Image.fromarray(fr); d = ImageDraw.Draw(im)
        for (cx, cy, rad), c in (((ex, ey, a.earth_r), (255, 0, 0)), ((mx, my, a.moon_r), (255, 0, 0)),
                                  (earth, (0, 255, 0)), (moon, (0, 255, 0))):
            if cx is not None:
                d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=c, width=2)
        overlays.append(np.asarray(im))

df = pd.DataFrame(rows)
df.to_csv(a.csv, index=False)
diag = float(np.hypot(W, H))
print(f"{len(df)} frame(s); detection rate: Earth {df.get('earth_err_px', pd.Series(dtype=float)).notna().mean():.0%}, "
      f"Moon {df.get('moon_err_px', pd.Series(dtype=float)).notna().mean():.0%}")
for k, label in (("earth_err_px", "Earth position error"), ("moon_err_px", "Moon position error")):
    if k in df and df[k].notna().any():
        print(f"{label}: median {df[k].median():.0f} px, max {df[k].max():.0f} px "
              f"({100 * df[k].median() / diag:.1f}% of frame diagonal)")
for k, label in (("earth_r_ratio", "Earth size vs guide"), ("moon_r_ratio", "Moon size vs guide"),
                 ("axis_angle_err_deg", "Earth-Moon axis angle error (deg)"), ("separation_ratio", "Earth-Moon separation ratio")):
    if k in df and df[k].notna().any():
        print(f"{label}: median {df[k].median():.2f}")
print("wrote", a.csv)

if a.annotate:
    if is_img or a.annotate.endswith(".png"):
        Image.fromarray(overlays[0]).save(a.annotate)
    else:
        w = imageio.get_writer(a.annotate, fps=24, codec="libx264", quality=8)
        for o in overlays: w.append_data(o)
        w.close()
    print("wrote", a.annotate)