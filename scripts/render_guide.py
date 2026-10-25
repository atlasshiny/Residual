# scripts/render_guide.py
import argparse, numpy as np, pandas as pd, imageio.v2 as imageio
from PIL import Image, ImageDraw

MU_E, MU_M, D_NOM = 3.986004418e14, 4.9048695e12, 3.844e8

ap = argparse.ArgumentParser()
ap.add_argument("--reference", required=True)      # data/reference_artemis2.csv
ap.add_argument("--moon", required=True)           # data/raw_moon_artemis2.csv (km)
ap.add_argument("--corrected")                     # corrected_trajectory.csv (optional)
ap.add_argument("--frames", type=int, default=189)
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--out", default="guide.mp4")
ap.add_argument("--positions", default="guide_positions.csv")
a = ap.parse_args()

W, H, SS = 1280, 720, 2
ref = pd.read_csv(a.reference)
moon = pd.read_csv(a.moon)
t = ref["t"].values
P = ref[["x", "y", "z"]].values

# Moon into the same frame as convert_spacecraft.py
m = moon[["x", "y", "z"]].values * 1000.0
v0 = moon[["vx", "vy", "vz"]].values[0] * 1000.0
d0 = m[0]
h = np.cross(d0, v0)
xh = d0 / np.linalg.norm(d0)
zh = h / np.linalg.norm(h)
yh = np.cross(zh, xh)
R = np.vstack([xh, yh, zh])
off = -D_NOM * MU_M / (MU_E + MU_M)
M = m @ R.T + np.array([off, 0, 0])
Mt = moon["t"].values
Mi = np.column_stack([np.interp(t, Mt, M[:, i]) for i in range(3)])
earth = np.array([off, 0.0])

C = None
if a.corrected:
    c = pd.read_csv(a.corrected)
    C = np.column_stack([np.interp(t, c["t"], c[k]) for k in "xyz"])

pts = [P[:, :2], Mi[:, :2], earth[None, :]] + ([C[:, :2]] if C is not None else [])
allp = np.vstack(pts)
lo, hi = allp.min(0), allp.max(0)
ctr = (lo + hi) / 2
scale = 0.88 * min(W * SS / (hi[0] - lo[0]), H * SS / (hi[1] - lo[1]))

def px(p):
    return (W * SS / 2 + (p[0] - ctr[0]) * scale, H * SS / 2 - (p[1] - ctr[1]) * scale)

rng = np.random.default_rng(3)
stars = [(rng.integers(0, W * SS), rng.integers(0, H * SS), rng.integers(1, 3)) for _ in range(500)]

def disc(d, p, r, fill):
    x, y = px(p)
    d.ellipse([x - r, y - r, x + r, y + r], fill=fill)

def trail(d, arr, k, fill, w):
    q = [px(p) for p in arr[: k + 1, :2]]
    if len(q) > 1:
        d.line(q, fill=fill, width=w)

tk = np.linspace(t[0], t[-1], a.frames)
writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8)
rows = []
for f, tf in enumerate(tk):
    k = int(np.searchsorted(t, tf, side="right") - 1)
    ip = lambda arr: np.array([np.interp(tf, t, arr[:, i]) for i in range(arr.shape[1])])
    sc, mn = ip(P), ip(Mi)
    img = Image.new("RGB", (W * SS, H * SS), (0, 0, 0))
    d = ImageDraw.Draw(img)
    for x, y, r in stars:
        d.ellipse([x, y, x + r, y + r], fill=(200, 200, 200))
    trail(d, Mi, k, (90, 90, 90), 3)
    trail(d, P, k, (255, 255, 255), 4)
    if C is not None:
        trail(d, C, k, (0, 220, 255), 4)
    disc(d, earth, 16 * SS, (40, 90, 200))
    disc(d, mn[:2], 8 * SS, (170, 170, 170))
    disc(d, (ip(C)[:2] if C is not None else sc[:2]), 6 * SS, (255, 140, 0))
    writer.append_data(np.asarray(img.resize((W, H), Image.LANCZOS)))
    rows.append([f, tf, *map(lambda p: px(p)[0] / SS, [sc]), px(sc)[1] / SS,
                 px(mn)[0] / SS, px(mn)[1] / SS, *sc, *mn])
writer.close()
pd.DataFrame(rows, columns=["frame", "t", "orion_px_x", "orion_px_y", "moon_px_x", "moon_px_y",
                            "ox", "oy", "oz", "mx", "my", "mz"]).to_csv(a.positions, index=False)
print("wrote", a.out, a.positions)