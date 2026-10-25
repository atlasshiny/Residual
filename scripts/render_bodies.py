# scripts/render_bodies.py
import argparse, numpy as np, pandas as pd, imageio.v2 as imageio
from PIL import Image

MU_E, MU_M, D_NOM = 3.986004418e14, 4.9048695e12, 3.844e8
W, H, EARTH_R, MOON_R, PAD = 1280, 720, 45, 22, 90

ap = argparse.ArgumentParser()
ap.add_argument("--reference", required=True)
ap.add_argument("--moon", required=True)
ap.add_argument("--path", action="append", default=[], help="name=csv, e.g. mlp=out/corrected_mlp.csv")
ap.add_argument("--frames", type=int, default=189)
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--out", default="out/guide_bodies.mp4")
ap.add_argument("--positions", default="out/positions.csv")
a = ap.parse_args()

ref, moon = pd.read_csv(a.reference), pd.read_csv(a.moon)
t = ref["t"].values
paths = {"truth": ref[["x", "y", "z"]].values}
for spec in a.path:
    name, f = spec.split("=", 1)
    c = pd.read_csv(f)
    paths[name] = np.column_stack([np.interp(t, c["t"], c[k]) for k in "xyz"])

# Moon into the same frame as convert_spacecraft.py
m = moon[["x", "y", "z"]].values * 1000.0
v0 = moon[["vx", "vy", "vz"]].values[0] * 1000.0
xh = m[0] / np.linalg.norm(m[0])
zh = np.cross(m[0], v0); zh /= np.linalg.norm(zh)
R = np.vstack([xh, np.cross(zh, xh), zh])
off = -D_NOM * MU_M / (MU_E + MU_M)
M = m @ R.T + np.array([off, 0, 0])
moon_i = np.column_stack([np.interp(t, moon["t"].values, M[:, i]) for i in range(3)])
earth = np.array([off, 0.0])

allp = np.vstack([p[:, :2] for p in paths.values()] + [moon_i[:, :2], earth[None]])
lo, hi = allp.min(0), allp.max(0)
ctr = (lo + hi) / 2
scale = min((W - 2 * PAD) / (hi[0] - lo[0]), (H - 2 * PAD) / (hi[1] - lo[1]))
px = lambda p: (W / 2 + (p[0] - ctr[0]) * scale, H / 2 - (p[1] - ctr[1]) * scale)

def sphere(r, color):
    s = np.arange(-r, r + 1) / r
    xx, yy = np.meshgrid(s, -s)
    d = np.sqrt(xx**2 + yy**2)
    z = np.sqrt(np.clip(1 - d**2, 0, 1))
    L = np.array([-0.5, 0.5, 0.7]); L /= np.linalg.norm(L)
    shade = 0.10 + 0.90 * np.clip(xx * L[0] + yy * L[1] + z * L[2], 0, 1)
    rgb = np.array(color)[None, None, :] * shade[..., None]
    alpha = np.clip((1 - d) * r, 0, 1) * 255
    return Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA")

earth_s, moon_s = sphere(EARTH_R, (50, 110, 210)), sphere(MOON_R, (175, 175, 168))
tk = np.linspace(t[0], t[-1], a.frames)
interp = lambda arr, tf: np.array([np.interp(tf, t, arr[:, i]) for i in range(arr.shape[1])])

writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=9)
rows = []
ex, ey = px(earth)
for f, tf in enumerate(tk):
    mn = interp(moon_i, tf)
    mx, my = px(mn)
    img = Image.new("RGB", (W, H), (0, 0, 0))
    img.paste(earth_s, (int(ex) - EARTH_R, int(ey) - EARTH_R), earth_s)
    img.paste(moon_s, (int(mx) - MOON_R, int(my) - MOON_R), moon_s)
    writer.append_data(np.asarray(img))
    row = {"frame": f, "t": tf, "earth_px_x": ex, "earth_px_y": ey, "moon_px_x": mx, "moon_px_y": my}
    for name, arr in paths.items():
        ox, oy = px(interp(arr, tf))
        row[f"{name}_px_x"], row[f"{name}_px_y"] = ox, oy
    rows.append(row)
writer.close()
pd.DataFrame(rows).to_csv(a.positions, index=False)
print("wrote", a.out, a.positions)