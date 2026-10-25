import argparse, base64, io, requests, imageio.v2 as imageio
from PIL import Image, ImageDraw
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="out/guide_bodies49.mp4")
ap.add_argument("--positions", default="out/positions49.csv")
ap.add_argument("--strengths", default="0.3,0.5,0.7,0.9")
ap.add_argument("--seed", default="17")
ap.add_argument("--base-url", default="http://localhost:8000")
a = ap.parse_args()

frame = Image.fromarray(imageio.get_reader(a.video).get_data(0)).convert("RGB").resize((1280, 720))
buf = io.BytesIO(); frame.save(buf, "PNG"); buf.seek(0)
r0 = pd.read_csv(a.positions).iloc[0]
circles = ((r0.earth_px_x, r0.earth_px_y, 45), (r0.moon_px_x, r0.moon_px_y, 22))

prompt = ("Photorealistic space photograph. A blue Earth with white clouds and a thin atmospheric glow "
          "and a gray cratered Moon on a black star field. Keep each body at exactly the same position "
          "and size as in the input image. Natural sunlight, sharp stars, no text.")

for s in a.strengths.split(","):
    buf.seek(0)
    r = requests.post(a.base_url + "/v1/images/edits",
        data={"model": "nvidia/Cosmos3-Super", "prompt": prompt, "size": "auto",
              "resolution": "1024", "strength": s, "seed": a.seed,
              "num_inference_steps": "30", "n": "1", "response_format": "b64_json"},
        files={"image[]": ("frame.png", buf, "image/png")}, timeout=1800)
    if not r.ok:
        print("strength", s, r.status_code, r.text[:600]); continue
    img = Image.open(io.BytesIO(base64.b64decode(r.json()["data"][0]["b64_json"]))).convert("RGB").resize((1280, 720))
    img.save(f"out/edit_s{s}.png")
    d = ImageDraw.Draw(img)
    for cx, cy, rad in circles:
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(255, 0, 0), width=2)
    img.save(f"out/edit_s{s}_check.png")
    print("saved strength", s)