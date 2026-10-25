# scripts/gen_textures.py
import argparse, base64, io, os, requests
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--outdir", default="out/tex")
ap.add_argument("--base-url", default="http://localhost:8000")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)

prompts = {
  "earth": "Photorealistic planet Earth, a single full sphere centered in the frame, deep blue oceans, white swirling clouds, thin blue atmospheric glow, lit from the upper left, pure black background, no stars",
  "moon": "Photorealistic full Moon, a single full sphere centered in the frame, gray cratered surface with darker maria, lit from the upper left, pure black background, no stars",
}
for name, p in prompts.items():
    for i in range(a.n):
        body = {"model": "nvidia/Cosmos3-Super", "prompt": p, "size": "1024x1024", "n": 1, "seed": 100 + i}
        r = requests.post(a.base_url + "/v1/images/generations", json=body, timeout=1800)
        if not r.ok and "seed" in r.text:
            body.pop("seed"); r = requests.post(a.base_url + "/v1/images/generations", json=body, timeout=1800)
        if not r.ok:
            print(name, i, r.status_code, r.text[:300]); continue
        Image.open(io.BytesIO(base64.b64decode(r.json()["data"][0]["b64_json"]))).convert("RGB").save(f"{a.outdir}/{name}_{i}.png")
        print("saved", name, i)

for name in prompts:
    ims = [Image.open(f"{a.outdir}/{name}_{i}.png").resize((256, 256)) for i in range(a.n) if os.path.exists(f"{a.outdir}/{name}_{i}.png")]
    if ims:
        sheet = Image.new("RGB", (256 * len(ims), 256))
        for k, im in enumerate(ims): sheet.paste(im, (256 * k, 0))
        sheet.save(f"{a.outdir}/{name}_sheet.png")
print("open out/tex/earth_sheet.png and out/tex/moon_sheet.png")