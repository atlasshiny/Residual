import argparse, json, requests

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)          # out/guide_mlp.mp4
ap.add_argument("--control-type", required=True)   # value from step 1
ap.add_argument("--prompt", required=True)
ap.add_argument("--frames", default="189")
ap.add_argument("--size", default="1280x720")
ap.add_argument("--steps", default="35")
ap.add_argument("--seed", default="17")
ap.add_argument("--out", default="out/cosmos_restyled.mp4")
ap.add_argument("--base-url", default="http://localhost:8000")
a = ap.parse_args()

data = {
    "model": "nvidia/Cosmos3-Super",
    "prompt": a.prompt,
    "size": a.size,
    "num_frames": a.frames,
    "fps": "24",
    "num_inference_steps": a.steps,
    "guidance_scale": "6.0",
    "max_sequence_length": "4096",
    "flow_shift": "10.0",
    "seed": a.seed,
    "control_type": a.control_type,
    "extra_params": json.dumps({"use_resolution_template": False,
                                "use_duration_template": False,
                                "guardrails": False}),   # True if your prompt passes
}
with open(a.video, "rb") as f:
    r = requests.post(a.base_url + "/v1/videos/sync",
                      data=data,
                      files={"control_reference": (a.video.split("/")[-1], f, "video/mp4")},
                      headers={"Accept": "video/mp4"}, timeout=7200)
if not r.ok:
    print(r.status_code, r.text[:2000]); raise SystemExit(1)
open(a.out, "wb").write(r.content)
print("saved", a.out, len(r.content))