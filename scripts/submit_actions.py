import argparse, json, requests

ap = argparse.ArgumentParser()
ap.add_argument("--actions", required=True)
ap.add_argument("--first-frame", required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--out", default="out/action_mlp.mp4")
ap.add_argument("--steps", default="20")
ap.add_argument("--seed", default="17")
ap.add_argument("--base-url", default="http://localhost:8000")
a = ap.parse_args()

actions = json.load(open(a.actions))["actions"]
assert 16 <= len(actions) <= 400, len(actions)
assert all(len(r) == 9 for r in actions), "expected 9D rows"

data = {
    "model": "nvidia/Cosmos3-Super",
    "prompt": a.prompt,
    "size": "1280x720",
    "num_frames": str(len(actions) + 1),
    "fps": "24",
    "num_inference_steps": a.steps,
    "guidance_scale": "6.0",
    "flow_shift": "10.0",
    "seed": a.seed,
    "extra_params": json.dumps({
        "action_mode": "forward_dynamics",
        "domain_name": "camera_pose",
        "raw_action_dim": 9,
        "action_chunk_size": len(actions),
        "action": actions,
        "resolution": "720", "image_size": 720,
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": False,
    }),
}
with open(a.first_frame, "rb") as f:
    r = requests.post(a.base_url + "/v1/videos/sync", data=data,
                      files={"input_reference": ("first.png", f, "image/png")},
                      headers={"Accept": "video/mp4"}, timeout=7200)
if not r.ok:
    print(r.status_code, r.text[:2000]); raise SystemExit(1)
open(a.out, "wb").write(r.content)
print("saved", a.out, len(r.content))