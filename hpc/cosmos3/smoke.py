"""Exercise a running Cosmos3-Super Reasoner or vLLM-Omni endpoint."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("reasoner", "omni"), required=True)
    parser.add_argument("--base-url", default=os.getenv("COSMOS_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--prompt", default="A small red ball rolls across a wooden table and stops.")
    parser.add_argument("--output", default="cosmos3-super-smoke.mp4")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    headers = {}
    if token := os.getenv("COSMOS_API_KEY"):
        headers["Authorization"] = f"Bearer {token}"

    models = requests.get(f"{base_url}/v1/models", headers=headers, timeout=30)
    models.raise_for_status()
    print("models:", models.text[:1000])

    if args.backend == "reasoner":
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "model": "nvidia/Cosmos3-Super",
                "messages": [{"role": "user", "content": args.prompt}],
                "max_tokens": 256,
                "temperature": 0,
            },
            timeout=300,
        )
        response.raise_for_status()
        print(response.json()["choices"][0]["message"]["content"])
        return

    fields = {
        "model": "nvidia/Cosmos3-Super",
        "prompt": args.prompt,
        "size": "1280x720",
        "num_frames": "189",
        "fps": "24",
        "num_inference_steps": "35",
        "guidance_scale": "6.0",
        "max_sequence_length": "4096",
        "flow_shift": "10.0",
        "seed": "17",
        "extra_params": json.dumps(
            {"use_resolution_template": False, "use_duration_template": False, "guardrails": True}
        ),
    }
    response = requests.post(
        f"{base_url}/v1/videos/sync",
        headers={**headers, "Accept": "video/mp4"},
        data=fields,
        timeout=1800,
    )
    response.raise_for_status()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    if len(response.content) < 1024 or b"ftyp" not in response.content[:32]:
        raise RuntimeError(f"Response saved to {output}, but it does not look like an MP4")
    print(f"saved {len(response.content)} bytes to {output}")


if __name__ == "__main__":
    main()
