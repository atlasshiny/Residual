"""
Launch NVIDIA Cosmos3-Super via vLLM Reasoner or vLLM-Omni Generator.
Reasoner = text/image understanding
vLLM Omni = video/image generation
"""

from __future__ import annotations

import argparse
import os
import shutil


MODEL_ID = "nvidia/Cosmos3-Super"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("reasoner", "omni"), required=True)
    args = parser.parse_args()

    cli = shutil.which("vllm")
    if not cli:
        parser.error("vllm executable not found; activate the configured serving venv")

    port = os.getenv("COSMOS_PORT", "8000")
    host = os.getenv("COSMOS_HOST", "127.0.0.1")
    model = os.getenv("COSMOS_MODEL_PATH", MODEL_ID)

    common = [
        "serve",
        model,
        "--served-model-name",
        MODEL_ID,
        "--host",
        host,
        "--port",
        port,
    ]
    if api_key := os.getenv("COSMOS_API_KEY"):
        common.extend(["--api-key", api_key])
    elif host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("COSMOS_API_KEY is required when binding beyond loopback")
    if args.backend == "reasoner":
        # NVIDIA's Cosmos3 Reasoner recipe uses the unified HF checkpoint with
        # an architecture override; TP=8 gives one model replica across all
        # GPUs allocated to this job. Override COSMOS_TP_SIZE to try TP=4.
        tp_size = os.getenv("COSMOS_TP_SIZE", "8")
        command = [
            cli,
            *common,
            "--hf-overrides",
            '{"architectures":["Cosmos3ReasonerForConditionalGeneration"]}',
            "--tensor-parallel-size",
            tp_size,
            "--mm-encoder-tp-mode",
            "data",
            "--async-scheduling",
            "--allowed-local-media-path",
            os.getenv("COSMOS_MEDIA_ROOT", "/"),
            "--media-io-kwargs",
            '{"video":{"num_frames":-1}}',
            "--max-model-len",
            os.getenv("COSMOS_MAX_MODEL_LEN", "32768"),
        ]
    else:
        # Official 8-GPU Cosmos3-Super Generator topology: CFG=2, Ulysses=4,
        # HSDP shard=8. Keep guardrails enabled (the default).
        command = [
            cli,
            *common,
            "--omni",
            "--cfg-parallel-size",
            "2",
            "--ulysses-degree",
            "4",
            "--use-hsdp",
            "--hsdp-shard-size",
            "8",
            "--init-timeout",
            os.getenv("COSMOS_INIT_TIMEOUT", "1800"),
        ]

    print("Starting Cosmos3-Super", args.backend, flush=True)
    display_command = command.copy()
    if "--api-key" in display_command:
        display_command[display_command.index("--api-key") + 1] = "<redacted>"
    print("Command:", " ".join(display_command), flush=True)
    os.execvpe(command[0], command, os.environ.copy())


if __name__ == "__main__":
    main()
