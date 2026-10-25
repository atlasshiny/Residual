"""vLLM-Omni client for the Cosmos 3 workers.

Talks to a running ``vllm serve ... --omni`` endpoint (OpenAI-compatible) over
HTTP using only the standard library, so importing a worker never requires
``requests`` or ``torch``.  Field names mirror ``hpc/cosmos3/smoke.py``.

Endpoints (``[verify]`` against the Cosmos-Framework docs):

* ``GET  /v1/models``                    -- availability probe
* ``POST /v1/videos/sync``               -- video generation (transfer, forward)
* ``POST /v1/videos/inverse_dynamics``   -- video -> action rows; override the
                                            path with ``COSMOS_INVERSE_PATH``

Configuration is environment-driven so a SLURM job can point a worker at the
server without code changes:

===========================  ==========================================
``COSMOS_OMNI_URL``          base URL of the vLLM-Omni server
``COSMOS_BASE_URL``          fallback base URL (used by ``smoke.py``)
``COSMOS_API_KEY``           bearer token, if the server sets one
``COSMOS_MODEL``             served model name
``COSMOS_VIDEO_PATH``        override the generation path
``COSMOS_INVERSE_PATH``      override the inverse-dynamics path
===========================  ==========================================

When nothing is configured or reachable, :meth:`OmniClient.available` returns
``False`` and the workers fall back to a deterministic synthetic run so the file
contract can be exercised offline.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common  # type: ignore
else:
    from . import common

DEFAULT_PROBE_PATH = "/v1/models"
DEFAULT_VIDEO_PATH = "/v1/videos/sync"
DEFAULT_INVERSE_PATH = "/v1/videos/inverse_dynamics"


class OmniUnavailable(RuntimeError):
    """The vLLM-Omni endpoint is not configured or not reachable."""


class OmniClient:
    """Thin HTTP client for the vLLM-Omni generator."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        probe_timeout: float = 5.0,
        video_timeout: float = 1800.0,
        inverse_timeout: float = 600.0,
    ) -> None:
        resolved = (
            base_url
            or os.getenv("COSMOS_OMNI_URL")
            or os.getenv("COSMOS_BASE_URL")
            or ""
        )
        self.base_url = resolved.strip().rstrip("/")
        self.api_key = api_key or os.getenv("COSMOS_API_KEY")
        self.model = model or os.getenv("COSMOS_MODEL")
        self.probe_timeout = float(probe_timeout)
        self.video_timeout = float(video_timeout)
        self.inverse_timeout = float(inverse_timeout)

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    # -- transport ---------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
        accept: str = "application/json",
        timeout: float,
    ) -> bytes:
        if not self.configured:
            raise OmniUnavailable(
                "COSMOS_OMNI_URL/COSMOS_BASE_URL is not set; no vLLM-Omni server"
            )
        url = f"{self.base_url}{path}"
        request = urllib.request.Request(url, data=body, method=method)
        request.add_header("Accept", accept)
        if content_type:
            request.add_header("Content-Type", content_type)
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OmniUnavailable(f"{method} {url} failed: {exc}") from exc

    def _get(self, path: str, *, timeout: float) -> bytes:
        return self._request("GET", path, timeout=timeout)

    def _post_form(
        self,
        path: str,
        fields: Mapping[str, Any],
        *,
        timeout: float,
        accept: str = "application/json",
    ) -> bytes:
        payload = {key: value for key, value in fields.items() if value is not None}
        body = urllib.parse.urlencode(payload).encode("utf-8")
        return self._request(
            "POST",
            path,
            body=body,
            content_type="application/x-www-form-urlencoded",
            accept=accept,
            timeout=timeout,
        )

    # -- capabilities ------------------------------------------------------

    def available(self) -> bool:
        """True when a server answers ``GET /v1/models`` within the probe timeout."""

        if not self.configured:
            return False
        try:
            self._get(DEFAULT_PROBE_PATH, timeout=self.probe_timeout)
            return True
        except OmniUnavailable:
            return False

    # -- generation --------------------------------------------------------

    def generate_video(self, fields: Mapping[str, Any]) -> bytes:
        """POST generation fields and return the raw MP4 bytes."""

        path = os.getenv("COSMOS_VIDEO_PATH", DEFAULT_VIDEO_PATH)
        data = self._post_form(
            path, fields, timeout=self.video_timeout, accept="video/mp4"
        )
        if not data:
            raise OmniUnavailable("vLLM-Omni returned an empty video response")
        return data

    def generate_video_to(self, dest: Path, fields: Mapping[str, Any]) -> int:
        """Write the generated video to ``dest`` and return its byte size."""

        data = self.generate_video(fields)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return len(data)

    def infer_actions(
        self,
        video: Path,
        *,
        model: str | None = None,
        domain_name: str = common.DEFAULT_DOMAIN_NAME,
        fps: Any = None,
        frames: Any = None,
    ) -> list[list[float]]:
        """Recover a 9D action sequence from a video clip (job B)."""

        fields: dict[str, Any] = {
            "model": model or self.model or common.DEFAULT_MODEL,
            "video": str(video),
            "domain_name": domain_name,
        }
        if fps is not None:
            fields["fps"] = common.format_number(fps)
        if frames is not None:
            fields["frames"] = str(int(frames))
        path = os.getenv("COSMOS_INVERSE_PATH", DEFAULT_INVERSE_PATH)
        raw = self._post_form(path, fields, timeout=self.inverse_timeout)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise OmniUnavailable(
                f"invalid JSON from inverse-dynamics endpoint: {exc}"
            ) from exc
        return common.coerce_actions(data)
