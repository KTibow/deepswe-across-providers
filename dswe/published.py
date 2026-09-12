"""DeepSWE's published data: rollout tables and per-rollout artifacts, cached.

Where this lives and its quirks are in NOTES.md.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SITE = "https://deepswe.datacurve.ai"
CDN = "https://d3ujjcmjq6o8v6.cloudfront.net"
RELEASE = "v1.1"
CACHE = Path(__file__).resolve().parent.parent / ".cache"
UA = {"User-Agent": "deepswe-across-providers/1.0"}


def _get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def table(name: str, release: str = RELEASE) -> list[dict[str, Any]]:
    """`trials`, `tasks` or `leaderboard-live`, as a list of rows."""
    local = CACHE / f"{release}-{name}.json"
    if not local.exists():
        print(f"fetching {name} table ({release})", file=sys.stderr)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(_get(f"{SITE}/artifacts/{release}/{name}.json"))
    return json.loads(local.read_text())["rows"]


def artifact_url(trial: str, path: str, release: str = RELEASE) -> str:
    return f"{CDN}/{release}/trial-artifacts/{trial}/{path}"


def trajectory(ref: str, release: str = RELEASE) -> dict[str, Any]:
    """An ATIF trajectory from a local path, a URL, or a published trial name."""
    path = Path(ref)
    if path.is_file():
        return json.loads(path.read_text())
    if ref.startswith(("http://", "https://")):
        return json.loads(_get(ref))
    local = CACHE / "trajectories" / release / f"{ref}.json"
    if not local.exists():
        try:
            body = _get(artifact_url(ref, "agent/trajectory.json", release))
        except urllib.error.HTTPError as e:
            # ~2.4% of published artifacts return 403 (deep-swe#59).
            raise SystemExit(f"no published trajectory for {ref}: HTTP {e.code}") from e
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(body)
    return json.loads(local.read_text())
