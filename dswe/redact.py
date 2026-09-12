"""Replace provider keys with a placeholder in files about to be uploaded.

    python3 -m dswe.redact results out

mini-swe-agent runs the agent's bash commands with the API key in their
environment, so a model that runs `env` while debugging prints the key into
its trajectory. GitHub masks secrets in job logs but not in artifacts, and a
public repo's artifacts can be downloaded by any signed-in GitHub user. This
replaces every provider key named in providers.toml (read from the
environment) wherever it appears under the given directories.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

from dswe.profiles import DEFAULT


def keys_from_env(path: Path = DEFAULT) -> dict[str, str]:
    config = tomllib.loads(path.read_text())
    names = sorted({p["key_env"] for p in config["providers"].values()})
    # Short values would match ordinary text; no real key is under 8 characters.
    return {name: os.environ[name] for name in names if len(os.environ.get(name, "")) >= 8}


def redact(dirs: list[str | Path], keys: dict[str, str]) -> dict[str, int]:
    """Replace each key's value in place. Returns how many times each was found."""
    counts = {name: 0 for name in keys}
    for directory in dirs:
        root = Path(directory)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            data = new = path.read_bytes()
            for name, value in keys.items():
                found = new.count(value.encode())
                if found:
                    counts[name] += found
                    new = new.replace(value.encode(), f"<{name} redacted>".encode())
            if new != data:
                path.write_bytes(new)
    return counts


def main() -> None:
    keys = keys_from_env()
    if not keys:
        print("no provider keys in the environment; nothing to redact")
        return
    for name, found in redact(sys.argv[1:], keys).items():
        if found:
            # A GitHub annotation, so it shows on the run page.
            print(f"::warning::{name} appeared {found} time(s) in files about to be uploaded; replaced it. "
                  "The agent probably printed its environment.")
        else:
            print(f"{name}: not found in any uploaded file")


if __name__ == "__main__":
    main()
