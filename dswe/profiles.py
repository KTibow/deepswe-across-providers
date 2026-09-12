"""Resolve `provider:model` against providers.toml.

    python3 -m dswe.profiles crof:glm-5.3            # show the resolved profile
    python3 -m dswe.profiles crof:glm-5.3 --pier     # pier flags, one per line
"""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT = Path(__file__).resolve().parent.parent / "providers.toml"


@dataclass
class Profile:
    spec: str
    provider: str
    model: str
    api_base: str
    key_env: str
    model_id: str
    litellm_model: str
    model_class: str
    reference_config: str | None = None
    mini_swe_agent_version: str | None = None
    set_cache_control: str | None = None
    request: dict[str, Any] = field(default_factory=dict)

    @property
    def litellm_key_env(self) -> str:
        """The variable litellm reads the key from for this model id."""
        return "OPENROUTER_API_KEY" if self.litellm_model.startswith("openrouter/") else "OPENAI_API_KEY"

    def pier_args(self) -> list[str]:
        """Flags for `pier run` that reproduce the published agent settings.

        The key is passed as a `${VAR}` template that pier resolves from its own
        environment, so its value never reaches a command line or a config.
        """
        args = [
            "--agent-import-path", "dswe.agent:ProviderMiniSweAgent",
            "--model", self.litellm_model,
            "--ae", f"{self.litellm_key_env}=${{{self.key_env}}}",
            "--ak", f"model_class={self.model_class}",
            "--ak", "model_kwargs=" + json.dumps({"extra_body": self.request}),
        ]
        if not self.litellm_model.startswith("openrouter/"):
            # Also what pier's egress proxy allowlists.
            args += ["--ae", f"OPENAI_API_BASE={self.api_base}", "--ae", f"OPENAI_BASE_URL={self.api_base}"]
        if self.mini_swe_agent_version:
            args += ["--ak", f"version={self.mini_swe_agent_version}"]
        if self.set_cache_control:
            args += ["--ak", f"set_cache_control={self.set_cache_control}"]
        return args


def load(spec: str, path: Path = DEFAULT) -> Profile:
    config = tomllib.loads(path.read_text())
    if ":" not in spec:
        raise SystemExit(f"profile must be provider:model, got {spec!r}")
    provider, model = spec.split(":", 1)
    if provider not in config["providers"]:
        raise SystemExit(f"unknown provider {provider!r}; have {sorted(config['providers'])}")
    if model not in config["models"]:
        raise SystemExit(f"unknown model {model!r}; have {sorted(config['models'])}")
    p, m = config["providers"][provider], config["models"][model]
    model_id = m["ids"].get(provider)
    if not model_id:
        raise SystemExit(f"{model} has no id on {provider}")
    request = dict(m.get("request", {}))
    request.update(m.get("provider_request", {}).get(provider, {}))
    return Profile(
        spec=spec,
        provider=provider,
        model=model,
        api_base=p["api_base"],
        key_env=p["key_env"],
        model_id=model_id,
        litellm_model=p["litellm_prefix"] + model_id,
        model_class=m.get("model_class", "litellm"),
        reference_config=m.get("reference_config"),
        mini_swe_agent_version=m.get("mini_swe_agent_version"),
        set_cache_control=m.get("set_cache_control"),
        request=request,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--pier", action="store_true", help="print pier flags, one per line")
    args = ap.parse_args()
    profile = load(args.spec)
    if args.pier:
        print("\n".join(profile.pier_args()))
    else:
        print(json.dumps(asdict(profile), indent=2))


if __name__ == "__main__":
    main()
