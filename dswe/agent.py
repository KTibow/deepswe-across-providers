"""pier's mini-swe-agent, able to start from a recorded trajectory.

    PYTHONPATH=. pier run -p deep-swe/tasks/<task> \
        --agent-import-path dswe.agent:ProviderMiniSweAgent --model openai/glm-5.3 ... \
        [--ak prefix=/abs/trajectory.json --ak prefix_steps=40 --ak prefix_observations=recorded]

Without ``prefix`` the command it runs is exactly pier's own. With it, the
first ``prefix_steps`` steps are replayed by ``dswe.replay_model`` and the model
takes over after that. ``prefix_steps=-1`` replays everything and never needs
the model, which is how a published rollout is reproduced end to end.
"""

from __future__ import annotations

import base64
import shlex
from pathlib import Path

from pier.agents.installed.mini_swe_agent import MiniSweAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep

PACKAGE = Path(__file__).resolve().parent
CONTAINER_PREFIX = "/tmp/dswe-prefix.json"


class ProviderMiniSweAgent(MiniSweAgent):
    def __init__(
        self,
        *args,
        prefix: str | None = None,
        prefix_steps: int | str = -1,
        prefix_observations: str = "recorded",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._prefix = Path(prefix).expanduser().resolve() if prefix else None
        if self._prefix and not self._prefix.is_file():
            raise FileNotFoundError(f"prefix trajectory not found: {self._prefix}")
        self._prefix_steps = int(prefix_steps)
        self._prefix_observations = prefix_observations

    def install_spec(self) -> AgentInstallSpec:
        """pier's install, plus the replay model next to mini-swe-agent.

        Installed unconditionally so that live and replayed runs share one
        derived image.
        """
        spec = super().install_spec()
        writes = []
        for name in ("__init__.py", "atif.py", "replay_model.py"):
            payload = base64.b64encode((PACKAGE / name).read_bytes()).decode()
            writes.append(f'echo {payload} | base64 -d > "$site/dswe/{name}"')
        run = "\n".join(
            [
                "set -euo pipefail",
                'source "$HOME/.local/bin/env"',
                'python_bin="$(head -n 1 "$(command -v mini-swe-agent)" | sed \'s/^#!//\')"',
                'site="$("$python_bin" -c \'import sysconfig; print(sysconfig.get_paths()["purelib"])\')"',
                'mkdir -p "$site/dswe"',
                *writes,
                '"$python_bin" -c "import dswe.replay_model"',
            ]
        )
        return AgentInstallSpec(
            agent_name=spec.agent_name,
            version=spec.version,
            steps=[*spec.steps, InstallStep(user="agent", run=run)],
            verification_command=spec.verification_command,
        )

    def _build_config_flags(self, *, custom_config_path: str | None = None) -> str:
        flags = super()._build_config_flags(custom_config_path=custom_config_path)
        if self._prefix:
            # Later -c specs win, so this overrides the model class set above.
            flags += (
                "-c model.model_class=dswe.replay_model.ReplayModel "
                f"-c model.replay_path={CONTAINER_PREFIX} "
                f"-c model.replay_steps={self._prefix_steps} "
                f"-c model.replay_observations={shlex.quote(self._prefix_observations)} "
            )
        return flags

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if self._prefix:
            await environment.upload_file(source_path=self._prefix, target_path=CONTAINER_PREFIX)
            await environment.exec(command=f"chmod 644 {CONTAINER_PREFIX}", user="root")
        await super().run(instruction, environment, context)
