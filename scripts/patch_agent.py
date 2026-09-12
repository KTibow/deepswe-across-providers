"""A pier agent that replays a recorded submission instead of calling a model.

DeepSWE publishes the ``model.patch`` every rollout submitted. This agent drops
one of those patches into the task environment and commits it, which is exactly
what a real agent's turn ends up looking like to the harness: the
``[[verifier.collect]]`` hook diffs the agent's commits against the task's base
commit, and the verifier grades that diff in a pristine container.

Used as::

    PYTHONPATH=scripts pier run -p <task> \
        --agent-import-path patch_agent:PatchAgent \
        --ak patch_path=/abs/path/to/model.patch
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from pier.agents.base import BaseAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext

STAGING_DIR = "/replay"
APPLY_SCRIPT = r"""#!/bin/bash
# Apply a recorded submission and commit it, the way a finishing agent would.
set -uo pipefail
cd /app || { echo "[replay] no /app"; exit 90; }
git config --global --add safe.directory /app || true

PATCH=/replay/submission.patch
if [ ! -s "$PATCH" ]; then
  echo "[replay] submission patch is empty; committing nothing"
  exit 0
fi
echo "[replay] patch: $(wc -c < "$PATCH") bytes, $(grep -c '^diff --git' "$PATCH") file(s)"
echo "[replay] head: $(git rev-parse HEAD)"

applied=""
while IFS= read -r mode; do
  [ -n "$mode" ] || continue
  # shellcheck disable=SC2086
  if git apply $mode "$PATCH" 2>/replay/apply.err; then
    applied="$mode"
    break
  fi
  echo "[replay] 'git apply $mode' failed: $(tail -3 /replay/apply.err)"
done <<'MODES'
--whitespace=nowarn --binary
--whitespace=nowarn --binary -3
--whitespace=nowarn --binary --reject
MODES

if [ -z "$applied" ]; then
  echo "[replay] PATCH DID NOT APPLY"
  exit 91
fi
echo "[replay] applied with: git apply $applied"

git checkout -b replay/submission 2>/dev/null || true
git add -A
git -c user.name=replay -c user.email=replay@local \
    commit -q --no-verify -m "Replayed recorded submission" || true
echo "[replay] committed: $(git rev-parse HEAD)"
git --no-pager diff --stat HEAD~1 HEAD 2>/dev/null | tail -5
exit 0
"""


class PatchAgent(BaseAgent):
    SUPPORTS_ATIF = False
    SUPPORTS_WINDOWS = False

    @staticmethod
    def name() -> str:
        return "patch-replay"

    def version(self) -> str:
        return "1.0.0"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        patch_path: str | None = None,
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        if not patch_path:
            raise ValueError("PatchAgent needs --ak patch_path=<path to model.patch>")
        self._patch_path = Path(patch_path).expanduser().resolve()
        if not self._patch_path.is_file():
            raise FileNotFoundError(f"patch not found: {self._patch_path}")

    async def setup(self, environment: BaseEnvironment) -> None:
        return

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        staging = Path(tempfile.mkdtemp(prefix="pier-patch-replay-"))
        try:
            shutil.copyfile(self._patch_path, staging / "submission.patch")
            (staging / "apply.sh").write_text(APPLY_SCRIPT)

            await environment.upload_dir(source_dir=staging, target_dir=STAGING_DIR)
            await environment.exec(
                command=f"chmod +x {STAGING_DIR}/apply.sh", user="root"
            )
            log_path = f"{environment.env_paths.agent_dir}/replay.txt"
            result = await environment.exec(
                command=f"(bash {STAGING_DIR}/apply.sh) > {log_path} 2>&1",
                user="root",
            )
            if result.return_code != 0:
                self.logger.error(
                    f"patch replay exited {result.return_code} for {self._patch_path.name}"
                )
                (self.logs_dir / "exit-code.txt").write_text(str(result.return_code))
        finally:
            shutil.rmtree(staging, ignore_errors=True)
