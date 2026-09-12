"""A mini-swe-agent model that replays a recorded prefix, then goes live.

Runs inside the task container, in mini-swe-agent's own Python (it is
installed next to mini-swe-agent by ``dswe.agent``). Selected with::

    -c model.model_class=dswe.replay_model.ReplayModel
    -c model.replay_path=/tmp/dswe-prefix.json -c model.replay_steps=40

For the first ``replay_steps`` queries it returns the recorded assistant
message instead of calling the API. mini-swe-agent then executes those
commands for real, so the container reaches the state the recorded agent was
in. From then on it is an ordinary LitellmModel.

``replay_observations`` decides what the history holds for replayed steps:

* ``recorded`` (default) — the recorded tool output, so the live model is sent
  a history identical to the recorded one. This isolates the model/provider.
* ``live`` — what our commands actually printed. Use this to see whether the
  recorded run's environment and ours agree.

Either way, every replayed step notes whether our output matched the
recording, and a summary is saved in the trajectory under ``info.replay``.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Literal

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig

from dswe import atif


def difference_kind(ours: str, recorded: str) -> str:
    """How a replayed command's output differs from the recording's."""
    # Pointer addresses (0xc000123abc) and git hashes change run to run too.
    digits = re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-f]{7,40}\b|\d+")
    a, b = digits.sub("#", ours), digits.sub("#", recorded)
    if a == b:
        return "numbers"
    if sorted(a.replace("\\n", "\n").splitlines()) == sorted(b.replace("\\n", "\n").splitlines()):
        return "order"
    return "content"


class ReplayModelConfig(LitellmModelConfig):
    replay_path: str
    replay_steps: int = -1
    """Agent steps to replay; -1 replays the whole recording."""
    replay_observations: Literal["recorded", "live"] = "recorded"


class ReplayModel(LitellmModel):
    def __init__(self, **kwargs: Any):
        super().__init__(config_class=ReplayModelConfig, **kwargs)
        self._traj = json.loads(Path(self.config.replay_path).read_text())
        # A step without tool calls can't be executed; mini-swe-agent never
        # keeps one in its history, so neither does the replay.
        steps = [s for s in atif.agent_steps(self._traj) if s.get("tool_calls")]
        n = len(steps) if self.config.replay_steps < 0 else min(self.config.replay_steps, len(steps))
        self._prefix = steps[:n]
        self._served = 0
        self._mismatched: list[int] = []
        self._kinds: dict[str, list[int]] = {}
        self._prompt_checked = False
        self._prompt_matches: bool | None = None
        print(f"[replay] replaying {n} of {len(steps)} recorded steps, observations={self.config.replay_observations}", flush=True)

    def _check_prompt(self, messages: list[dict]) -> None:
        recorded = atif.prompt_messages(self._traj)
        ours = [{"role": m["role"], "content": m["content"]} for m in messages[:2]]
        self._prompt_matches = ours == recorded
        if not self._prompt_matches:
            for mine, theirs in zip(ours, recorded):
                if mine != theirs:
                    a, b = str(mine["content"]).splitlines(), str(theirs["content"]).splitlines()
                    diff = [f"  ours: {x!r}\n  recorded: {y!r}" for x, y in zip(a, b) if x != y][:3]
                    print(f"[replay] {mine['role']} prompt differs from the recording:\n" + "\n".join(diff), flush=True)
        self._prompt_checked = True

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        if not self._prompt_checked:
            self._check_prompt(messages)
        if self._served < len(self._prefix):
            index = self._served
            self._served += 1
            step = self._prefix[index]
            message = atif.assistant_message(step)
            message["extra"] = {
                "actions": atif.actions(step),
                "cost": 0.0,
                "timestamp": time.time(),
                "replayed_step": index,
            }
            return message
        if self._served == len(self._prefix):
            self._served += 1
            print(f"[replay] handing off to {self.config.model_name} after {len(self._prefix)} replayed steps", flush=True)
        return super().query(messages, **kwargs)

    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None) -> list[dict]:
        rendered = super().format_observation_messages(message, outputs, template_vars)
        index = message.get("extra", {}).get("replayed_step")
        if index is None:
            return rendered
        recorded, followups = atif.observation_contents(self._prefix[index])
        mismatch = False
        for msg, rec in zip(rendered, recorded):
            matches = msg["content"] == rec
            mismatch = mismatch or not matches
            msg["extra"]["replay"] = {"step": index, "matches_recording": matches}
            if not matches:
                kind = difference_kind(msg["content"], rec)
                msg["extra"]["replay"]["difference"] = kind
                self._kinds.setdefault(kind, []).append(index)
            if self.config.replay_observations == "recorded":
                if not matches:
                    msg["extra"]["replay"]["live_content"] = msg["content"]
                msg["content"] = rec
            elif not matches:
                msg["extra"]["replay"]["recorded_content"] = rec
        if mismatch:
            self._mismatched.append(index)
            print(f"[replay] step {index}: our output differs from the recording", flush=True)
        if self.config.replay_observations == "recorded":
            # Format-error prompts the recorded agent was sent before its next step.
            rendered += [{"role": "user", "content": c, "extra": {"replay": {"step": index, "followup": True}}} for c in followups]
        return rendered

    def serialize(self) -> dict:
        data = super().serialize()
        data["info"]["replay"] = {
            "path": self.config.replay_path,
            "steps_replayed": min(self._served, len(self._prefix)),
            "steps_requested": len(self._prefix),
            "observations": self.config.replay_observations,
            "prompt_matches_recording": self._prompt_matches,
            "steps_with_different_output": self._mismatched,
            # step indices by kind: "numbers" (timings, sizes, dates' digits),
            # "order" (same lines, e.g. find over a different filesystem), "content"
            "differences": {kind: sorted(set(steps)) for kind, steps in self._kinds.items()},
        }
        return data
