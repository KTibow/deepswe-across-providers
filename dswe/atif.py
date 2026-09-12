"""Rebuild mini-swe-agent's message history from an ATIF trajectory.

pier converts every mini-swe-agent run into ATIF (``agent/trajectory.json``),
and that is the only full trajectory DeepSWE publishes. The conversion keeps
enough to reconstruct what the model was sent at any step:

* step 0 is the system message, step 1 the task prompt;
* each agent step carries the assistant text, reasoning and tool calls;
* its observation holds one rendered tool result per call, followed by any
  format-error prompts mini-swe-agent sent before the next agent step.

What it loses: assistant messages that failed to parse (mini-swe-agent drops
those from its own history too, so nothing is lost for replay), and whether
the assistant content was ``None`` or ``""``.

Stdlib only — this file is also copied into the task container.
"""

from __future__ import annotations

import json
from typing import Any


def agent_steps(traj: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in traj["steps"] if s.get("source") == "agent"]


def prompt_messages(traj: dict[str, Any]) -> list[dict[str, Any]]:
    steps = traj["steps"]
    system = next(s for s in steps if s.get("source") == "system")
    user = next(s for s in steps if s.get("source") == "user")
    return [
        {"role": "system", "content": system["message"]},
        {"role": "user", "content": user["message"]},
    ]


def actions(step: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"command": str(tc["arguments"].get("command", "")), "tool_call_id": tc["tool_call_id"]}
        for tc in step.get("tool_calls") or []
    ]


def assistant_message(step: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": step.get("message") or "",
        "tool_calls": [
            {
                "id": tc["tool_call_id"],
                "type": "function",
                "function": {"name": tc["function_name"], "arguments": json.dumps(tc["arguments"])},
            }
            for tc in step.get("tool_calls") or []
        ],
    }
    if step.get("reasoning_content"):
        message["reasoning_content"] = step["reasoning_content"]
    return message


def observation_contents(step: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(one rendered result per tool call, trailing format-error prompts)."""
    results = [r.get("content", "") for r in (step.get("observation") or {}).get("results") or []]
    n = len(step.get("tool_calls") or [])
    return results[:n], results[n:]


def observation_messages(step: dict[str, Any]) -> list[dict[str, Any]]:
    tool_results, followups = observation_contents(step)
    messages = [
        {"role": "tool", "tool_call_id": act["tool_call_id"], "content": content}
        for act, content in zip(actions(step), tool_results)
    ]
    messages += [{"role": "user", "content": content} for content in followups]
    return messages


def history(traj: dict[str, Any], n_steps: int) -> list[dict[str, Any]]:
    """Messages the model was sent before agent step ``n_steps`` (0-based)."""
    messages = prompt_messages(traj)
    for step in agent_steps(traj)[:n_steps]:
        messages.append(assistant_message(step))
        messages.extend(observation_messages(step))
    return messages
