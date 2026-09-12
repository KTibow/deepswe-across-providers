"""Ask a provider for one step of a recorded rollout, without a container.

Rebuilds the exact history mini-swe-agent sent before agent step K of a
recorded trajectory, sends it to the provider under test the way litellm
would, and puts the answer next to what the recorded model did:

    python3 -m dswe.probe crof:glm-5.3 abs-module-cache-flags__KeBKjxc --steps 8,40

What to read in the output:

* `prompt` vs `recorded` — the same history through the same tokenizer should
  cost about the same prompt tokens. A big gap means the provider's chat
  template renders the history differently (dropped reasoning, reformatted
  tool results), which is invisible from the outside otherwise.
* `format` — whether the reply is something mini-swe-agent can execute.
* `c1` — C1 control characters in the reply, the signature of UTF-8 decoded
  as Latin-1 somewhere in the serving stack.
* latency and output tokens — how long a 100-step rollout will take.

Every raw response is kept in the JSONL output.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dswe import atif, profiles, published

# mini-swe-agent 2.x's only tool (minisweagent/models/utils/actions_toolcall.py).
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}
C1 = re.compile("[\u0080-\u009f]")


def with_cache_control(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What mini-swe-agent's set_cache_control="default_end" plus litellm put on the wire."""
    messages = copy.deepcopy(messages)
    last = messages[-1]
    if last["role"] == "tool":
        last["content"] = [{"type": "text", "text": last["content"]}]
        last["cache_control"] = {"type": "ephemeral"}
    else:
        last["content"] = [{"type": "text", "text": last["content"], "cache_control": {"type": "ephemeral"}}]
    return messages


def request_body(profile: profiles.Profile, messages: list[dict[str, Any]], stream: bool) -> dict[str, Any]:
    if profile.set_cache_control == "default_end":
        messages = with_cache_control(messages)
    body = {"model": profile.model_id, "messages": messages, "tools": [BASH_TOOL], **profile.request}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def post(profile: profiles.Profile, key: str, body: dict[str, Any], timeout: int) -> tuple[dict[str, Any], float | None]:
    req = urllib.request.Request(
        profile.api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "deepswe-across-providers/1.0",
        },
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if not body.get("stream"):
            return json.load(resp), None
        return accumulate_stream(resp, start)


def accumulate_stream(resp: Any, start: float) -> tuple[dict[str, Any], float | None]:
    """Fold SSE chunks back into one chat.completion. Returns (response, ttft)."""
    ttft = None
    content, reasoning = [], []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish, usage, rid = None, None, None
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        rid = rid or chunk.get("id")
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if ttft is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")):
                ttft = time.monotonic() - start
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            for tc in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(tc.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                slot["id"] = tc.get("id") or slot["id"]
                fn = tc.get("function") or {}
                slot["function"]["name"] += fn.get("name") or ""
                slot["function"]["arguments"] += fn.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    message = {"role": "assistant", "content": "".join(content) or None, "reasoning_content": "".join(reasoning) or None,
               "tool_calls": [tool_calls[i] for i in sorted(tool_calls)]}
    return {"id": rid, "choices": [{"message": message, "finish_reason": finish}], "usage": usage}, ttft


def commands_from(message: dict[str, Any]) -> tuple[list[str], str | None]:
    """The bash commands mini-swe-agent would run, or the format error it would raise."""
    calls = message.get("tool_calls") or []
    if not calls:
        return [], "no tool call"
    commands = []
    for call in calls:
        fn = call.get("function") or {}
        if fn.get("name") != "bash":
            return commands, f"unknown tool {fn.get('name')!r}"
        try:
            args = json.loads(fn.get("arguments") or "")
        except json.JSONDecodeError:
            return commands, "arguments are not JSON"
        if not isinstance(args, dict) or "command" not in args:
            return commands, "no command argument"
        commands.append(str(args["command"]))
    return commands, None


def probe(profile: profiles.Profile, key: str, traj: dict[str, Any], step: int, stream: bool, timeout: int) -> dict[str, Any]:
    steps = atif.agent_steps(traj)
    recorded = steps[step] if step < len(steps) else None
    body = request_body(profile, atif.history(traj, step), stream)
    record: dict[str, Any] = {"profile": profile.spec, "step": step, "stream": stream, "started": time.time()}
    start = time.monotonic()
    try:
        response, ttft = post(profile, key, body, timeout)
    except urllib.error.HTTPError as e:
        record.update(latency_s=time.monotonic() - start, error=f"HTTP {e.code}: {e.read()[:2000].decode(errors='replace')}")
        return record
    except Exception as e:  # noqa: BLE001 — a probe records failures, it doesn't raise them
        record.update(latency_s=time.monotonic() - start, error=f"{type(e).__name__}: {e}")
        return record
    record["latency_s"] = time.monotonic() - start
    record["ttft_s"] = ttft
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = response.get("usage") or {}
    commands, format_error = commands_from(message)
    text = (message.get("content") or "") + (message.get("reasoning_content") or "") + "".join(commands)
    record.update(
        finish_reason=choice.get("finish_reason"),
        usage=usage,
        commands=commands,
        format_error=format_error,
        c1_chars=len(C1.findall(text)),
        response=response,
    )
    if recorded:
        rec_metrics = recorded.get("metrics") or {}
        rec_commands = [a["command"] for a in atif.actions(recorded)]
        record["recorded"] = {
            "prompt_tokens": rec_metrics.get("prompt_tokens"),
            "completion_tokens": rec_metrics.get("completion_tokens"),
            "reasoning_tokens": ((rec_metrics.get("extra") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "commands": rec_commands,
        }
        record["same_commands"] = commands == rec_commands
    return record


def summary_line(r: dict[str, Any]) -> str:
    if r.get("error"):
        return f"step {r['step']:>3}: {r['latency_s']:.1f}s ERROR {r['error'][:300]}"
    usage = r.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    rec = r.get("recorded") or {}
    ratio = f" x{prompt / rec['prompt_tokens']:.3f}" if prompt and rec.get("prompt_tokens") else ""
    reasoning = usage.get("reasoning_tokens") or (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    ttft = f" ttft {r['ttft_s']:.1f}s" if r.get("ttft_s") is not None else ""
    return (
        f"step {r['step']:>3}: {r['latency_s']:.1f}s{ttft}  prompt {prompt} (recorded {rec.get('prompt_tokens')}{ratio})  "
        f"out {usage.get('completion_tokens')} (reasoning {reasoning}; recorded {rec.get('completion_tokens')})  "
        f"finish={r.get('finish_reason')}  format={r.get('format_error') or 'ok'}  "
        f"same_commands={r.get('same_commands')}  c1={r.get('c1_chars')}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", help="provider:model from providers.toml")
    ap.add_argument("trajectory", help="published trial name, ATIF path, or URL")
    ap.add_argument("--steps", default="0", help="agent steps to probe, comma separated (0-based)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--stream", action="store_true", help="stream, to measure time to first token")
    ap.add_argument("--key-file", type=Path, help="read the key from a file instead of the profile's env var")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", type=Path, default=Path("out/probe.jsonl"))
    args = ap.parse_args()

    profile = profiles.load(args.profile)
    key = args.key_file.read_text().strip() if args.key_file else os.environ.get(profile.key_env, "")
    if not key:
        sys.exit(f"set {profile.key_env} or pass --key-file")
    traj = published.trajectory(args.trajectory)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_steps = len(atif.agent_steps(traj))
    print(f"{args.trajectory}: {n_steps} agent steps, model {(traj.get('agent') or {}).get('model_name')}", file=sys.stderr)

    for step in [int(s) for s in args.steps.split(",") if s.strip()]:
        for _ in range(args.repeat):
            record = probe(profile, key, traj, step, args.stream, args.timeout)
            record["trajectory"] = args.trajectory
            with args.out.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            print(summary_line(record), flush=True)


if __name__ == "__main__":
    main()
