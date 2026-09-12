"""Turn one shard's pier job directory into compact records plus kept logs.

    python3 -m dswe.collect --jobs-dir jobs --status out/status.jsonl --shard 3 --out results

Writes ``<out>/shard-NNN.json`` and copies, per trial, the verifier's report
files and the agent's full trajectories into ``<out>/logs/<unit>/<trial>/`` —
the ATIF ``trajectory.json`` there is what a later run can resume from.

Besides the grade, each record carries what the agent did and how its model
calls went, read from mini-swe-agent's own trajectory and log: per-call
latency, tokens and finish reasons, format errors, empty replies, retries,
and C1 control characters (UTF-8 mis-decoded as Latin-1).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

VERIFIER_FILES = ("reward.json", "ctrf.json", "test-stdout.txt", "run.log")
AGENT_FILES = ("trajectory.json", "mini-swe-agent.trajectory.json", "mini-swe-agent.txt", "oracle.txt", "exit-code.txt")
STDOUT_TAIL = 6000
MAX_COPY_BYTES = 60_000_000
C1 = re.compile("[\u0080-\u009f]")
# tenacity's before_sleep_log line for a failed model call.
RETRY = re.compile(r"Retrying \S+ in [\d.]+ seconds as it raised (\w+)")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def duration(phase: Any) -> float | None:
    """pier's TimingInfo carries started_at/finished_at, not a duration."""
    if not isinstance(phase, dict):
        return None
    if phase.get("duration_sec") is not None:
        return phase["duration_sec"]
    start, end = phase.get("started_at"), phase.get("finished_at")
    if not start or not end:
        return None
    try:
        parse = lambda s: datetime.fromisoformat(str(s).replace("Z", "+00:00"))  # noqa: E731
        return (parse(end) - parse(start)).total_seconds()
    except ValueError:
        return None


def failed_tests(ctrf: Any) -> list[dict[str, str]]:
    """Failing test names from a CTRF report, tolerating shape drift."""
    tests: list[Any] = []
    stack = [ctrf]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if isinstance(node.get("tests"), list):
                tests = node["tests"]
                break
            stack.extend(node.values())
    return [
        {"name": str(t.get("name", "?")), "status": str(t.get("status", "?")), "message": str(t.get("message", ""))[:600]}
        for t in tests
        if isinstance(t, dict) and str(t.get("status", "")).lower() not in ("passed", "pass", "ok")
    ]


def call_metrics(response: Any) -> dict[str, Any]:
    """One model call, from the litellm response mini-swe-agent saved with it."""
    if not isinstance(response, dict):
        return {}
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = response.get("usage") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    tool_calls = message.get("tool_calls") or []
    arguments = "".join(str((tc.get("function") or {}).get("arguments") or "") for tc in tool_calls)
    return {
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or usage.get("reasoning_tokens"),
        "empty": not content.strip() and not reasoning.strip() and not tool_calls,
        "c1_chars": len(C1.findall(content + reasoning + arguments)),
    }


def agent_summary(agent_dir: Path) -> dict[str, Any]:
    """What the agent did and how its model calls went.

    Latency is the gap between a live assistant message and the message before
    it (the previous tool output): time spent waiting on the provider,
    retries included. It is left out after a format-error call, which has no
    timestamp of its own.
    """
    data = load_json(agent_dir / "mini-swe-agent.trajectory.json")
    if not isinstance(data, dict):
        return {}
    info = data.get("info") or {}

    calls: list[dict[str, Any]] = []
    latencies: list[float] = []
    last_ts: float | None = None
    after_format_error = False
    replayed = 0
    for message in data.get("messages") or []:
        extra = message.get("extra") or {}
        if message.get("role") == "assistant":
            if "replayed_step" in extra:
                replayed += 1
            else:
                calls.append(call_metrics(extra.get("response")))
                if last_ts is not None and extra.get("timestamp") and not after_format_error:
                    latencies.append(round(extra["timestamp"] - last_ts, 2))
            after_format_error = False
        elif extra.get("interrupt_type") == "FormatError":
            calls.append({**call_metrics(extra.get("response")), "format_error": True})
            after_format_error = True
        if extra.get("timestamp"):
            last_ts = extra["timestamp"]

    retries: dict[str, int] = {}
    log = agent_dir / "mini-swe-agent.txt"
    if log.exists():
        for match in RETRY.finditer(log.read_text(errors="replace")):
            retries[match.group(1)] = retries.get(match.group(1), 0) + 1

    def series(key: str) -> list[float]:
        return [c[key] for c in calls if isinstance(c.get(key), (int, float))]

    finish: dict[str, int] = {}
    for c in calls:
        finish[str(c.get("finish_reason"))] = finish.get(str(c.get("finish_reason")), 0) + 1
    return {
        "exit_status": info.get("exit_status") or None,
        "mini_swe_agent_version": info.get("mini_version"),
        "steps": replayed + sum(1 for c in calls if not c.get("format_error")),
        "replayed_steps": replayed,
        "replay": info.get("replay"),
        "inference": {
            "calls": len(calls),
            "format_errors": sum(1 for c in calls if c.get("format_error")),
            "retries": retries,
            "finish_reasons": finish,
            "empty_replies": sum(1 for c in calls if c.get("empty")),
            "c1_chars": sum(series("c1_chars")),
            "latency_s": latencies,
            "output_tokens": series("output_tokens"),
            "reasoning_tokens": series("reasoning_tokens"),
            "prompt_tokens": series("prompt_tokens"),
            "cached_tokens": series("cached_tokens"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-dir", type=Path, required=True)
    ap.add_argument("--status", type=Path, help="status.jsonl from run_shard.sh")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--runner", default="")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    status: dict[str, dict[str, Any]] = {}
    if args.status and args.status.exists():
        for line in args.status.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            status[entry["unit"]] = entry

    records: list[dict[str, Any]] = []
    seen_units: set[str] = set()
    # pier writes jobs/<job>/<trial>/result.json per trial, plus a job-level
    # result.json with no task_name, which is skipped.
    for result_path in sorted(args.jobs_dir.rglob("result.json")) if args.jobs_dir.exists() else []:
        data = load_json(result_path)
        if not isinstance(data, dict) or "task_name" not in data:
            continue
        trial_dir = result_path.parent
        unit = result_path.relative_to(args.jobs_dir).parts[0]
        seen_units.add(unit)
        verifier_dir, agent_dir = trial_dir / "verifier", trial_dir / "agent"

        rewards = (data.get("verifier_result") or {}).get("rewards") or load_json(verifier_dir / "reward.json") or {}
        reward = rewards.get("reward") if isinstance(rewards, dict) else None
        exc = data.get("exception_info") or {}
        if exc:
            state = "error"
        elif reward is None:
            state = "no-reward"
        else:
            state = "resolved" if float(reward) >= 1 else "unresolved"
        stdout = verifier_dir / "test-stdout.txt"
        agent_result = data.get("agent_result") or {}

        records.append({
            "unit": unit,
            "task": str(data["task_name"]).split("/")[-1],
            "trial": data.get("trial_name"),
            "shard": args.shard,
            "runner": args.runner,
            "state": state,
            "reward": reward,
            "rewards": rewards,
            "exception": {
                "type": exc.get("exception_type") or exc.get("type"),
                "message": str(exc.get("exception_message") or exc.get("message") or "")[:4000],
            } if exc else None,
            "timings": {
                key: duration(data.get(key))
                for key in ("environment_setup", "agent_setup", "agent_execution", "verifier")
                if isinstance(data.get(key), dict)
            },
            "metrics": {
                key: agent_result.get(key)
                for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd")
            },
            "agent": agent_summary(agent_dir),
            "started_at": data.get("started_at"),
            "finished_at": data.get("finished_at"),
            "failed_tests": failed_tests(load_json(verifier_dir / "ctrf.json")),
            "stdout_tail": stdout.read_text(errors="replace")[-STDOUT_TAIL:] if stdout.exists() else "",
            "runner_status": {k: v for k, v in status.get(unit, {}).items() if k != "tail"},
            "logs": f"logs/{unit}/{trial_dir.name}",
        })

        dest = args.out / "logs" / unit / trial_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for src_dir, names in ((verifier_dir, VERIFIER_FILES), (agent_dir, AGENT_FILES)):
            for name in names:
                src = src_dir / name
                if src.exists() and src.stat().st_size < MAX_COPY_BYTES:
                    shutil.copy2(src, dest / name)
        for name in ("trial.log", "result.json"):
            src = trial_dir / name
            if src.exists():
                (dest / name).write_text(src.read_text(errors="replace")[-200_000:])

    # Units the runner attempted that never produced a trial result at all.
    for unit, entry in status.items():
        if unit in seen_units:
            continue
        records.append({
            "unit": unit, "task": None, "trial": None, "shard": args.shard, "runner": args.runner,
            "state": "harness-failure", "reward": None, "rewards": {},
            "exception": {"type": "NoTrialResult", "message": entry.get("tail", "")},
            "timings": {}, "metrics": {}, "agent": {}, "failed_tests": [], "stdout_tail": "",
            "runner_status": {k: v for k, v in entry.items() if k != "tail"},
        })

    (args.out / f"shard-{args.shard:03d}.json").write_text(json.dumps({"shard": args.shard, "records": records}, indent=2))
    print(f"collected {len(records)} record(s) for shard {args.shard}")
    for r in records:
        print(f"  {r['state']:<16} reward={r['reward']}  {r['unit']}")


if __name__ == "__main__":
    main()
