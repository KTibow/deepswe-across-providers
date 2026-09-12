#!/usr/bin/env python3
"""Turn one shard's pier job directory into a compact JSON record set.

Reads every trial ``results.json`` pier wrote, pairs it with the verifier's
``reward.json`` / ``ctrf.json`` / ``test-stdout.txt``, folds in the per-task
status line the runner script recorded, and copies the small log files worth
keeping into ``<out>/logs/<task>/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

KEEP_LOGS = ("reward.json", "ctrf.json", "test-stdout.txt", "run.log")
STDOUT_TAIL = 6000


def _duration(phase: Any) -> float | None:
    """TimingInfo carries started_at/finished_at, not a duration."""
    if not isinstance(phase, dict):
        return None
    if phase.get("duration_sec") is not None:
        return phase["duration_sec"]
    start, end = phase.get("started_at"), phase.get("finished_at")
    if not start or not end:
        return None
    try:
        return (
            datetime.fromisoformat(str(end).replace("Z", "+00:00"))
            - datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        ).total_seconds()
    except ValueError:
        return None


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def failed_tests(ctrf: Any) -> list[dict[str, str]]:
    """Pull failing test names out of a CTRF report, tolerating shape drift."""
    tests: list[Any] = []
    stack = [ctrf]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if isinstance(node.get("tests"), list):
                tests = node["tests"]
                break
            stack.extend(node.values())
    out = []
    for test in tests:
        if not isinstance(test, dict):
            continue
        if str(test.get("status", "")).lower() in ("passed", "pass", "ok"):
            continue
        out.append(
            {
                "name": str(test.get("name", "?")),
                "status": str(test.get("status", "?")),
                "message": str(test.get("message", ""))[:600],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-dir", type=Path, required=True)
    ap.add_argument("--status", type=Path, help="status.jsonl from the runner loop")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--runner", default="")
    ap.add_argument("--agent", default="oracle")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logs_root = args.out / "logs"

    status: dict[str, dict[str, Any]] = {}
    if args.status and args.status.exists():
        for line in args.status.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            status[entry["task"]] = entry

    records: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()

    # pier writes jobs/<job>/<trial>/result.json per trial and a job-level
    # result.json next to it; the latter has no task_name and is skipped.
    candidates = sorted(
        set(args.jobs_dir.rglob("result.json")) | set(args.jobs_dir.rglob("results.json"))
    )
    for results_path in candidates:
        data = load_json(results_path)
        if not isinstance(data, dict) or "task_name" not in data:
            continue
        trial_dir = results_path.parent
        task = str(data["task_name"]).split("/")[-1]
        # jobs/<job-name>/... — the job name is how we tie a trial replay
        # back to the rollout it came from.
        rel = results_path.relative_to(args.jobs_dir).parts
        job = rel[0] if rel else ""
        seen_tasks.add(task)

        verifier_dir = trial_dir / "verifier"
        reward_json = load_json(verifier_dir / "reward.json")
        rewards = (data.get("verifier_result") or {}).get("rewards")
        if not rewards and isinstance(reward_json, dict):
            rewards = reward_json
        rewards = rewards or {}
        reward = rewards.get("reward")

        exc = data.get("exception_info") or {}
        stdout_path = verifier_dir / "test-stdout.txt"
        stdout_tail = ""
        if stdout_path.exists():
            text = stdout_path.read_text(errors="replace")
            stdout_tail = text[-STDOUT_TAIL:]

        if exc:
            state = "error"
        elif reward is None:
            state = "no-reward"
        elif float(reward) >= 1:
            state = "resolved"
        else:
            state = "unresolved"

        record = {
            "task": task,
            "job": job,
            "trial": data.get("trial_name"),
            "shard": args.shard,
            "runner": args.runner,
            "agent": args.agent,
            "state": state,
            "reward": reward,
            "rewards": rewards,
            "apply_failed": bool(rewards.get("apply_failed")),
            "exception": {
                "type": exc.get("exception_type") or exc.get("type"),
                "message": str(exc.get("exception_message") or exc.get("message") or "")[:1000],
            }
            if exc
            else None,
            "timings": {
                key: _duration(data.get(key))
                for key in ("environment_setup", "agent_setup", "agent_execution", "verifier")
                if isinstance(data.get(key), dict)
            },
            "metrics": {
                key: (data.get("agent_result") or {}).get(key)
                for key in (
                    "n_input_tokens",
                    "n_cache_tokens",
                    "n_output_tokens",
                    "cost_usd",
                    "peak_context_tokens",
                    "summarization_count",
                    "n_agent_steps",
                )
            },
            "n_agent_steps": data.get("n_agent_steps")
            or (data.get("agent_result") or {}).get("n_agent_steps"),
            "started_at": data.get("started_at"),
            "finished_at": data.get("finished_at"),
            "failed_tests": failed_tests(load_json(verifier_dir / "ctrf.json")),
            "stdout_tail": stdout_tail,
            "runner_status": status.get(task),
        }
        records.append(record)

        dest = logs_root / job / trial_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in KEEP_LOGS:
            src = verifier_dir / name
            if src.exists() and src.stat().st_size < 2_000_000:
                shutil.copy2(src, dest / name)
        for agent_log in ("oracle.txt", "replay.txt", "exit-code.txt"):
            src = trial_dir / "agent" / agent_log
            if src.exists() and src.stat().st_size < 1_000_000:
                shutil.copy2(src, dest / agent_log)
        trial_log = trial_dir / "trial.log"
        if trial_log.exists():
            (dest / "trial.log").write_text(
                trial_log.read_text(errors="replace")[-40_000:]
            )

    # Tasks the runner attempted that never produced a trial result at all.
    for task, entry in status.items():
        if task in seen_tasks:
            continue
        records.append(
            {
                "task": task,
                "job": task,
                "trial": None,
                "shard": args.shard,
                "runner": args.runner,
                "agent": args.agent,
                "state": "harness-failure",
                "reward": None,
                "rewards": {},
                "apply_failed": False,
                "exception": {"type": "NoTrialResult", "message": entry.get("tail", "")},
                "timings": {},
                "failed_tests": [],
                "stdout_tail": "",
                "runner_status": entry,
            }
        )

    payload = {"shard": args.shard, "runner": args.runner, "agent": args.agent, "records": records}
    (args.out / f"shard-{args.shard:03d}.json").write_text(json.dumps(payload, indent=2))
    print(f"collected {len(records)} record(s) for shard {args.shard}")
    for record in records:
        print(f"  {record['state']:<16} reward={record['reward']}  {record['task']}")


if __name__ == "__main__":
    main()
