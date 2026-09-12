"""Turn a run request into units of work, sharded across runners.

A unit is one pier job: a task, run by some agent, optionally resumed from a
recorded trajectory.

    # a provider on a committed subset, two rollouts per task
    python3 -m dswe.plan make --agent crof:glm-5.3 --subset glm-5.3-sentinel --attempts 2

    # resume published rollouts a quarter and half way through, then let crof finish
    python3 -m dswe.plan make --agent crof:glm-5.3 \
        --prefix-trials abs-module-cache-flags__KeBKjxc --prefix-steps 25%,50%

    # the same, picking the reference config's failed rollout on each subset task
    python3 -m dswe.plan make --agent crof:glm-5.3 --subset glm-5.3-sentinel --prefix-pick fail --prefix-steps 50%

    # reproduce published rollouts end to end with no model at all
    python3 -m dswe.plan make --agent replay --prefix-trials abs-module-cache-flags__KeBKjxc

    # the reference solution / an empty submission
    python3 -m dswe.plan make --agent oracle --n-tasks 12 --seed 1

    python3 -m dswe.plan args plan.json <unit>   # pier flags for one unit
    python3 -m dswe.plan task plan.json <unit>   # its task id

`make` writes plan.json, fetches prefix trajectories into prefixes/ next to it,
and emits a GitHub Actions matrix on $GITHUB_OUTPUT.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from dswe import atif, profiles, published

REPO = Path(__file__).resolve().parent.parent
BUILTIN = {"oracle", "nop", "replay"}


def all_tasks(tasks_dir: Path) -> list[str]:
    return sorted(p.name for p in tasks_dir.iterdir() if (p / "task.toml").is_file())


def select_tasks(args: argparse.Namespace, every: list[str]) -> tuple[list[str], str]:
    """Explicit globs, a committed subset, a seeded sample, or everything.

    Sorted before shuffling: pier's own --n-tasks/--sample-seed shuffles in
    filesystem order, so the same seed can pick different tasks on different
    machines.
    """
    patterns = [p for p in args.tasks.replace(",", " ").split() if p]
    if patterns:
        unmatched = [p for p in patterns if not any(fnmatch(t, p) for t in every)]
        if unmatched:
            sys.exit(f"no task matched: {unmatched}")
        chosen = [t for t in every if any(fnmatch(t, p) for p in patterns)]
        return chosen, f"{len(chosen)} chosen task(s)"
    if args.subset:
        data = json.loads((REPO / "subsets" / f"{args.subset}.json").read_text())
        tasks = [row["task"] for row in data["tasks"]]
        missing = sorted(set(tasks) - set(every))
        if missing:
            sys.exit(f"subset {args.subset} names tasks not in this benchmark checkout: {missing}")
        return sorted(tasks), f"subset {args.subset} ({len(tasks)} tasks)"
    if 0 < args.n_tasks < len(every):
        shuffled = list(every)
        random.Random(args.seed).shuffle(shuffled)
        return sorted(shuffled[: args.n_tasks]), f"{args.n_tasks} tasks sampled with seed {args.seed}"
    return list(every), f"all {len(every)} tasks"


def resolve_steps(spec: str, total: int) -> int:
    spec = spec.strip()
    if spec in ("all", "-1"):
        return -1
    if spec.endswith("%"):
        return round(total * float(spec[:-1]) / 100)
    return min(int(spec), total)


def pick_trials(tasks: list[str], config: str, outcome: str, rows: list[dict[str, Any]]) -> list[str]:
    """One published rollout per task from `config`, deterministically."""
    chosen = []
    for task in tasks:
        candidates = sorted(
            r["trial_name"]
            for r in rows
            if r["task_name"] == task and r["config"] == config and r.get("has_trajectory")
            and r.get("included_in_score") and (outcome == "any" or r.get("outcome") == outcome)
        )
        if candidates:
            chosen.append(candidates[0])
        else:
            print(f"note: {config} has no {outcome} rollout with a trajectory on {task}", file=sys.stderr)
    return chosen


def expand_shorthand(args: argparse.Namespace) -> None:
    """The workflow's two free-text inputs, spelled out as the regular flags."""
    select = args.select.strip()
    if select.startswith("subset:"):
        args.subset = select.split(":", 1)[1]
    elif select.startswith("sample:"):
        _, n, *seed = select.split(":")
        args.n_tasks, args.seed = int(n), int(seed[0]) if seed else 0
    elif select and select != "all":
        args.tasks = select
    for token in args.prefix.split():
        key, _, value = token.partition("=")
        if key == "trials":
            args.prefix_trials = value
        elif key == "pick":
            args.prefix_pick = value
        elif key == "steps":
            args.prefix_steps = value
        elif key == "observations":
            args.prefix_observations = value
        else:
            sys.exit(f"unknown prefix setting {key!r}; use trials=, pick=, steps=, observations=")


def published_source(row: dict[str, Any]) -> dict[str, Any]:
    return {"trial": row["trial_name"], "traj": published.trajectory(row["trial_name"]), "row": row}


def own_rollouts(ref: str, cache: Path) -> list[dict[str, Any]]:
    """Trajectories from one of our earlier runs: `run:<run id>` or `run:<run id>/<unit>`.

    Downloads that run's shard artifacts with the gh CLI (GH_TOKEN in Actions).
    """
    run_id, _, unit = ref.removeprefix("run:").partition("/")
    dest = cache / run_id
    if not dest.exists():
        repo = ["-R", os.environ["GH_REPO"]] if os.environ.get("GH_REPO") else []
        try:
            subprocess.run(["gh", "run", "download", run_id, *repo, "-p", "results-shard-*", "-D", str(dest)], check=True)
        except (OSError, subprocess.CalledProcessError) as e:
            sys.exit(f"could not download the results of run {run_id}: {e}")
    sources = []
    for shard in sorted(dest.rglob("shard-*.json")):
        for record in json.loads(shard.read_text()).get("records", []):
            if unit and record["unit"] != unit or not record.get("logs"):
                continue
            path = shard.parent / record["logs"] / "trajectory.json"
            if not path.exists():
                print(f"note: no trajectory kept for {record['unit']} in run {run_id}", file=sys.stderr)
                continue
            rewards = record.get("rewards") or {}
            sources.append({
                "trial": record["trial"],
                "traj": json.loads(path.read_text()),
                "row": {"task_name": record["task"], "config": f"run {run_id}", "reward": record.get("reward"),
                        "f2p_passed": rewards.get("f2p_passed"), "f2p_total": rewards.get("f2p_total")},
            })
    if not sources:
        sys.exit(f"no trajectories found for {ref}")
    return sources


def make(args: argparse.Namespace) -> None:
    expand_shorthand(args)
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    every = all_tasks(args.tasks_dir)
    profile = None if args.agent in BUILTIN else profiles.load(args.agent)
    reference_config = profile.reference_config if profile else None

    units: list[dict[str, Any]] = []
    explicit_trials = [t for t in args.prefix_trials.replace(",", " ").split() if t]
    if explicit_trials or args.prefix_pick:
        sources: list[dict[str, Any]] = []
        if explicit_trials:
            own = [t for t in explicit_trials if t.startswith("run:")]
            for ref in own:
                sources += own_rollouts(ref, out_dir / "runs")
            names = [t for t in explicit_trials if not t.startswith("run:")]
            if names:
                by_name = {r["trial_name"]: r for r in published.table("trials")}
                unknown = [t for t in names if t not in by_name]
                if unknown:
                    sys.exit(f"not a published rollout: {unknown}")
                sources += [published_source(by_name[t]) for t in names]
            mode = "continuing named recorded rollouts"
        else:
            if not reference_config:
                sys.exit("--prefix-pick needs a profile with a reference_config")
            tasks, mode = select_tasks(args, every)
            rows = published.table("trials")
            by_name = {r["trial_name"]: r for r in rows}
            sources = [published_source(by_name[t]) for t in pick_trials(tasks, reference_config, args.prefix_pick, rows)]
            kind = {"pass": "passing ", "fail": "failing ", "any": ""}[args.prefix_pick]
            mode = f"continuing {kind}published rollouts of {reference_config} on {mode}"
        (out_dir / "prefixes").mkdir(parents=True, exist_ok=True)
        for source in sources:
            trial, traj = source["trial"], source["traj"]
            (out_dir / "prefixes" / f"{trial}.json").write_text(json.dumps(traj))
            total = len([s for s in atif.agent_steps(traj) if s.get("tool_calls")])
            row = source["row"]
            for spec in args.prefix_steps.split(","):
                steps = resolve_steps(spec, total)
                if args.agent == "replay" and steps != -1:
                    sys.exit("--agent replay has no model to hand off to; use --prefix-steps all")
                units.append({
                    "id": f"{trial}.s{'all' if steps == -1 else steps}",
                    "task": row["task_name"],
                    "attempts": args.attempts,
                    "prefix": {
                        "trial": trial,
                        "path": f"prefixes/{trial}.json",
                        "steps": steps,
                        "total_steps": total,
                        "observations": args.prefix_observations,
                        "mini_swe_agent_version": (traj.get("agent") or {}).get("version"),
                        "recorded_model": (traj.get("agent") or {}).get("model_name"),
                        "recorded_config": row["config"],
                        "recorded_reward": row.get("reward"),
                        "recorded_f2p": [row.get("f2p_passed"), row.get("f2p_total")],
                    },
                })
        missing = sorted({u["task"] for u in units} - set(every))
        if missing:
            sys.exit(f"rollouts are for tasks not in this benchmark checkout: {missing}")
    else:
        if args.agent == "replay":
            sys.exit("--agent replay needs --prefix-trials or --prefix-pick")
        tasks, mode = select_tasks(args, every)
        units = [{"id": t, "task": t, "attempts": args.attempts, "prefix": None} for t in tasks]

    if not units:
        sys.exit("nothing to run")
    n_shards = max(1, min(args.shards, len(units)))
    buckets: list[list[str]] = [[] for _ in range(n_shards)]
    for i, unit in enumerate(units):
        buckets[i % n_shards].append(unit["id"])
    shards = [{"shard": i, "units": " ".join(b)} for i, b in enumerate(buckets) if b]

    plan = {
        "agent": args.agent,
        "profile": asdict(profile) if profile else None,
        "reference_config": reference_config,
        "selection": mode,
        "tasks_from": "rollouts" if explicit_trials else "selection",
        "tasks": sorted({u["task"] for u in units}),
        "units": units,
        "shards": shards,
    }
    args.out.write_text(json.dumps(plan, indent=2))

    if gh_out := os.environ.get("GITHUB_OUTPUT"):
        with open(gh_out, "a") as fh:
            fh.write(f"matrix={json.dumps(shards)}\n")
            fh.write(f"n_units={len(units)}\n")
            fh.write(f"n_tasks={len(plan['tasks'])}\n")
            fh.write(f"selection={mode}\n")

    print(f"{args.agent}: {len(units)} unit(s) over {len(plan['tasks'])} task(s), {len(shards)} shard(s); {mode}")
    for unit in units:
        prefix = unit["prefix"]
        extra = f"  from {prefix['trial']} step {prefix['steps']}/{prefix['total_steps']}" if prefix else ""
        print(f"  {unit['id']}{extra}")


def unit_of(plan_path: Path, unit_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = json.loads(plan_path.read_text())
    for unit in plan["units"]:
        if unit["id"] == unit_id:
            return plan, unit
    sys.exit(f"no unit {unit_id} in {plan_path}")


def pier_args(plan_path: Path, unit_id: str) -> list[str]:
    plan, unit = unit_of(plan_path, unit_id)
    agent = plan["agent"]
    prefix = unit["prefix"]
    if agent in ("oracle", "nop"):
        args = ["--agent", agent]
    elif agent == "replay":
        # No model is ever called; pier still wants a provider/model id and a key.
        args = [
            "--agent-import-path", "dswe.agent:ProviderMiniSweAgent",
            "--model", "openai/replay-only",
            "--ae", "OPENAI_API_KEY=unused",
            "--ak", "model_class=litellm",
        ]
        if prefix.get("mini_swe_agent_version"):
            args += ["--ak", f"version={prefix['mini_swe_agent_version']}"]
    else:
        profile = profiles.load(agent)
        if not os.environ.get(profile.key_env):
            sys.exit(f"{profile.key_env} is empty — set the repository secret before dispatching")
        args = profile.pier_args()
    if prefix:
        path = (plan_path.parent / prefix["path"]).resolve()
        args += [
            "--ak", f"prefix={path}",
            "--ak", f"prefix_steps={prefix['steps']}",
            "--ak", f"prefix_observations={prefix['observations']}",
        ]
    return args + ["--n-attempts", str(unit["attempts"])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make")
    mk.add_argument("--agent", required=True, help="provider:model, or oracle / nop / replay")
    mk.add_argument("--tasks-dir", type=Path, default=Path("deep-swe/tasks"))
    mk.add_argument("--tasks", default="", help="explicit task ids or globs")
    mk.add_argument("--subset", default="", help="name of a file in subsets/")
    mk.add_argument("--n-tasks", type=int, default=0, help="seeded sample size; 0 = every task")
    mk.add_argument("--seed", type=int, default=0)
    mk.add_argument("--attempts", type=int, default=1)
    mk.add_argument("--prefix-trials", default="", help="published rollouts to resume from")
    mk.add_argument("--prefix-pick", choices=["pass", "fail", "any"], help="pick one reference rollout per selected task")
    mk.add_argument("--prefix-steps", default="all", help="comma separated: N, N%%, or all")
    mk.add_argument("--prefix-observations", choices=["recorded", "live"], default="recorded")
    mk.add_argument("--select", default="", help="shorthand: subset:NAME | sample:N:SEED | all | task globs")
    mk.add_argument("--prefix", default="", help="shorthand: 'trials=A,B steps=50%%,all' or 'pick=fail steps=50%%'")
    mk.add_argument("--shards", type=int, default=10)
    mk.add_argument("--out", type=Path, default=Path("plan.json"))

    for name in ("args", "task"):
        p = sub.add_parser(name)
        p.add_argument("plan", type=Path)
        p.add_argument("unit")

    args = ap.parse_args()
    if args.cmd == "make":
        make(args)
    elif args.cmd == "args":
        print("\n".join(pier_args(args.plan, args.unit)))
    else:
        print(unit_of(args.plan, args.unit)[1]["task"])


if __name__ == "__main__":
    main()
