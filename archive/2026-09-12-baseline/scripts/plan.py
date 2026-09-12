#!/usr/bin/env python3
"""Select and shard a subset of DeepSWE tasks for a replay run.

Emits plan.json plus a GitHub Actions matrix on $GITHUB_OUTPUT.

Sampling note: pier's own ``--n-tasks/--sample-seed`` shuffles the task list in
``Path.iterdir()`` order (pier/src/pier/models/job/config.py), which is
filesystem-dependent, so the same seed can select different tasks on different
machines. We sort the ids before shuffling so a (seed, n) pair names one fixed
subset everywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from fnmatch import fnmatch
from pathlib import Path


def discover(tasks_dir: Path) -> list[str]:
    return sorted(p.name for p in tasks_dir.iterdir() if (p / "task.toml").is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", type=Path, required=True)
    ap.add_argument("--tasks", default="", help="explicit task ids or globs")
    ap.add_argument("--n-tasks", type=int, default=0, help="0 = every task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("plan.json"))
    args = ap.parse_args()

    every = discover(args.tasks_dir)
    if not every:
        sys.exit(f"no tasks found under {args.tasks_dir}")

    patterns = [p for p in args.tasks.replace(",", " ").split() if p]
    if patterns:
        unmatched = [p for p in patterns if not any(fnmatch(t, p) for t in every)]
        if unmatched:
            sys.exit(f"no task matched: {unmatched}")
        selected = [t for t in every if any(fnmatch(t, p) for p in patterns)]
        mode = "explicit"
    elif 0 < args.n_tasks < len(every):
        shuffled = list(every)
        random.Random(args.seed).shuffle(shuffled)
        selected = sorted(shuffled[: args.n_tasks])
        mode = f"sample(seed={args.seed}, n={args.n_tasks})"
    else:
        selected = list(every)
        mode = "all"

    n_shards = max(1, min(args.shards, len(selected)))
    buckets: list[list[str]] = [[] for _ in range(n_shards)]
    for i, task in enumerate(selected):
        buckets[i % n_shards].append(task)

    matrix = [
        {"shard": i, "tasks": " ".join(bucket)}
        for i, bucket in enumerate(buckets)
        if bucket
    ]
    plan = {
        "mode": mode,
        "seed": args.seed,
        "n_available": len(every),
        "n_selected": len(selected),
        "tasks": selected,
        "shards": matrix,
    }
    args.out.write_text(json.dumps(plan, indent=2))

    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as fh:
            fh.write(f"matrix={json.dumps(matrix)}\n")
            fh.write(f"n_selected={len(selected)}\n")
            fh.write(f"mode={mode}\n")

    print(json.dumps({k: v for k, v in plan.items() if k != "tasks"}, indent=2))
    for task in selected:
        print(f"  - {task}")


if __name__ == "__main__":
    main()
