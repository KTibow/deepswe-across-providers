#!/usr/bin/env python3
"""Choose a subset seed whose tasks are as hard as the benchmark as a whole.

`plan.py --n-tasks N --seed S` names a fixed subset. Which S you pick matters:
at N=12 the subsets range several points either side of the full benchmark, and
that swing is the same size as the provider effect you are probably trying to
measure. This scores each candidate seed against the published rollout table and
recommends the closest one.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

SITE = "https://deepswe.datacurve.ai"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", type=Path, required=True)
    ap.add_argument("--n-tasks", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=20, help="try seeds 0..N-1")
    ap.add_argument("--release", default="v1.1")
    ap.add_argument("--site", default=SITE)
    ap.add_argument("--trials-json", type=Path)
    args = ap.parse_args()

    every = sorted(p.name for p in args.tasks_dir.iterdir() if (p / "task.toml").is_file())
    if args.n_tasks >= len(every):
        sys.exit(f"--n-tasks must be below {len(every)}")

    if args.trials_json and args.trials_json.exists():
        rows = json.loads(args.trials_json.read_text())["rows"]
    else:
        url = f"{args.site}/artifacts/{args.release}/trials.json"
        print(f"fetching {url}", file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": "deepswe-replay/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            rows = json.load(resp)["rows"]

    scored = [r for r in rows if r.get("included_in_score")]
    full = sum(1 for r in scored if r.get("passed")) / len(scored)

    per_task: dict[str, list[int]] = {}
    for row in scored:
        per_task.setdefault(row["task_name"], []).append(1 if row.get("passed") else 0)

    results = []
    for seed in range(args.seeds):
        shuffled = list(every)
        random.Random(seed).shuffle(shuffled)
        chosen = sorted(shuffled[: args.n_tasks])
        hits = sum(sum(per_task.get(t, [])) for t in chosen)
        total = sum(len(per_task.get(t, [])) for t in chosen)
        rate = hits / total if total else 0.0
        results.append((seed, rate, chosen))

    print(f"full benchmark: {100 * full:.1f}% over {len(scored)} rollouts\n")
    print(f"{'seed':>5} {'subset rate':>12} {'vs full':>9}")
    for seed, rate, _ in results:
        print(f"{seed:>5} {100 * rate:>11.1f}% {100 * (rate - full):>+8.1f}")

    best = min(results, key=lambda r: abs(r[1] - full))
    print(
        f"\nmost representative: --seed {best[0]} "
        f"({100 * best[1]:.1f}%, {100 * (best[1] - full):+.1f} vs full)"
    )
    for task in best[2]:
        rates = per_task.get(task, [])
        print(f"  {100 * (sum(rates) / len(rates)) if rates else 0:5.1f}%  {task}")


if __name__ == "__main__":
    main()
