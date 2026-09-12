#!/usr/bin/env python3
"""Pick published DeepSWE rollouts to regrade, and shard them.

DeepSWE publishes every rollout's outcome (``/artifacts/<release>/trials.json``)
and the submission each one produced (``model.patch`` under the trial-artifact
CDN). This selects a subset with a recorded patch, writes the expected outcome
alongside each, and splits them across runners.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

SITE = "https://deepswe.datacurve.ai"
ARTIFACT_BASE = "https://d3ujjcmjq6o8v6.cloudfront.net"


def fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "deepswe-replay/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def csv_list(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="v1.1")
    ap.add_argument("--trials-json", type=Path, help="local copy instead of fetching")
    ap.add_argument("--site", default=SITE)
    ap.add_argument("--artifact-base", default=ARTIFACT_BASE)
    ap.add_argument(
        "--trial-names",
        default="",
        help="explicit trial names (space/comma separated); bypasses sampling",
    )
    ap.add_argument("--models", default="", help="model globs, empty = any")
    ap.add_argument("--configs", default="", help="config globs, empty = any")
    ap.add_argument("--tasks", default="", help="task globs, empty = any")
    ap.add_argument(
        "--outcome",
        default="any",
        choices=["any", "pass", "fail"],
        help="restrict to rollouts the benchmark recorded as passing or failing",
    )
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument(
        "--balance",
        action="store_true",
        help="take half passing / half failing rollouts",
    )
    ap.add_argument("--out", type=Path, default=Path("trial-plan.json"))
    args = ap.parse_args()

    if args.trials_json:
        payload = json.loads(args.trials_json.read_text())
    else:
        url = f"{args.site}/artifacts/{args.release}/trials.json"
        print(f"fetching {url}", file=sys.stderr)
        payload = fetch_json(url)

    rows: list[dict[str, Any]] = payload["rows"]
    print(f"{len(rows)} published rollouts", file=sys.stderr)

    explicit = csv_list(args.trial_names)

    def keep(row: dict[str, Any]) -> bool:
        if explicit:
            return row["trial_name"] in explicit
        if not row.get("has_model_patch"):
            return False
        if not row.get("included_in_score", True):
            return False
        if row.get("reward") is None:
            return False
        if args.outcome != "any" and row.get("outcome") != args.outcome:
            return False
        for field, pats in (
            ("model", csv_list(args.models)),
            ("config", csv_list(args.configs)),
            ("task_name", csv_list(args.tasks)),
        ):
            if pats and not any(fnmatch(str(row.get(field, "")), p) for p in pats):
                return False
        return True

    pool = [r for r in rows if keep(r)]
    print(f"{len(pool)} match the filters and have a recorded patch", file=sys.stderr)
    if not pool:
        sys.exit("nothing to replay")

    if explicit:
        missing = set(explicit) - {r["trial_name"] for r in pool}
        if missing:
            sys.exit(f"no such rollout(s): {sorted(missing)}")

    rng = random.Random(args.seed)
    pool.sort(key=lambda r: r["trial_name"])

    if explicit:
        chosen = list(pool)
    elif args.balance:
        passing = [r for r in pool if r.get("outcome") == "pass"]
        failing = [r for r in pool if r.get("outcome") == "fail"]
        rng.shuffle(passing)
        rng.shuffle(failing)
        half = max(1, args.n_trials // 2)
        chosen = passing[:half] + failing[: args.n_trials - half]
    else:
        chosen = list(pool)
        rng.shuffle(chosen)
        chosen = chosen[: args.n_trials] if args.n_trials > 0 else chosen
    chosen.sort(key=lambda r: r["trial_name"])

    trials = [
        {
            "trial_name": r["trial_name"],
            "task_name": r["task_name"],
            "model": r.get("model"),
            "config": r.get("config"),
            "reasoning_effort": r.get("reasoning_effort"),
            "published_reward": r.get("reward"),
            "published_outcome": r.get("outcome"),
            "published_f2p": [r.get("f2p_passed"), r.get("f2p_total")],
            "published_p2p": [r.get("p2p_passed"), r.get("p2p_total")],
            "published_duration_sec": r.get("trial_duration_seconds"),
            "patch_url": f"{args.artifact_base}/{args.release}/trial-artifacts/{r['trial_name']}/artifacts/model.patch",
            "verifier_url": f"{args.artifact_base}/{args.release}/trial-artifacts/{r['trial_name']}/verifier/reward.json",
        }
        for r in chosen
    ]

    n_shards = max(1, min(args.shards, len(trials)))
    buckets: list[list[str]] = [[] for _ in range(n_shards)]
    for i, trial in enumerate(trials):
        buckets[i % n_shards].append(trial["trial_name"])
    matrix = [
        {"shard": i, "trials": " ".join(bucket)}
        for i, bucket in enumerate(buckets)
        if bucket
    ]

    plan = {
        "release": args.release,
        "filters": {
            "models": args.models,
            "configs": args.configs,
            "tasks": args.tasks,
            "outcome": args.outcome,
            "balance": args.balance,
        },
        "seed": args.seed,
        "n_pool": len(pool),
        "n_selected": len(trials),
        "trials": trials,
        "shards": matrix,
    }
    args.out.write_text(json.dumps(plan, indent=2))

    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as fh:
            fh.write(f"matrix={json.dumps(matrix)}\n")
            fh.write(f"n_selected={len(trials)}\n")

    n_pass = sum(1 for t in trials if t["published_outcome"] == "pass")
    print(f"selected {len(trials)} rollouts: {n_pass} recorded pass, {len(trials) - n_pass} recorded fail")
    for t in trials:
        print(f"  {t['published_outcome']:<5} {t['model']:<18} {t['trial_name']}")


if __name__ == "__main__":
    main()
