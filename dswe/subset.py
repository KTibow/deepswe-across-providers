"""Pick task subsets that spend few rollouts for a lot of signal on one model.

    python3 -m dswe.subset glm-5.3 sentinel --n 8
    python3 -m dswe.subset glm-5.3 representative --n 12

Both read the reference config's four published rollouts per task.

* sentinel — tasks the reference config solved every time, shortest first.
  A healthy provider should pass nearly all of them, so each failure is
  informative; a subset of tasks the model already fails can't show a
  provider making it worse.
* representative — stratified by how many of the four the reference config
  solved, in proportion to the whole benchmark, so the subset's expected
  score is close to the config's headline number. Shortest tasks within each
  stratum, to keep runs cheap.

Both take at most one task per upstream repository, so one flaky test suite
or one misunderstood codebase can't count twice.

Each task's expected pass rate shrinks the reference config's 4-rollout rate
toward the task's rate across every published config:
(passes + 2 * pooled) / (attempts + 2). Four rollouts alone would put 1.0 on
48 tasks for glm-5.3, which no rerun will match.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any

from dswe import profiles, published

REPO = Path(__file__).resolve().parent.parent
PRIOR = 2.0


def task_stats(config: str) -> list[dict[str, Any]]:
    rows = [r for r in published.table("trials") if r.get("included_in_score")]
    meta = {t["id"]: t for t in published.table("tasks")}
    pooled: dict[str, list[int]] = collections.defaultdict(list)
    ref: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        pooled[r["task_name"]].append(int(bool(r["passed"])))
        if r["config"] == config:
            ref[r["task_name"]].append(r)
    if not ref:
        raise SystemExit(f"no published rollouts for {config}")
    stats = []
    for task, rs in sorted(ref.items()):
        passes = sum(1 for r in rs if r["passed"])
        q = sum(pooled[task]) / len(pooled[task])
        stats.append({
            "task": task,
            "language": meta.get(task, {}).get("language"),
            "repo": meta.get(task, {}).get("repo") or meta.get(task, {}).get("repository"),
            "reference": [passes, len(rs)],
            "pooled_rate": round(q, 4),
            "expected": round((passes + PRIOR * q) / (len(rs) + PRIOR), 4),
            "median_agent_seconds": round(statistics.median([r.get("agent_duration_seconds") or 0 for r in rs])),
            "median_steps": statistics.median([r.get("n_agent_steps") or 0 for r in rs]),
        })
    return stats


def take(candidates: list[dict], n: int, used_repos: set[str]) -> list[dict]:
    """Shortest first, one per repository, languages taken in turn."""
    by_language: dict[str, list[dict]] = collections.defaultdict(list)
    for row in sorted(candidates, key=lambda r: (r["median_agent_seconds"], r["task"])):
        by_language[row["language"] or "?"].append(row)
    # Languages with the most candidates go first in each round.
    order = sorted(by_language, key=lambda lang: (-len(by_language[lang]), lang))
    chosen: list[dict] = []
    while len(chosen) < n and any(by_language.values()):
        for lang in order:
            queue = by_language[lang]
            while queue and queue[0]["repo"] in used_repos:
                queue.pop(0)
            if queue and len(chosen) < n:
                row = queue.pop(0)
                chosen.append(row)
                used_repos.add(row["repo"])
    return chosen


def sentinel(stats: list[dict], n: int) -> list[dict]:
    return take([r for r in stats if r["reference"][0] == r["reference"][1]], n, set())


def representative(stats: list[dict], n: int) -> list[dict]:
    strata: dict[int, list[dict]] = collections.defaultdict(list)
    for r in stats:
        strata[r["reference"][0]].append(r)
    quotas = {k: n * len(v) / len(stats) for k, v in strata.items()}
    counts = {k: math.floor(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: quotas[k] - counts[k], reverse=True)[: n - sum(counts.values())]:
        counts[k] += 1
    used: set[str] = set()
    chosen = []
    for k in sorted(strata, reverse=True):
        chosen += take(strata[k], counts[k], used)
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="a model from providers.toml")
    ap.add_argument("kind", choices=["sentinel", "representative"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--name", help="default: <model>-<kind>")
    args = ap.parse_args()

    config = next(profiles.load(f"{p}:{args.model}") for p in ("crof", "openrouter")).reference_config
    stats = task_stats(config)
    chosen = sentinel(stats, args.n) if args.kind == "sentinel" else representative(stats, args.n)
    ref_rate = sum(r["reference"][0] for r in stats) / sum(r["reference"][1] for r in stats)
    subset = {
        "name": args.name or f"{args.model}-{args.kind}",
        "model": args.model,
        "reference_config": config,
        "kind": args.kind,
        "rule": (__doc__ or "").split("* " + args.kind)[1].split("\n\n")[0].strip(" —\n"),
        "expected_pass_rate": round(statistics.mean(r["expected"] for r in chosen), 4),
        "reference_pass_rate_here": round(sum(r["reference"][0] for r in chosen) / sum(r["reference"][1] for r in chosen), 4),
        "reference_pass_rate_overall": round(ref_rate, 4),
        "median_agent_seconds": statistics.median(r["median_agent_seconds"] for r in chosen),
        "tasks": sorted(chosen, key=lambda r: r["task"]),
    }
    path = REPO / "subsets" / f"{subset['name']}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(subset, indent=2) + "\n")
    print(f"wrote {path.relative_to(REPO)}: {len(chosen)} tasks, expected {subset['expected_pass_rate']:.1%}, "
          f"reference {subset['reference_pass_rate_here']:.1%} here vs {ref_rate:.1%} overall, "
          f"median {subset['median_agent_seconds'] / 60:.0f} min")
    for r in subset["tasks"]:
        print(f"  {r['task']:<46} {r['language'] or '?':<11} ref {r['reference'][0]}/{r['reference'][1]}  "
              f"expected {r['expected']:.2f}  {r['median_agent_seconds'] / 60:.0f} min")


if __name__ == "__main__":
    main()
