#!/usr/bin/env python3
"""Score a model run and put it next to DeepSWE's published results.

The comparison is only meaningful on a like-for-like subset, so the reference
numbers are recomputed from DeepSWE's per-rollout table restricted to exactly
the tasks that were run here — not taken off the leaderboard, which covers all
113 tasks.

Two rates are reported, because they answer different questions:

* **strict** — a rollout that errored counts as a failure. This is the one that
  matters when the thing under test is a provider: a 502 is a failure of the
  run you paid for.
* **DeepSWE policy** — infrastructure/provider errors are dropped and not
  resampled, which is how the published leaderboard is computed. Use this one
  when comparing against published numbers.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Any

SITE = "https://deepswe.datacurve.ai"


def wilson(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = passed / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fetch_trials(site: str, release: str, local: Path | None) -> list[dict[str, Any]]:
    if local and local.exists():
        return json.loads(local.read_text())["rows"]
    url = f"{site}/artifacts/{release}/trials.json"
    print(f"fetching {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "deepswe-replay/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["rows"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--release", default="v1.1")
    ap.add_argument("--site", default=SITE)
    ap.add_argument("--trials-json", type=Path)
    ap.add_argument(
        "--reference-configs",
        default="",
        help="published configs to compare against (comma separated); "
        "default picks the three nearest in pass rate",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    subset = list(plan["tasks"])
    meta = json.loads(args.meta.read_text()) if args.meta and args.meta.exists() else {}

    records: list[dict[str, Any]] = []
    for shard_file in sorted(args.results_dir.rglob("shard-*.json")):
        records.extend(json.loads(shard_file.read_text()).get("records", []))

    errored = [r for r in records if r["state"] in ("error", "harness-failure", "no-reward")]
    graded = [r for r in records if r["state"] in ("resolved", "unresolved")]
    passed = [r for r in graded if r["state"] == "resolved"]

    strict_total = len(graded) + len(errored)
    strict_lo, strict_hi = wilson(len(passed), strict_total)
    policy_lo, policy_hi = wilson(len(passed), len(graded))

    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    solved_any = {t for t, rs in by_task.items() if any(r["state"] == "resolved" for r in rs)}

    published = fetch_trials(args.site, args.release, args.trials_json)
    in_subset = [
        r
        for r in published
        if r["task_name"] in set(subset) and r.get("included_in_score")
    ]
    ref_rows: dict[str, list[dict]] = collections.defaultdict(list)
    for row in in_subset:
        ref_rows[row["config"]].append(row)

    ref_rates = {
        config: (sum(1 for r in rows if r.get("passed")), len(rows))
        for config, rows in ref_rows.items()
    }

    wanted = [c for c in args.reference_configs.replace(",", " ").split() if c]
    our_rate = len(passed) / len(graded) if graded else 0.0
    if not wanted:
        wanted = [
            config
            for config, _ in sorted(
                ref_rates.items(),
                key=lambda kv: abs((kv[1][0] / kv[1][1] if kv[1][1] else 0) - our_rate),
            )[:3]
        ]

    subset_pool_passed = sum(1 for r in in_subset if r.get("passed"))
    subset_pool_total = len(in_subset)
    all_scored = [r for r in published if r.get("included_in_score")]
    full_rate = (
        sum(1 for r in all_scored if r.get("passed")) / len(all_scored)
        if all_scored
        else 0.0
    )
    subset_rate = subset_pool_passed / subset_pool_total if subset_pool_total else 0.0

    lines: list[str] = []
    lines.append("# DeepSWE model run")
    lines.append("")
    if meta:
        lines.append("| setting | value |")
        lines.append("| --- | --- |")
        for key, value in meta.items():
            lines.append(f"| {key} | `{value}` |")
        lines.append("")

    lines.append("## Score")
    lines.append("")
    lines.append(f"- tasks: **{len(subset)}**, rollouts: **{len(records)}** ({len(graded)} graded, {len(errored)} errored)")
    lines.append(
        f"- **strict pass rate** (errors count as failures): "
        f"**{len(passed)}/{strict_total} = {100.0 * len(passed) / strict_total if strict_total else 0:.1f}%** "
        f"[{100 * strict_lo:.1f}, {100 * strict_hi:.1f}]"
    )
    lines.append(
        f"- **DeepSWE-policy pass rate** (errors dropped): "
        f"**{len(passed)}/{len(graded)} = {100.0 * our_rate:.1f}%** "
        f"[{100 * policy_lo:.1f}, {100 * policy_hi:.1f}]"
    )
    lines.append(f"- tasks solved at least once: **{len(solved_any)}/{len(subset)}**")
    if errored:
        cats = collections.Counter(
            (r.get("exception") or {}).get("type") or r["state"] for r in errored
        )
        lines.append("- errors: " + ", ".join(f"`{k}` {v}" for k, v in cats.most_common()))
    costs = [
        (r.get("metrics") or {}).get("cost_usd")
        for r in records
        if (r.get("metrics") or {}).get("cost_usd") is not None
    ]
    if costs:
        lines.append(f"- reported cost: ${sum(costs):.2f} total, ${sum(costs) / len(costs):.2f}/rollout")
    steps = [r.get("n_agent_steps") for r in records if r.get("n_agent_steps")]
    if steps:
        lines.append(f"- agent steps: median {sorted(steps)[len(steps) // 2]}, max {max(steps)}")
    lines.append("")

    lines.append("## Against published results, same tasks")
    lines.append("")
    swing = 100.0 * (subset_rate - full_rate)
    lines.append(
        f"Across every published config, these {len(subset)} tasks were solved in "
        f"**{subset_pool_passed}/{subset_pool_total} = {100.0 * subset_rate:.1f}%** of "
        "rollouts, so that is roughly what an average leaderboard entry scores here."
    )
    lines.append("")
    lines.append(
        f"The full 113-task benchmark sits at {100.0 * full_rate:.1f}%, so this subset is "
        f"**{abs(swing):.1f} points {'easier' if swing >= 0 else 'harder'} than average** — "
        "worth subtracting before reading anything into a gap with the leaderboard."
    )
    lines.append("")
    lines.append("| published config | pass rate on these tasks | n |")
    lines.append("| --- | ---: | ---: |")
    for config in wanted:
        hits, total = ref_rates.get(config, (0, 0))
        rate = 100.0 * hits / total if total else 0.0
        lo, hi = wilson(hits, total)
        lines.append(f"| `{config}` | {rate:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] | {total} |")
    lines.append("")

    lines.append("## Per-task")
    lines.append("")
    lines.append("| task | ours | published (all configs) | steps | cost |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    pub_by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for row in in_subset:
        pub_by_task[row["task_name"]].append(row)
    for task in subset:
        rs = by_task.get(task, [])
        ours = "".join("P" if r["state"] == "resolved" else ("." if r["state"] == "unresolved" else "E") for r in rs) or "-"
        pub = pub_by_task.get(task, [])
        pub_rate = 100.0 * sum(1 for r in pub if r.get("passed")) / len(pub) if pub else float("nan")
        step_vals = [r.get("n_agent_steps") for r in rs if r.get("n_agent_steps")]
        cost_vals = [(r.get("metrics") or {}).get("cost_usd") for r in rs if (r.get("metrics") or {}).get("cost_usd")]
        lines.append(
            f"| `{task}` | `{ours}` | {pub_rate:.0f}% | "
            f"{max(step_vals) if step_vals else '-'} | "
            f"{('$%.2f' % sum(cost_vals)) if cost_vals else '-'} |"
        )
    lines.append("")
    lines.append("`P` solved, `.` not solved, `E` errored — one character per attempt.")
    lines.append("")

    failures = [r for r in records if r["state"] not in ("resolved",)]
    if failures:
        lines.append("## Non-passing rollouts")
        lines.append("")
        for record in failures:
            head = f"- `{record['task']}` — {record['state']}"
            exc = record.get("exception")
            if exc and exc.get("type"):
                head += f": `{exc['type']}` {str(exc.get('message', ''))[:200]}"
            rewards = record.get("rewards") or {}
            if rewards:
                head += f" (f2p {rewards.get('f2p_passed')}/{rewards.get('f2p_total')})"
            lines.append(head)
        lines.append("")

    summary = "\n".join(lines)
    (args.out / "summary.md").write_text(summary)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "subset": subset,
                "n_records": len(records),
                "n_graded": len(graded),
                "n_errored": len(errored),
                "n_passed": len(passed),
                "strict_rate": len(passed) / strict_total if strict_total else None,
                "policy_rate": our_rate,
                "reference": {c: ref_rates.get(c) for c in wanted},
                "subset_pool": [subset_pool_passed, subset_pool_total],
                "subset_rate": subset_rate,
                "full_benchmark_rate": full_rate,
                "records": records,
            },
            indent=2,
        )
    )
    print(summary[:12000])


if __name__ == "__main__":
    main()
