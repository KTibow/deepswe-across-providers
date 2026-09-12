#!/usr/bin/env python3
"""Recompute DeepSWE's published leaderboard from its published per-rollout data.

No containers, no runners: this only checks that the numbers on the leaderboard
follow from the rollout table underneath it, and reports what the v1 -> v1.1
regrade (same rollouts, exit-code scoring replaced by node-id scoring) did to
individual configs and tasks.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import urllib.request
from pathlib import Path
from typing import Any

SITE = "https://deepswe.datacurve.ai"


def load(site: str, release: str, name: str, cache: Path | None) -> Any:
    if cache:
        local = cache / f"{release}-{name}.json"
        if local.exists():
            return json.loads(local.read_text())
    url = f"{site}/artifacts/{release}/{name}.json"
    print(f"fetching {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "deepswe-replay/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{release}-{name}.json").write_text(json.dumps(data))
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="v1.1")
    ap.add_argument("--site", default=SITE)
    ap.add_argument("--cache", type=Path, default=Path(".cache"))
    ap.add_argument("--out", type=Path, default=Path("report/published.md"))
    args = ap.parse_args()

    trials_raw = load(args.site, args.release, "trials", args.cache)["rows"]
    board = load(args.site, args.release, "leaderboard-live", args.cache)["rows"]
    tasks = load(args.site, args.release, "tasks", args.cache)["rows"]
    try:
        delta = load(args.site, args.release, "v1-delta", args.cache)
    except Exception:
        delta = None

    language = {t["id"]: t.get("language", "?") for t in tasks}
    scored = [r for r in trials_raw if r.get("included_in_score")]

    by_config: dict[str, list[dict]] = collections.defaultdict(list)
    for row in scored:
        by_config[row["config"]].append(row)

    mismatches = []
    for row in board:
        rows = by_config.get(row["config"], [])
        if not rows:
            mismatches.append((row["config"], "no rollouts", None, None))
            continue
        rate = sum(1 for r in rows if r.get("passed")) / len(rows)
        solved = len({r["task_name"] for r in rows if r.get("passed")})
        attempted = len({r["task_name"] for r in rows})
        p4 = solved / attempted if attempted else 0.0
        if abs(rate - row["pass_rate"]) > 1e-9:
            mismatches.append((row["config"], "pass@1", row["pass_rate"], rate))
        if abs(p4 - row["pass_at_4"]) > 1e-9:
            mismatches.append((row["config"], "pass@4", row["pass_at_4"], p4))

    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for row in scored:
        by_task[row["task_name"]].append(row)
    task_rate = {
        task: sum(1 for r in rows if r.get("passed")) / len(rows)
        for task, rows in by_task.items()
    }
    ordered = sorted(task_rate.items(), key=lambda kv: kv[1])
    never = [t for t, v in ordered if v == 0]
    always = [t for t, v in ordered if v == 1]

    errors = collections.Counter(
        r.get("error_category") for r in trials_raw if r.get("error_category")
    )
    excluded = [r for r in trials_raw if not r.get("included_in_score")]
    excl_by_config = collections.Counter(r["config"] for r in excluded)

    lines: list[str] = []
    lines.append(f"# DeepSWE {args.release} — published data, recomputed")
    lines.append("")
    lines.append(f"- rollouts published: **{len(trials_raw)}** ({len(scored)} scored, {len(excluded)} excluded)")
    lines.append(f"- configs: **{len(board)}**, tasks: **{len(task_rate)}**")
    lines.append(f"- pooled pass rate over scored rollouts: **{sum(1 for r in scored if r.get('passed')) / len(scored):.4f}**")
    lines.append("")
    if mismatches:
        lines.append(f"## Leaderboard does not follow from the rollout table ({len(mismatches)})")
        lines.append("")
        lines.append("| config | metric | published | recomputed |")
        lines.append("| --- | --- | ---: | ---: |")
        for config, metric, pub, mine in mismatches:
            lines.append(f"| `{config}` | {metric} | {pub} | {mine} |")
    else:
        lines.append("## Leaderboard check")
        lines.append("")
        lines.append(
            f"All **{len(board)}** configs reproduce exactly — pass@1 and pass@4 both "
            "follow from the published per-rollout table."
        )
    lines.append("")

    lines.append("## Task difficulty")
    lines.append("")
    lines.append(f"- tasks no rollout ever solved: **{len(never)}**" + (f" ({', '.join(never)})" if never else ""))
    lines.append(f"- tasks every rollout solved: **{len(always)}**" + (f" ({', '.join(always)})" if always else ""))
    lines.append("")
    lines.append("| hardest | rate | language |")
    lines.append("| --- | ---: | --- |")
    for task, rate in ordered[:10]:
        lines.append(f"| `{task}` | {rate:.3f} | {language.get(task, '?')} |")
    lines.append("")
    by_lang: dict[str, list[float]] = collections.defaultdict(list)
    for task, rate in task_rate.items():
        by_lang[language.get(task, "?")].append(rate)
    lines.append("| language | mean task pass rate | tasks |")
    lines.append("| --- | ---: | ---: |")
    for lang, rates in sorted(by_lang.items(), key=lambda kv: -statistics.mean(kv[1])):
        lines.append(f"| {lang} | {statistics.mean(rates):.3f} | {len(rates)} |")
    lines.append("")

    lines.append("## Excluded rollouts")
    lines.append("")
    lines.append(f"- excluded: **{len(excluded)}** ({100.0 * len(excluded) / len(trials_raw):.2f}%)")
    for category, count in errors.most_common():
        lines.append(f"  - `{category}`: {count}")
    worst = excl_by_config.most_common(5)
    if worst:
        lines.append("- worst-affected configs: " + ", ".join(f"`{c}` ({n})" for c, n in worst))
    lines.append("")

    if delta:
        configs = delta.get("configs", [])
        dtasks = delta.get("tasks", [])
        pooled = delta.get("pooled", {})
        lines.append("## What re-grading alone did (v1 -> v1.1)")
        lines.append("")
        lines.append(f"> {delta.get('scope', '')}")
        lines.append("")
        lines.append(
            f"- pooled pass rate barely moved: {pooled.get('v1')} -> {pooled.get('current')}"
        )
        if configs:
            biggest = max(configs, key=lambda c: abs(c.get("delta") or 0))
            lines.append(
                f"- but individual configs moved up to **{abs(biggest['delta']) * 100:.1f} points** "
                f"(`{biggest['config']}`: {biggest['v1']} -> {biggest['current']})"
            )
            lines.append("")
            lines.append("| config | v1 | v1.1 | delta |")
            lines.append("| --- | ---: | ---: | ---: |")
            for cfg in sorted(configs, key=lambda c: -abs(c.get("delta") or 0)):
                lines.append(
                    f"| `{cfg['config']}` | {cfg['v1']:.4f} | {cfg['current']:.4f} | {cfg['delta']:+.4f} |"
                )
        if dtasks:
            movers = sorted(dtasks, key=lambda t: -abs(t.get("delta") or 0))[:10]
            lines.append("")
            lines.append("| task | v1 | v1.1 | delta |")
            lines.append("| --- | ---: | ---: | ---: |")
            for task in movers:
                lines.append(
                    f"| `{task['task']}` | {task['v1']:.4f} | {task['current']:.4f} | {task['delta']:+.4f} |"
                )
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
