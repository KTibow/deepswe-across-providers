#!/usr/bin/env python3
"""Compare our regrade of recorded rollouts against DeepSWE's published results.

Each selected rollout has a published reward. We replay its submission patch in
our own runner and grade it with the benchmark's verifier. Agreement means the
benchmark reproduces off Datacurve's infrastructure; disagreement is the
interesting part.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def verdict(reward: Any) -> str:
    if reward is None:
        return "no-result"
    return "pass" if float(reward) >= 1 else "fail"


def fmt_secs(value: Any) -> str:
    try:
        return f"{float(value):.0f}s"
    except (TypeError, ValueError):
        return "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    published = {t["trial_name"]: t for t in plan["trials"]}
    meta = json.loads(args.meta.read_text()) if args.meta and args.meta.exists() else {}

    ours: dict[str, dict[str, Any]] = {}
    for shard_file in sorted(args.results_dir.rglob("shard-*.json")):
        for record in json.loads(shard_file.read_text()).get("records", []):
            key = record.get("job") or record.get("task")
            if key in published:
                ours[key] = record

    rows: list[dict[str, Any]] = []
    for name, pub in published.items():
        record = ours.get(name)
        theirs = verdict(pub.get("published_reward"))
        mine = verdict(record.get("reward")) if record else "no-result"
        rows.append(
            {
                "trial_name": name,
                "task": pub["task_name"],
                "model": pub.get("model"),
                "config": pub.get("config"),
                "published_reward": pub.get("published_reward"),
                "published_verdict": theirs,
                "published_f2p": pub.get("published_f2p"),
                "published_p2p": pub.get("published_p2p"),
                "our_reward": record.get("reward") if record else None,
                "our_verdict": mine,
                "our_rewards": (record or {}).get("rewards", {}),
                "agree": theirs == mine,
                "state": (record or {}).get("state", "missing"),
                "exception": (record or {}).get("exception"),
                "runner_status": (record or {}).get("runner_status"),
                "failed_tests": (record or {}).get("failed_tests", []),
                "stdout_tail": (record or {}).get("stdout_tail", ""),
                "timings": (record or {}).get("timings", {}),
            }
        )
    rows.sort(key=lambda r: (r["agree"], r["task"]))

    graded = [r for r in rows if r["our_verdict"] in ("pass", "fail")]
    agree = [r for r in graded if r["agree"]]
    disagree = [r for r in graded if not r["agree"]]
    ungraded = [r for r in rows if r["our_verdict"] == "no-result"]
    matrix = Counter((r["published_verdict"], r["our_verdict"]) for r in graded)

    lines: list[str] = []
    lines.append("# DeepSWE regrade — replaying published rollouts")
    lines.append("")
    lines.append(
        "Every rollout below is one DeepSWE published: its `model.patch` is "
        "replayed into the task environment and graded by the benchmark's own "
        "verifier, on a GitHub-hosted runner instead of Datacurve's infrastructure."
    )
    lines.append("")
    if meta:
        lines.append("| setting | value |")
        lines.append("| --- | --- |")
        for key, value in meta.items():
            lines.append(f"| {key} | `{value}` |")
        lines.append("")

    lines.append("## Agreement")
    lines.append("")
    lines.append(f"- rollouts selected: **{len(rows)}**")
    lines.append(f"- graded here: **{len(graded)}**" + (f" ({len(ungraded)} produced no verdict)" if ungraded else ""))
    if graded:
        rate = 100.0 * len(agree) / len(graded)
        lines.append(f"- agreement with published outcome: **{len(agree)}/{len(graded)} ({rate:.1f}%)**")
    lines.append(f"- **disagreements: {len(disagree)}**")
    lines.append("")
    lines.append("| published \\ ours | pass | fail |")
    lines.append("| --- | ---: | ---: |")
    for theirs in ("pass", "fail"):
        lines.append(
            f"| **{theirs}** | {matrix[(theirs, 'pass')]} | {matrix[(theirs, 'fail')]} |"
        )
    lines.append("")

    lines.append("## Per-rollout")
    lines.append("")
    lines.append("| rollout | task | model | published | ours | f2p | p2p | verify |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: |")
    for row in rows:
        rewards = row.get("our_rewards") or {}
        f2p = f"{rewards.get('f2p_passed', '-')}/{rewards.get('f2p_total', '-')}"
        p2p = f"{rewards.get('p2p_passed', '-')}/{rewards.get('p2p_total', '-')}"
        mark = "" if row["agree"] else " **≠**"
        lines.append(
            f"| `{row['trial_name']}` | `{row['task']}` | {row['model']} "
            f"| {row['published_verdict']} | {row['our_verdict']}{mark} | {f2p} | {p2p} "
            f"| {fmt_secs((row.get('timings') or {}).get('verifier'))} |"
        )
    lines.append("")

    if disagree or ungraded:
        lines.append("## Differences")
        lines.append("")
        for row in disagree + ungraded:
            lines.append(
                f"### `{row['trial_name']}` — published **{row['published_verdict']}**, "
                f"we got **{row['our_verdict']}**"
            )
            lines.append("")
            lines.append(f"- task `{row['task']}`, model `{row['model']}`, config `{row['config']}`")
            lines.append(
                f"- published f2p {row['published_f2p']}, p2p {row['published_p2p']}; "
                f"ours `{json.dumps(row.get('our_rewards') or {})}`"
            )
            exc = row.get("exception")
            if exc and exc.get("type"):
                lines.append(f"- exception: `{exc['type']}` — {str(exc.get('message', ''))[:300]}")
            status = row.get("runner_status") or {}
            if status:
                lines.append(
                    f"- runner: exit={status.get('rc')}, patch HTTP {status.get('patch_http')} "
                    f"({status.get('patch_bytes')} bytes), {fmt_secs(status.get('seconds'))}"
                )
            failed = row.get("failed_tests") or []
            if failed:
                lines.append(f"- failing tests ({len(failed)}):")
                for test in failed[:12]:
                    lines.append(f"  - `{test['name']}` [{test['status']}] {test['message'][:160]}")
                if len(failed) > 12:
                    lines.append(f"  - ... {len(failed) - 12} more")
            tail = (row.get("stdout_tail") or "").strip()
            if tail:
                lines.append("")
                lines.append("<details><summary>verifier stdout tail</summary>")
                lines.append("")
                lines.append("```")
                lines.append(tail[-2000:])
                lines.append("```")
                lines.append("")
                lines.append("</details>")
            lines.append("")
    else:
        lines.append("## Differences")
        lines.append("")
        lines.append("None — every replayed rollout landed on the same verdict it was published with.")
        lines.append("")

    summary = "\n".join(lines)
    (args.out / "summary.md").write_text(summary)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "n_selected": len(rows),
                "n_graded": len(graded),
                "n_agree": len(agree),
                "n_disagree": len(disagree),
                "confusion": {f"{k[0]}->{k[1]}": v for k, v in matrix.items()},
                "rows": rows,
            },
            indent=2,
        )
    )
    print(summary[:12000])


if __name__ == "__main__":
    main()
