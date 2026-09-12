#!/usr/bin/env python3
"""Merge shard results into one report and compare against the expected score.

For ``--agent oracle`` the expectation is exact: DeepSWE's fail-to-pass and
pass-to-pass whitelists were materialised from an oracle-vs-nop differential
(see tests/grader.py in datacurve-ai/deep-swe), so replaying the reference
solution must score reward=1 on every task. ``--agent nop`` inverts that:
every task must score 0. Anything else is a reproduction difference.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ICON = {"resolved": "PASS", "unresolved": "FAIL", "error": "ERROR",
        "no-reward": "NO-REWARD", "harness-failure": "HARNESS"}


def pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.1f}%" if den else "n/a"


def fmt_secs(value: Any) -> str:
    try:
        return f"{float(value):.0f}s"
    except (TypeError, ValueError):
        return "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--plan", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--agent", default="oracle")
    ap.add_argument("--meta", type=Path, help="JSON blob of run metadata")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    expected_reward = 0 if args.agent == "nop" else 1

    records: list[dict[str, Any]] = []
    for shard_file in sorted(args.results_dir.rglob("shard-*.json")):
        payload = json.loads(shard_file.read_text())
        records.extend(payload.get("records", []))
    records.sort(key=lambda r: (r["task"], str(r.get("trial"))))

    language = {}
    if args.manifest and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
        language = {t["task_id"]: t.get("language", "?") for t in manifest.get("tasks", [])}

    planned = []
    if args.plan and args.plan.exists():
        planned = json.loads(args.plan.read_text()).get("tasks", [])

    meta = {}
    if args.meta and args.meta.exists():
        meta = json.loads(args.meta.read_text())

    attempted = {r["task"] for r in records}
    missing = [t for t in planned if t not in attempted]

    def matches(record: dict[str, Any]) -> bool:
        return record.get("reward") is not None and float(record["reward"]) == expected_reward

    agreeing = [r for r in records if matches(r)]
    differing = [r for r in records if not matches(r)]

    by_language: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        lang = language.get(record["task"], "?")
        by_language[lang]["total"] += 1
        by_language[lang]["match"] += 1 if matches(record) else 0

    lines: list[str] = []
    lines.append(f"# DeepSWE replay — `{args.agent}` agent")
    lines.append("")
    if meta:
        lines.append("| setting | value |")
        lines.append("| --- | --- |")
        for key, value in meta.items():
            lines.append(f"| {key} | `{value}` |")
        lines.append("")

    lines.append("## Result")
    lines.append("")
    lines.append(f"- trials run: **{len(records)}** across **{len({r['task'] for r in records})}** task(s)")
    lines.append(f"- expected reward for `{args.agent}`: **{expected_reward}** on every task")
    lines.append(f"- matched expectation: **{len(agreeing)}/{len(records)}** ({pct(len(agreeing), len(records))})")
    lines.append(f"- **differences: {len(differing)}**")
    if missing:
        lines.append(f"- planned but never reported: {len(missing)} ({', '.join(missing)})")
    lines.append("")

    states = Counter(r["state"] for r in records)
    lines.append("State breakdown: " + ", ".join(f"`{k}` {v}" for k, v in sorted(states.items())) or "none")
    lines.append("")

    if by_language:
        lines.append("| language | matched | total |")
        lines.append("| --- | ---: | ---: |")
        for lang in sorted(by_language):
            counts = by_language[lang]
            lines.append(f"| {lang} | {counts['match']} | {counts['total']} |")
        lines.append("")

    lines.append("## Per-task")
    lines.append("")
    lines.append("| task | lang | state | reward | f2p | p2p | env | agent | verify |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for record in records:
        rewards = record.get("rewards") or {}
        timings = record.get("timings") or {}
        f2p = f"{rewards.get('f2p_passed', '-')}/{rewards.get('f2p_total', '-')}"
        p2p = f"{rewards.get('p2p_passed', '-')}/{rewards.get('p2p_total', '-')}"
        lines.append(
            f"| `{record['task']}` | {language.get(record['task'], '?')} "
            f"| {ICON.get(record['state'], record['state'])} | {record.get('reward')} "
            f"| {f2p} | {p2p} | {fmt_secs(timings.get('environment_setup'))} "
            f"| {fmt_secs(timings.get('agent_execution'))} | {fmt_secs(timings.get('verifier'))} |"
        )
    lines.append("")

    if differing:
        lines.append("## Differences")
        lines.append("")
        for record in differing:
            lines.append(f"### `{record['task']}` — {record['state']} (reward={record.get('reward')})")
            lines.append("")
            rewards = record.get("rewards") or {}
            if rewards:
                lines.append(f"- rewards: `{json.dumps(rewards)}`")
            if record.get("apply_failed"):
                lines.append("- **the collected patch failed to apply in the verifier container**")
            exc = record.get("exception")
            if exc and exc.get("type"):
                lines.append(f"- exception: `{exc['type']}` — {exc.get('message', '')[:400]}")
            status = record.get("runner_status") or {}
            if status:
                lines.append(f"- runner: exit={status.get('rc')} after {fmt_secs(status.get('seconds'))}")
            failed = record.get("failed_tests") or []
            if failed:
                lines.append(f"- failing tests ({len(failed)}):")
                for test in failed[:15]:
                    lines.append(f"  - `{test['name']}` [{test['status']}] {test['message'][:200]}")
                if len(failed) > 15:
                    lines.append(f"  - ... {len(failed) - 15} more")
            tail = (record.get("stdout_tail") or "").strip()
            if tail:
                lines.append("")
                lines.append("<details><summary>verifier stdout tail</summary>")
                lines.append("")
                lines.append("```")
                lines.append(tail[-2500:])
                lines.append("```")
                lines.append("")
                lines.append("</details>")
            lines.append("")
    else:
        lines.append("## Differences")
        lines.append("")
        lines.append("None. Every trial scored exactly what the benchmark's own construction implies.")
        lines.append("")

    summary = "\n".join(lines)
    (args.out / "summary.md").write_text(summary)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "agent": args.agent,
                "expected_reward": expected_reward,
                "meta": meta,
                "n_records": len(records),
                "n_matched": len(agreeing),
                "n_differences": len(differing),
                "missing_tasks": missing,
                "states": dict(states),
                "records": records,
            },
            indent=2,
        )
    )
    print(summary[:12000])


if __name__ == "__main__":
    main()
