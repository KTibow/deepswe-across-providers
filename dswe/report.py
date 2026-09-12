"""Turn a run's collected records into the report a provider is judged by.

    python3 -m dswe.report --plan plan.json --results-dir shards --meta meta.json --out report

Writes ``summary.md`` (also the job summary) and ``results.json``. The summary
answers, in order:

1. **Verdict** — did the provider solve as many rollouts as the published
   reference config would be expected to on these tasks?
2. **Who owns each failure** — the model, the provider (a call that failed
   after retries, or format errors), the clock (agent timeout), or our
   harness (infrastructure, grading), which is excluded from the score.
3. **Inference health** — per call, this run next to the reference config's
   published trajectories on the same tasks: format errors, mis-decoded
   UTF-8, output and prompt tokens, cache share; plus what only this run can
   show (retries, finish reasons, empty replies, wait per call), and per
   rollout steps, tokens, minutes and cost.
4. **Per task**, then **resumed rollouts** and **replay fidelity** when the
   run started from recorded trajectories, and **non-passing rollouts** with
   where to find their logs.

oracle/nop runs get a short expectation check instead of a verdict.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any

from dswe import atif, published
from dswe.collect import C1, repeat_stats

# A streak this long of steps repeating recent commands is called out as a loop.
LOOP_STREAK = 5

PRIOR = 2.0
INFRASTRUCTURE = {
    "NoTrialResult", "EnvironmentStartTimeoutError", "HealthcheckError", "AgentSetupTimeoutError",
    "AddTestsDirError", "DownloadVerifierDirError", "MissingExtraError",
}
GRADING = {"VerifierTimeoutError", "VerifierOutputParseError", "RewardFileEmptyError", "RewardFileNotFoundError"}
AGENT_EXITS = {"Submitted", "LimitsExceeded", "TimeExceeded", "RepeatedFormatError"}
OWNER_MARK = {
    "solved": "P", "model": ".", "provider": "X", "format errors": "F",
    "agent timeout": "T", "infrastructure": "I", "grading": "G", "unknown": "?",
}
EXCLUDED = ("infrastructure", "grading")


def owner(record: dict[str, Any]) -> str:
    """Who a rollout's outcome belongs to."""
    exc = (record.get("exception") or {}).get("type") or ""
    exit_status = (record.get("agent") or {}).get("exit_status") or ""
    if record["state"] == "resolved":
        return "solved"
    if record["state"] == "harness-failure" or exc in INFRASTRUCTURE:
        return "infrastructure"
    if exc in GRADING:
        return "grading"
    if exc == "AgentTimeoutError":
        return "agent timeout"
    if exit_status == "RepeatedFormatError":
        return "format errors"
    if exit_status and exit_status not in AGENT_EXITS:
        # mini-swe-agent ends with the exception class once retries run out:
        # ServiceUnavailableError, RateLimitError, APIConnectionError, ...
        return "provider"
    if record["state"] == "unresolved":
        return "model"
    return "unknown"


def pct(num: float, den: float) -> str:
    return f"{100.0 * num / den:.1f}%" if den else "n/a"


def med(values: list[Any]) -> float | None:
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def quantile(values: list[Any], q: float) -> float | None:
    values = sorted(v for v in values if isinstance(v, (int, float)))
    return values[min(len(values) - 1, int(len(values) * q))] if values else None


def fmt(value: float | None, unit: str = "", digits: int = 0) -> str:
    if value is None:
        return "-"
    if unit == "k":
        return f"{value / 1000:.{max(digits, 1)}f}k"
    return f"{value:,.{digits}f}{unit}"


def reference_stats(config: str | None, tasks: list[str]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Per-task expected pass rate and the reference config's own rollouts."""
    if not config:
        return {}, []
    rows = [r for r in published.table("trials") if r.get("included_in_score") and r["task_name"] in set(tasks)]
    pooled: dict[str, list[int]] = collections.defaultdict(list)
    ref_rows = [r for r in rows if r["config"] == config]
    for r in rows:
        pooled[r["task_name"]].append(int(bool(r["passed"])))
    stats = {}
    for task in tasks:
        mine = [r for r in ref_rows if r["task_name"] == task]
        q = statistics.mean(pooled[task]) if pooled[task] else 0.5
        passes = sum(1 for r in mine if r["passed"])
        stats[task] = {
            "passes": passes,
            "attempts": len(mine),
            "expected": (passes + PRIOR * q) / (len(mine) + PRIOR),
            "steps": med([r.get("n_agent_steps") for r in mine]),
            "output_tokens": med([r.get("n_output_tokens") for r in mine]),
            "minutes": (med([r.get("agent_duration_seconds") for r in mine]) or 0) / 60 or None,
        }
    return stats, ref_rows


def reference_calls(ref_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The per-call numbers ATIF keeps, from the reference config's published trajectories."""
    out: dict[str, Any] = {"rollouts": 0, "calls": 0, "format_errors": 0, "c1_chars": 0,
                           "output_tokens": [], "prompt_tokens": [], "cached_tokens": [], "repeats": []}
    for row in ref_rows:
        if not row.get("has_trajectory"):
            continue
        try:
            traj = published.trajectory(row["trial_name"])
        except SystemExit:
            continue
        out["rollouts"] += 1
        out["repeats"].append(repeat_stats([
            ("\n".join(a["command"] for a in atif.actions(step)), (step.get("message") or "") + (step.get("reasoning_content") or ""))
            for step in atif.agent_steps(traj)
        ]))
        for step in atif.agent_steps(traj):
            _, followups = atif.observation_contents(step)
            # Each format-error prompt followed a call ATIF doesn't keep.
            out["calls"] += 1 + len(followups)
            out["format_errors"] += len(followups)
            text = (step.get("message") or "") + (step.get("reasoning_content") or "") + "".join(
                json.dumps(tc.get("arguments"), ensure_ascii=False) for tc in step.get("tool_calls") or [])
            out["c1_chars"] += len(C1.findall(text))
            metrics = step.get("metrics") or {}
            for key, source in (("output_tokens", "completion_tokens"), ("prompt_tokens", "prompt_tokens"), ("cached_tokens", "cached_tokens")):
                if isinstance(metrics.get(source), (int, float)):
                    out[key].append(metrics[source])
    return out


def repeat_share(stats: list[dict[str, Any]]) -> str:
    steps = sum(s["steps"] for s in stats)
    return pct(sum(s["repeated"] for s in stats), steps) if steps else "-"


def loop_count(stats: list[dict[str, Any]]) -> str:
    if not stats:
        return "-"
    loops = sum(1 for s in stats if s["longest_streak"] >= LOOP_STREAK)
    return f"{loops} of {len(stats)} (longest {max(s['longest_streak'] for s in stats)})"


def verdict_lines(records: list[dict], owners: list[str], expected: dict[str, dict], config: str | None) -> tuple[list[str], dict]:
    scored = [(r, o) for r, o in zip(records, owners) if o not in EXCLUDED]
    solved = sum(1 for _, o in scored if o == "solved")
    lines = ["## Verdict", ""]
    if not config or not scored:
        lines += ["No reference config to compare against." if not config else "Nothing was scored.", ""]
        return lines, {"scored": len(scored), "solved": solved}
    exp = sum(expected[r["task"]]["expected"] for r, _ in scored)
    sd = math.sqrt(sum(expected[r["task"]]["expected"] * (1 - expected[r["task"]]["expected"]) for r, _ in scored))
    z = (solved - exp) / sd if sd else 0.0
    tasks = {r["task"] for r, _ in scored}
    ref_pass = sum(expected[t]["passes"] for t in tasks)
    ref_total = sum(expected[t]["attempts"] for t in tasks)
    if z <= -2:
        call = f"**Below the reference.** {solved} solved where {exp:.1f} ± {1.96 * sd:.1f} was expected (z = {z:.1f})."
    elif z >= 2:
        call = f"**Above the reference.** {solved} solved where {exp:.1f} ± {1.96 * sd:.1f} was expected (z = {z:.1f})."
    else:
        call = f"**Consistent with the reference.** {solved} solved; {exp:.1f} ± {1.96 * sd:.1f} expected (z = {z:+.1f})."
    clean = [o for _, o in scored if o in ("solved", "model")]
    lines += [
        call, "",
        f"- solved **{solved}/{len(scored)} = {pct(solved, len(scored))}** of scored rollouts (provider errors, format "
        f"errors and timeouts count as failures; {len(records) - len(scored)} infrastructure/grading failure(s) left out)",
        f"- `{config}` solved {ref_pass}/{ref_total} = {pct(ref_pass, ref_total)} of its published rollouts on these tasks; "
        "the expectation pulls each task's rate toward every config's rate on it, since four rollouts can't justify 100%",
        f"- rollouts that ended without provider trouble or timeout: {clean.count('solved')}/{len(clean)} = "
        f"{pct(clean.count('solved'), len(clean))} solved",
        "",
    ]
    return lines, {"scored": len(scored), "solved": solved, "expected": exp, "sd": sd, "z": z, "reference": [ref_pass, ref_total]}


def inference_lines(records: list[dict], ref_rows: list[dict]) -> tuple[list[str], dict]:
    inf = [i for i in ((r.get("agent") or {}).get("inference") for r in records) if i and i.get("calls")]
    lines = ["## Inference health", ""]
    if not inf:
        return lines + ["No live model calls were recorded.", ""], {}
    retries: collections.Counter = collections.Counter()
    finish: collections.Counter = collections.Counter()
    for i in inf:
        retries.update(i.get("retries") or {})
        finish.update(i.get("finish_reasons") or {})
    pool = {k: [v for i in inf for v in i.get(k) or []] for k in ("latency_s", "output_tokens", "reasoning_tokens", "prompt_tokens", "cached_tokens")}
    ours = {
        "rollouts": len(inf),
        "calls": sum(i["calls"] for i in inf),
        "format_errors": sum(i["format_errors"] for i in inf),
        "c1_chars": sum(i["c1_chars"] for i in inf),
        "empty_replies": sum(i["empty_replies"] for i in inf),
        "retries": dict(retries),
        "finish_reasons": dict(finish),
        "repeats": [(r.get("agent") or {}).get("repeats") for r in records
                    if (r.get("agent") or {}).get("inference", {}).get("calls") and (r.get("agent") or {}).get("repeats")],
        **pool,
    }
    ref_tasks = {r["task"] for r in records if (r.get("agent") or {}).get("inference", {}).get("calls")}
    ref = reference_calls([r for r in ref_rows if r["task_name"] in ref_tasks]) if ref_rows else None

    lines.append(f"- live model calls: **{ours['calls']}** over {ours['rollouts']} rollout(s); retried: "
                 f"**{sum(retries.values())}**" + (" (" + ", ".join(f"`{k}` {v}" for k, v in retries.most_common()) + ")" if retries else ""))
    lines.append("- finish reasons: " + ", ".join(f"`{k}` {v}" for k, v in finish.most_common()))
    lines.append(f"- empty replies (no text, reasoning or tool call): **{ours['empty_replies']}**")
    lines.append("")

    def per_call(side: dict[str, Any] | None) -> dict[str, str]:
        if not side or not side["calls"]:
            return {}
        cached_share = sum(side["cached_tokens"]) / sum(side["prompt_tokens"]) if sum(side["prompt_tokens"]) else None
        return {
            "format-error replies": f"{side['format_errors']} of {side['calls']} ({pct(side['format_errors'], side['calls'])})",
            "mis-decoded UTF-8 (C1 chars per 100 calls)": f"{100 * side['c1_chars'] / side['calls']:.1f}",
            "output tokens, p50 / p90": f"{fmt(quantile(side['output_tokens'], 0.5))} / {fmt(quantile(side['output_tokens'], 0.9))}",
            "prompt tokens, p50 / max": f"{fmt(quantile(side['prompt_tokens'], 0.5))} / {fmt(max(side['prompt_tokens']) if side['prompt_tokens'] else None)}",
            "prompt served from cache": pct(cached_share, 1) if cached_share is not None else "-",
            "steps repeating a recent step (numbers ignored)": repeat_share(side["repeats"]),
            f"rollouts with a repeat streak of {LOOP_STREAK}+ steps": loop_count(side["repeats"]),
        }

    mine, theirs = per_call(ours), per_call(ref)
    ref_label = f"reference ({ref['rollouts']} published rollouts)" if ref and ref["rollouts"] else "reference"
    lines += [f"| per call | this run | {ref_label} |", "| --- | ---: | ---: |"]
    for key, value in mine.items():
        lines.append(f"| {key} | {value} | {theirs.get(key, '-')} |")
    lines.append(f"| wait per call, p50 / p90 / max | {fmt(quantile(pool['latency_s'], 0.5), 's', 1)} / "
                 f"{fmt(quantile(pool['latency_s'], 0.9), 's', 1)} / {fmt(max(pool['latency_s']) if pool['latency_s'] else None, 's')} | not published |")
    lines += ["", "glm-5.3 writes some mis-decoded UTF-8 even at Z.AI, so compare the rate, not the count. "
              "A repeat streak can also be an agent polling a background job (`sleep 30; cat log`); "
              "the non-passing list below names where each streak starts, which is the step to probe.", ""]

    live = [r for r in records if (r.get("agent") or {}).get("inference", {}).get("calls") and not (r.get("agent") or {}).get("replayed_steps")]
    if live and ref_rows:
        ref_live = [r for r in ref_rows if r["task_name"] in {r["task"] for r in live}]
        rows = {
            "agent steps": ([r["agent"].get("steps") for r in live], [r.get("n_agent_steps") for r in ref_live], 0),
            "output tokens": ([sum(r["agent"]["inference"]["output_tokens"]) or None for r in live], [r.get("n_output_tokens") for r in ref_live], 0),
            "peak prompt tokens": ([max(r["agent"]["inference"]["prompt_tokens"] or [0]) or None for r in live], [r.get("peak_context_tokens") for r in ref_live], 0),
            "agent minutes": ([((r.get("timings") or {}).get("agent_execution") or 0) / 60 or None for r in live],
                              [(r.get("agent_duration_seconds") or 0) / 60 or None for r in ref_live], 0),
            "cost (USD, each provider's prices)": ([(r.get("metrics") or {}).get("cost_usd") for r in live], [r.get("cost_usd") for r in ref_live], 2),
        }
        lines += ["| median per rollout | this run | reference | ratio |", "| --- | ---: | ---: | ---: |"]
        for key, (a_values, b_values, digits) in rows.items():
            a, b = med(a_values), med(b_values)
            lines.append(f"| {key} | {fmt(a, digits=digits)} | {fmt(b, digits=digits)} | {f'{a / b:.2f}' if a is not None and b else '-'} |")
        lines += ["", "The reference ran with a 5400 s agent timeout.", ""]

    summary = {k: v for k, v in ours.items() if not isinstance(v, list)}
    summary["reference"] = {k: v for k, v in (ref or {}).items() if not isinstance(v, list)}
    return lines, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    plan = json.loads(args.plan.read_text())
    meta = json.loads(args.meta.read_text()) if args.meta and args.meta.exists() else {}
    units = {u["id"]: u for u in plan["units"]}
    records: list[dict[str, Any]] = []
    for shard in sorted(args.results_dir.rglob("shard-*.json")):
        records.extend(json.loads(shard.read_text()).get("records", []))
    for r in records:
        r["task"] = r.get("task") or units.get(r["unit"], {}).get("task")
    records.sort(key=lambda r: (r["unit"], str(r.get("trial"))))
    owners = [owner(r) for r in records]
    agent = plan["agent"]
    config = plan.get("reference_config")
    expected, ref_rows = reference_stats(config, plan["tasks"])

    lines = [f"# `{agent}` — {plan['selection']}", ""]
    if plan["selection"] == "prefix(explicit)":
        # Named rollouts decide the tasks; the workflow's task input was ignored.
        meta.pop("tasks", None)
    if meta:
        lines += ["| setting | value |", "| --- | --- |"] + [f"| {k} | `{v}` |" for k, v in meta.items() if v not in ("", None)] + [""]
    missing = sorted(set(units) - {r["unit"] for r in records})
    if missing:
        lines += [f"**{len(missing)} planned unit(s) never reported:** " + ", ".join(f"`{u}`" for u in missing), ""]

    results: dict[str, Any] = {"meta": meta, "agent": agent, "selection": plan["selection"], "reference_config": config}
    has_prefix = any(u["prefix"] for u in units.values())

    if agent in ("oracle", "nop"):
        want = 0 if agent == "nop" else 1
        matched = [r for r in records if r.get("reward") is not None and float(r["reward"]) == want]
        lines += ["## Expectation", "", f"`{agent}` must score reward {want} on every task: **{len(matched)}/{len(records)}** did.", ""]
        results["matched"] = [len(matched), len(records)]
    elif not has_prefix:
        vl, results["verdict"] = verdict_lines(records, owners, expected, config)
        lines += vl

    counts = collections.Counter(owners)
    meaning = {
        "solved": "graded reward 1",
        "model": "submitted (or stopped) and failed the held-out tests",
        "provider": "a model call still failed after mini-swe-agent's 10 retries",
        "format errors": "three unusable replies in a row",
        "agent timeout": "ran out of agent time",
        "infrastructure": "our harness failed before grading — rerun, don't score",
        "grading": "the verifier failed — rerun, don't score",
        "unknown": "no agent trajectory to tell",
    }
    lines += ["## Outcomes by owner", "", "| owner | rollouts | meaning |", "| --- | ---: | --- |"]
    lines += [f"| {key} | {counts[key]} | {meaning[key]} |" for key in OWNER_MARK if counts[key]]
    lines.append("")
    results["owners"] = dict(counts)

    if agent not in ("oracle", "nop", "replay"):
        il, results["inference"] = inference_lines(records, ref_rows)
        lines += il

    lines += ["## Per task", ""]
    if expected:
        lines += ["| task | ours | reference | expected | steps ours/ref | out tok ours/ref | agent min ours/ref |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    else:
        lines += ["| task | ours | steps | agent min |", "| --- | --- | ---: | ---: |"]
    by_task: dict[str, list[tuple[dict, str]]] = collections.defaultdict(list)
    for r, o in zip(records, owners):
        by_task[r["task"]].append((r, o))
    for task in plan["tasks"]:
        rs = by_task.get(task, [])
        marks = "".join(OWNER_MARK[o] for _, o in rs) or "-"
        steps = med([(r.get("agent") or {}).get("steps") for r, _ in rs])
        minutes = med([((r.get("timings") or {}).get("agent_execution") or 0) / 60 or None for r, _ in rs])
        out_tok = med([sum(((r.get("agent") or {}).get("inference") or {}).get("output_tokens") or []) or None for r, _ in rs])
        if expected:
            e = expected[task]
            lines.append(f"| `{task}` | `{marks}` | {e['passes']}/{e['attempts']} | {e['expected']:.2f} | "
                         f"{fmt(steps)}/{fmt(e['steps'])} | {fmt(out_tok, 'k')}/{fmt(e['output_tokens'], 'k')} | "
                         f"{fmt(minutes)}/{fmt(e['minutes'])} |")
        else:
            lines.append(f"| `{task}` | `{marks}` | {fmt(steps)} | {fmt(minutes)} |")
    lines += ["", "One mark per rollout: " + ", ".join(f"`{m}` {k}" for k, m in OWNER_MARK.items()) + ".", ""]

    if has_prefix:
        lines += ["## Resumed rollouts", "",
                  "| unit | recorded result | resumed at | ours | f2p ours | live steps | exit |",
                  "| --- | --- | ---: | --- | ---: | ---: | --- |"]
        buckets: dict[str, list[str]] = collections.defaultdict(list)
        fidelity = []
        for r, o in zip(records, owners):
            prefix = units.get(r["unit"], {}).get("prefix") or {}
            if not prefix:
                continue
            steps, total = prefix["steps"], prefix["total_steps"]
            bucket = "full replay" if steps == -1 else f"{round(100 * steps / total) if total else 0}%"
            buckets[bucket].append(o)
            rec = prefix.get("recorded_reward")
            rec_text = "?" if rec is None else ("pass" if float(rec) >= 1 else "fail")
            rec_f2p = prefix.get("recorded_f2p") or [None, None]
            if rec_f2p[1] is not None:
                rec_text += f" {rec_f2p[0]}/{rec_f2p[1]}"
            rw = r.get("rewards") or {}
            ours_f2p = f"{rw.get('f2p_passed')}/{rw.get('f2p_total')}" if rw.get("f2p_total") is not None else "-"
            ag = r.get("agent") or {}
            lines.append(f"| `{r['unit']}` | {rec_text} ({prefix.get('recorded_config')}) | {'end' if steps == -1 else f'{steps}/{total}'} | "
                         f"{o} | {ours_f2p} | {(ag.get('steps') or 0) - (ag.get('replayed_steps') or 0)} | {ag.get('exit_status') or '-'} |")
            if steps == -1 and rec is not None and o in ("solved", "model"):
                fidelity.append((float(rec) >= 1) == (o == "solved"))
        lines += ["", "| resumed at | solved | of |", "| --- | ---: | ---: |"]
        for bucket in sorted(buckets, key=lambda b: (b == "full replay", float(b.rstrip("%")) if b != "full replay" else 0)):
            outs = [o for o in buckets[bucket] if o not in EXCLUDED]
            lines.append(f"| {bucket} | {outs.count('solved')} | {len(outs)} |")
        lines.append("")
        replays = [(r["unit"], (r.get("agent") or {}).get("replay")) for r in records if (r.get("agent") or {}).get("replay")]
        if replays:
            lines += ["## Replay fidelity", ""]
            if fidelity:
                lines.append(f"- full replays graded the same as the recorded rollout: **{sum(fidelity)}/{len(fidelity)}**")
            diverged = [(u, rp) for u, rp in replays if rp.get("steps_with_different_output")]
            lines.append(f"- replays whose commands printed something different from the recording: {len(diverged)}/{len(replays)}"
                         " (timestamps, temp paths and test timings make some of this normal; look at the first differing step)")
            for u, rp in diverged[:20]:
                diff = rp["steps_with_different_output"]
                kinds = rp.get("differences") or {}
                detail = ", ".join(f"{kind} {len(steps)}" for kind, steps in sorted(kinds.items()))
                first_content = f"; first content difference at step {kinds['content'][0]}" if kinds.get("content") else ""
                lines.append(f"  - `{u}`: {len(diff)} of {rp['steps_replayed']} steps"
                             + (f" ({detail}){first_content}" if detail else f", first at step {diff[0]}"))
            prompt_diff = sum(1 for _, rp in replays if rp.get("prompt_matches_recording") is False)
            if prompt_diff:
                lines.append(f"- {prompt_diff} replay(s) rendered a different prompt from the recording (usually only the "
                             "`system_information` uname line, which names the host kernel)")
            lines.append("")

    failures = [(r, o) for r, o in zip(records, owners) if o != "solved"]
    if failures:
        lines += ["## Non-passing rollouts", ""]
        for r, o in failures:
            head = f"- `{r['unit']}` — **{o}**"
            ag = r.get("agent") or {}
            if ag.get("exit_status"):
                head += f", exit `{ag['exit_status']}`"
            exc = r.get("exception") or {}
            if exc.get("type"):
                first = str(exc.get("message") or "").strip().splitlines()
                head += f", `{exc['type']}`" + (f": {first[0][:200]}" if first else "")
            rw = r.get("rewards") or {}
            if rw.get("f2p_total") is not None:
                head += f" (f2p {rw.get('f2p_passed')}/{rw.get('f2p_total')}, p2p {rw.get('p2p_passed')}/{rw.get('p2p_total')})"
            rep = ag.get("repeats") or {}
            if rep.get("longest_streak", 0) >= LOOP_STREAK:
                head += f"; **repeat streak of {rep['longest_streak']} steps from step {rep['streak_starts_at']}**"
            if r.get("logs"):
                head += f" — `results-shard-{r['shard']}/{r['logs']}`"
            lines.append(head)
            lines += [f"  - `{test['name'][:160]}`" for test in (r.get("failed_tests") or [])[:3]]
        lines.append("")

    run_id = str(meta.get("run_url", "")).rstrip("/").split("/")[-1]
    lines += ["## Digging in", "",
              f"- logs and trajectories: `gh run download {run_id or '<run id>'} -n results-shard-<N>`",
              "- ask a provider for one step of any trajectory, no container: "
              "`python3 -m dswe.probe <provider:model> <logs>/trajectory.json --steps K`",
              f"- resume these rollouts elsewhere: `prefix=trials=run:{run_id or '<run id>'}/<unit> steps=50%`",
              ""]

    summary = "\n".join(lines)
    (args.out / "summary.md").write_text(summary)
    results["records"] = [{**r, "owner": o} for r, o in zip(records, owners)]
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
