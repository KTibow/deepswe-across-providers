"""Turn a run's collected records into the report a provider is judged by.

    python3 -m dswe.report --plan plan.json --results-dir shards --meta meta.json --out report

Writes ``summary.md`` (also the job summary) and ``results.json``. The summary
answers, in order:

1. **Verdict** — did the provider solve as many rollouts as the published
   reference config would be expected to on these tasks?
2. **Who owns each failure** — the model, the provider (a call that failed
   after retries, or format errors), the clock (agent timeout), or our
   harness (infrastructure, grading), which is excluded from the score.
3. **Inference health** — retries, finish reasons, format errors, empty
   replies, mis-decoded characters, per-call latency and tokens, and how
   steps, tokens and time compare with the reference config's rollouts on
   the same tasks.
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

from dswe import published

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


def med(values: list[float]) -> float | None:
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def quantile(values: list[float], q: float) -> float | None:
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
    ref_pass = sum(expected[t]["passes"] for t in {r["task"] for r, _ in scored})
    ref_total = sum(expected[t]["attempts"] for t in {r["task"] for r, _ in scored})
    if z <= -2:
        call = f"**Below the reference.** {solved} solved where {exp:.1f} ± {1.96 * sd:.1f} was expected (z = {z:.1f})."
    elif z >= 2:
        call = f"**Above the reference.** {solved} solved where {exp:.1f} ± {1.96 * sd:.1f} was expected (z = {z:.1f})."
    else:
        call = f"**Consistent with the reference.** {solved} solved; {exp:.1f} ± {1.96 * sd:.1f} expected (z = {z:+.1f})."
    lines.append(call)
    lines.append("")
    no_provider = [(r, o) for r, o in scored if o in ("solved", "model")]
    lines.append(f"- solved **{solved}/{len(scored)} = {pct(solved, len(scored))}** of scored rollouts "
                 f"(provider errors, format errors and timeouts count as failures; "
                 f"{len(records) - len(scored)} infrastructure/grading failure(s) excluded)")
    lines.append(f"- `{config}` solved {ref_pass}/{ref_total} = {pct(ref_pass, ref_total)} of its published rollouts on these tasks; "
                 f"the expectation shrinks each task's rate toward all configs' rate on it, since four rollouts can't justify 100%")
    lines.append(f"- counting only rollouts that finished without provider trouble or timeout: "
                 f"{sum(1 for _, o in no_provider if o == 'solved')}/{len(no_provider)} = "
                 f"{pct(sum(1 for _, o in no_provider if o == 'solved'), len(no_provider))}")
    lines.append("")
    return lines, {"scored": len(scored), "solved": solved, "expected": exp, "sd": sd, "z": z,
                   "reference": [ref_pass, ref_total]}


def inference_lines(records: list[dict], ref_rows: list[dict]) -> tuple[list[str], dict]:
    inf = [(r.get("agent") or {}).get("inference") for r in records]
    inf = [i for i in inf if i and i.get("calls")]
    lines = ["## Inference health", ""]
    if not inf:
        return lines + ["No live model calls were recorded.", ""], {}
    calls = sum(i["calls"] for i in inf)
    fmt_errors = sum(i["format_errors"] for i in inf)
    retries: collections.Counter = collections.Counter()
    finish: collections.Counter = collections.Counter()
    for i in inf:
        retries.update(i.get("retries") or {})
        finish.update(i.get("finish_reasons") or {})
    pool = {k: [v for i in inf for v in i.get(k) or []] for k in ("latency_s", "output_tokens", "reasoning_tokens", "prompt_tokens", "cached_tokens")}
    summary = {
        "calls": calls, "format_errors": fmt_errors, "retries": dict(retries), "finish_reasons": dict(finish),
        "empty_replies": sum(i["empty_replies"] for i in inf), "c1_chars": sum(i["c1_chars"] for i in inf),
        **{f"{k}_p50": quantile(v, 0.5) for k, v in pool.items()},
        **{f"{k}_p90": quantile(v, 0.9) for k, v in pool.items()},
        "latency_s_max": max(pool["latency_s"]) if pool["latency_s"] else None,
        "cached_share": sum(pool["cached_tokens"]) / sum(pool["prompt_tokens"]) if sum(pool["prompt_tokens"]) else None,
    }
    lines.append(f"- live model calls: **{calls}** over {len(inf)} rollout(s)")
    lines.append(f"- retried calls: **{sum(retries.values())}**" + (" (" + ", ".join(f"`{k}` {v}" for k, v in retries.most_common()) + ")" if retries else ""))
    lines.append("- finish reasons: " + ", ".join(f"`{k}` {v}" for k, v in finish.most_common()))
    lines.append(f"- format errors: **{fmt_errors}** ({pct(fmt_errors, calls)} of calls); empty replies: **{summary['empty_replies']}**; "
                 f"C1 control characters in output: **{summary['c1_chars']}**")
    lines.append(f"- wait per call: p50 {fmt(summary['latency_s_p50'], 's', 1)}, p90 {fmt(summary['latency_s_p90'], 's', 1)}, max {fmt(summary['latency_s_max'], 's')}")
    lines.append(f"- output tokens per call: p50 {fmt(summary['output_tokens_p50'])}, p90 {fmt(summary['output_tokens_p90'])}"
                 + (f"; reasoning p50 {fmt(summary['reasoning_tokens_p50'])}" if pool["reasoning_tokens"] else ""))
    if summary["cached_share"] is not None:
        lines.append(f"- prompt tokens served from cache: {pct(summary['cached_share'], 1)}")
    lines.append("")

    live = [r for r in records if (r.get("agent") or {}).get("inference", {}).get("calls") and not (r.get("agent") or {}).get("replayed_steps")]
    if live and ref_rows:
        tasks = {r["task"] for r in live}
        ref = [r for r in ref_rows if r["task_name"] in tasks]
        ours = {
            "agent steps": [r["agent"].get("steps") for r in live],
            "output tokens": [sum(r["agent"]["inference"]["output_tokens"]) or (r.get("metrics") or {}).get("n_output_tokens") for r in live],
            "peak prompt tokens": [max(r["agent"]["inference"]["prompt_tokens"] or [0]) or None for r in live],
            "agent minutes": [((r.get("timings") or {}).get("agent_execution") or 0) / 60 or None for r in live],
            "cost (USD)": [(r.get("metrics") or {}).get("cost_usd") for r in live],
        }
        theirs = {
            "agent steps": [r.get("n_agent_steps") for r in ref],
            "output tokens": [r.get("n_output_tokens") for r in ref],
            "peak prompt tokens": [r.get("peak_context_tokens") for r in ref],
            "agent minutes": [(r.get("agent_duration_seconds") or 0) / 60 or None for r in ref],
            "cost (USD)": [r.get("cost_usd") for r in ref],
        }
        lines += ["Per rollout, against the reference config's published rollouts on the same tasks "
                  "(its timeout was 5400 s; cost is at each provider's own prices):", "",
                  "| median per rollout | this run | reference | ratio |", "| --- | ---: | ---: | ---: |"]
        for key in ours:
            a, b = med(ours[key]), med(theirs[key])
            digits = 2 if key.startswith("cost") else 0
            ratio = f"{a / b:.2f}" if a is not None and b else "-"
            lines.append(f"| {key} | {fmt(a, digits=digits)} | {fmt(b, digits=digits)} | {ratio} |")
        lines.append("")
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
        lines += ["## Expectation", "", f"`{agent}` must score reward {want} on every task: "
                  f"**{len(matched)}/{len(records)}** did.", ""]
        results["matched"] = [len(matched), len(records)]
    elif not has_prefix:
        vl, results["verdict"] = verdict_lines(records, owners, expected, config)
        lines += vl

    counts = collections.Counter(owners)
    lines += ["## Outcomes by owner", "", "| owner | rollouts | meaning |", "| --- | ---: | --- |"]
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
    for key in OWNER_MARK:
        if counts[key]:
            lines.append(f"| {key} | {counts[key]} | {meaning[key]} |")
    lines.append("")
    results["owners"] = dict(counts)

    if agent not in ("oracle", "nop", "replay"):
        il, results["inference"] = inference_lines(records, ref_rows)
        lines += il

    lines += ["## Per task", ""]
    header = "| task | ours |" + (" reference | expected | steps ours/ref | out tok ours/ref | agent min ours/ref |" if expected else " steps | agent min |")
    lines += [header, "| --- | --- |" + (" ---: |" * (5 if expected else 2))]
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
                  "| unit | recorded result | resumed at | ours | live steps | exit |", "| --- | --- | ---: | --- | ---: | --- |"]
        buckets: dict[str, list[str]] = collections.defaultdict(list)
        fidelity = []
        for r, o in zip(records, owners):
            prefix = units.get(r["unit"], {}).get("prefix") or {}
            if not prefix:
                continue
            steps, total = prefix["steps"], prefix["total_steps"]
            at = "end" if steps == -1 else f"{steps}/{total}"
            bucket = "full replay" if steps == -1 else f"{round(100 * steps / total) if total else 0}%"
            buckets[bucket].append(o)
            rec = prefix.get("recorded_reward")
            rec_text = "pass" if rec is not None and float(rec) >= 1 else ("fail" if rec is not None else "?")
            ag = r.get("agent") or {}
            lines.append(f"| `{r['unit']}` | {rec_text} ({prefix.get('recorded_config')}) | {at} | {o} | "
                         f"{(ag.get('steps') or 0) - (ag.get('replayed_steps') or 0)} | {ag.get('exit_status') or '-'} |")
            if steps == -1 and rec is not None and o in ("solved", "model"):
                fidelity.append((rec_text == "pass") == (o == "solved"))
        lines.append("")
        lines += ["| resumed at | solved | of |", "| --- | ---: | ---: |"]
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
                steps_diff = rp["steps_with_different_output"]
                lines.append(f"  - `{u}`: {len(steps_diff)} of {rp['steps_replayed']} steps, first at step {steps_diff[0]}")
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
                head += f", `{exc['type']}`: {str(exc.get('message', '')).splitlines()[0][:200] if exc.get('message') else ''}"
            rw = r.get("rewards") or {}
            if rw.get("f2p_total") is not None:
                head += f" (f2p {rw.get('f2p_passed')}/{rw.get('f2p_total')}, p2p {rw.get('p2p_passed')}/{rw.get('p2p_total')})"
            if r.get("logs"):
                head += f" — `results-shard-{r['shard']}/{r['logs']}`"
            lines.append(head)
            for test in (r.get("failed_tests") or [])[:3]:
                lines.append(f"  - `{test['name'][:160]}`")
        lines.append("")

    run_id = str(meta.get("run_url", "")).rstrip("/").split("/")[-1]
    lines += ["## Digging in", "",
              f"- logs and trajectories: `gh run download {run_id or '<run id>'} -n results-shard-<N>`",
              "- ask the provider for one step of any trajectory, no container: "
              "`python3 -m dswe.probe <provider:model> <logs>/trajectory.json --steps K`",
              "- resume a rollout from step K on another provider: dispatch `run.yml` with `prefix_trials`/`prefix_steps`",
              ""]

    summary = "\n".join(lines)
    (args.out / "summary.md").write_text(summary)
    results["records"] = [{**r, "owner": o} for r, o in zip(records, owners)]
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
