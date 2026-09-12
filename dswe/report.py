"""Turn a run's collected records into the summary a provider is judged by.

    python3 -m dswe.report --plan plan.json --results-dir shards --meta meta.json --out report

Writes ``summary.md`` (also the job summary) and ``results.json``. The summary
is meant to be read by someone who didn't build this repo, so it says things
plainly, in this order:

1. **Result** — did the provider pass as many rollouts as the published run of
   the same model would be expected to on these tasks?
2. **What happened to each rollout** — passed; finished but failed the tests;
   gave up after API errors; gave up after unusable replies; ran out of time;
   or our own setup / the grader broke, which isn't counted.
3. **How the API behaved** — per model call, this run next to the published
   run's trajectories on the same tasks (unusable replies, garbled
   characters, tokens, cache, looping), plus what only this run records
   (retries, why replies ended, empty replies, wait times), and per rollout
   steps, tokens, minutes and cost.
4. **Per task**, then **continuing recorded rollouts** and **did the replay
   match the recording** when a run started from recordings, then **rollouts
   that didn't pass** with where their logs are.

Reference-solution (oracle) and empty-submission (nop) runs get a one-line
check instead of a result.
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

# A streak this long of steps repeating recent steps is called out as looping.
LOOP_STREAK = 5
PRIOR = 2.0

INFRASTRUCTURE = {
    "NoTrialResult", "EnvironmentStartTimeoutError", "HealthcheckError", "AgentSetupTimeoutError",
    "AddTestsDirError", "DownloadVerifierDirError", "MissingExtraError",
}
GRADING = {"VerifierTimeoutError", "VerifierOutputParseError", "RewardFileEmptyError", "RewardFileNotFoundError"}
AGENT_EXITS = {"Submitted", "LimitsExceeded", "TimeExceeded", "RepeatedFormatError"}

# outcome -> (mark in the per-task table, what it means, counted in the score?)
OUTCOMES = {
    "passed": ("P", "passed the task's tests", True),
    "failed tests": (".", "finished, but failed the task's tests", True),
    "api errors": ("X", "gave up: a model call still failed after 10 retries", True),
    "unusable replies": ("F", "gave up: three replies in a row had no usable command", True),
    "timed out": ("T", "ran out of agent time", True),
    "unclear": ("?", "no agent log to tell what happened", True),
    "setup broke": ("I", "our own setup broke before grading; not counted, rerun it", False),
    "grader broke": ("G", "the grader broke; not counted, rerun it", False),
}
NOT_COUNTED = {k for k, (_, _, counted) in OUTCOMES.items() if not counted}
TESTS_NOTE = ("*Target tests* are the ones the task adds, which fail before the change and must pass after "
              "(DeepSWE's fail-to-pass); *existing tests* must keep passing (pass-to-pass).")


def outcome(record: dict[str, Any]) -> str:
    """What happened to one rollout."""
    exc = (record.get("exception") or {}).get("type") or ""
    exit_status = (record.get("agent") or {}).get("exit_status") or ""
    if record["state"] == "resolved":
        return "passed"
    if record["state"] == "harness-failure" or exc in INFRASTRUCTURE:
        return "setup broke"
    if exc in GRADING:
        return "grader broke"
    if exc == "AgentTimeoutError":
        return "timed out"
    if exit_status == "RepeatedFormatError":
        return "unusable replies"
    if exit_status and exit_status not in AGENT_EXITS:
        # mini-swe-agent ends with the exception class once retries run out:
        # ServiceUnavailableError, RateLimitError, APIConnectionError, ...
        return "api errors"
    if record["state"] == "unresolved":
        return "failed tests"
    return "unclear"


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


def tests_text(rewards: dict[str, Any]) -> str:
    if rewards.get("f2p_total") is None:
        return ""
    text = f"{rewards.get('f2p_passed')} of {rewards.get('f2p_total')} target tests"
    if rewards.get("p2p_total") is not None:
        text += f", {rewards.get('p2p_passed')} of {rewards.get('p2p_total')} existing tests"
    return text


def run_title(plan: dict[str, Any]) -> str:
    agent = plan["agent"]
    profile = plan.get("profile") or {}
    if profile:
        return f"{profile['model']} on {profile['provider']}"
    return {
        "oracle": "Reference solutions (oracle)",
        "nop": "Empty submissions (nop)",
        "replay": "Replaying recorded rollouts, no model",
    }.get(agent, agent)


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
    return f"{loops} of {len(stats)} (longest run {max(s['longest_streak'] for s in stats)})"


def result_lines(records: list[dict], outcomes: list[str], expected: dict[str, dict], config: str | None, model: str) -> tuple[list[str], dict]:
    counted = [(r, o) for r, o in zip(records, outcomes) if o not in NOT_COUNTED]
    passed = sum(1 for _, o in counted if o == "passed")
    lines = ["## Result", ""]
    if not config or not counted:
        lines += ["There's no published run to compare against." if not config else "No rollout could be counted.", ""]
        return lines, {"counted": len(counted), "passed": passed}
    exp = sum(expected[r["task"]]["expected"] for r, _ in counted)
    sd = math.sqrt(sum(expected[r["task"]]["expected"] * (1 - expected[r["task"]]["expected"]) for r, _ in counted))
    z = (passed - exp) / sd if sd else 0.0
    low, high = max(0.0, exp - 1.96 * sd), min(float(len(counted)), exp + 1.96 * sd)
    tasks = {r["task"] for r, _ in counted}
    ref_pass = sum(expected[t]["passes"] for t in tasks)
    ref_total = sum(expected[t]["attempts"] for t in tasks)
    where = f"Passed **{passed} of {len(counted)}**, where the published {model} run would pass about {exp:.1f} (normal range {low:.1f}–{high:.1f})."
    if z <= -2:
        head = f"**Worse than the published {model} run.** {where} That is {-z:.1f} standard deviations low, which is unlikely to be chance."
    elif z >= 2:
        head = f"**Better than the published {model} run.** {where} That is {z:.1f} standard deviations high."
    else:
        head = f"**In line with the published {model} run.** {where}"
    finished = [o for _, o in counted if o in ("passed", "failed tests")]
    left_out = len(records) - len(counted)
    lines += [
        head, "",
        "- Rollouts that gave up (API errors, unusable replies) or ran out of time count as fails."
        + (f" {left_out} rollout(s) where our own setup or the grader broke are left out; rerun those." if left_out else ""),
        f"- The published run (`{config}`) passed {ref_pass} of {ref_total} attempts at these tasks. The expected count is a "
        "little lower than its raw rate, because four attempts per task is a small sample: each task's rate is pulled "
        "toward how every published model did on it.",
        f"- Counting only rollouts that finished on their own: passed {finished.count('passed')} of {len(finished)}.",
        "",
    ]
    return lines, {"counted": len(counted), "passed": passed, "expected": exp, "sd": sd, "z": z, "reference": [ref_pass, ref_total]}


def api_lines(records: list[dict], ref_rows: list[dict]) -> tuple[list[str], dict]:
    live_records = [r for r in records if (r.get("agent") or {}).get("inference", {}).get("calls")]
    inf = [r["agent"]["inference"] for r in live_records]
    lines = ["## How the API behaved", ""]
    if not inf:
        return lines + ["No model calls were recorded.", ""], {}
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
        "repeats": [r["agent"]["repeats"] for r in live_records if r["agent"].get("repeats")],
        **pool,
    }
    tasks = {r["task"] for r in live_records}
    ref = reference_calls([r for r in ref_rows if r["task_name"] in tasks]) if ref_rows else None

    ended = {"tool_calls": "asked to run a command", "stop": "stopped on its own", "length": "cut off by the output limit"}
    lines.append(f"- Model calls: **{ours['calls']}** across {ours['rollouts']} rollout(s). Calls that errored and were retried: "
                 f"**{sum(retries.values())}**" + (" (" + ", ".join(f"`{k}` {v}" for k, v in retries.most_common()) + ")" if retries else ""))
    lines.append("- Why replies ended: " + ", ".join(
        f"{ended.get(k, f'`{k}`')} {v}" for k, v in finish.most_common()))
    lines.append(f"- Empty replies (no text, no reasoning, no command): **{ours['empty_replies']}**")
    lines.append("")

    def per_call(side: dict[str, Any] | None) -> dict[str, str]:
        if not side or not side["calls"]:
            return {}
        cached = sum(side["cached_tokens"]) / sum(side["prompt_tokens"]) if sum(side["prompt_tokens"]) else None
        return {
            "replies with no usable command": f"{side['format_errors']} of {side['calls']} ({pct(side['format_errors'], side['calls'])})",
            "garbled characters per 100 calls (UTF-8 read as Latin-1)": f"{100 * side['c1_chars'] / side['calls']:.1f}",
            "output tokens, typical / 90th percentile": f"{fmt(quantile(side['output_tokens'], 0.5))} / {fmt(quantile(side['output_tokens'], 0.9))}",
            "prompt tokens, typical / largest": f"{fmt(quantile(side['prompt_tokens'], 0.5))} / {fmt(max(side['prompt_tokens']) if side['prompt_tokens'] else None)}",
            "prompt tokens served from cache": pct(cached, 1) if cached is not None else "-",
            "steps that repeat a recent step (looping)": repeat_share(side["repeats"]),
            f"rollouts that repeated {LOOP_STREAK}+ steps in a row": loop_count(side["repeats"]),
        }

    mine, theirs = per_call(ours), per_call(ref)
    ref_label = f"published run ({ref['rollouts']} rollouts)" if ref and ref["rollouts"] else "published run"
    lines += [f"| per model call | this run | {ref_label} |", "| --- | ---: | ---: |"]
    lines += [f"| {key} | {value} | {theirs.get(key, '-')} |" for key, value in mine.items()]
    lines.append(f"| wait for a reply, typical / 90th percentile / longest | {fmt(quantile(pool['latency_s'], 0.5), 's', 1)} / "
                 f"{fmt(quantile(pool['latency_s'], 0.9), 's', 1)} / {fmt(max(pool['latency_s']) if pool['latency_s'] else None, 's')} | not published |")
    lines += ["",
              "Compare rates, not counts: glm-5.3 writes some garbled characters even at Z.AI. A run of repeated steps "
              "can also be an agent waiting on a background job (`sleep 30; cat log`); the list of rollouts that didn't "
              "pass says where each run of repeats starts, which is the step to look at.", ""]

    fresh = [r for r in live_records if not r["agent"].get("replayed_steps")]
    if fresh and ref_rows:
        ref_fresh = [r for r in ref_rows if r["task_name"] in {r["task"] for r in fresh}]
        rows = {
            "steps": ([r["agent"].get("steps") for r in fresh], [r.get("n_agent_steps") for r in ref_fresh], 0),
            "output tokens": ([sum(r["agent"]["inference"]["output_tokens"]) or None for r in fresh], [r.get("n_output_tokens") for r in ref_fresh], 0),
            "largest prompt, tokens": ([max(r["agent"]["inference"]["prompt_tokens"] or [0]) or None for r in fresh], [r.get("peak_context_tokens") for r in ref_fresh], 0),
            "minutes of agent time": ([((r.get("timings") or {}).get("agent_execution") or 0) / 60 or None for r in fresh],
                                      [(r.get("agent_duration_seconds") or 0) / 60 or None for r in ref_fresh], 0),
            "cost, USD at each provider's prices": ([(r.get("metrics") or {}).get("cost_usd") for r in fresh], [r.get("cost_usd") for r in ref_fresh], 2),
        }
        lines += ["| per rollout, median | this run | published run | this ÷ published |", "| --- | ---: | ---: | ---: |"]
        for key, (a_values, b_values, digits) in rows.items():
            a, b = med(a_values), med(b_values)
            lines.append(f"| {key} | {fmt(a, digits=digits)} | {fmt(b, digits=digits)} | {f'{a / b:.2f}' if a is not None and b else '-'} |")
        lines += ["", "The published run had 90 minutes of agent time per rollout, the same limit as this run by default.", ""]

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
    outcomes = [outcome(r) for r in records]
    agent = plan["agent"]
    config = plan.get("reference_config")
    model = (plan.get("profile") or {}).get("model", "")
    expected, ref_rows = reference_stats(config, plan["tasks"])

    lines = [f"# {run_title(plan)}: {plan['selection']}", ""]
    if plan.get("tasks_from") == "rollouts" or plan["selection"] == "prefix(explicit)":
        # Named rollouts decide the tasks; the workflow's task input was ignored.
        meta.pop("tasks", None)
    if meta:
        lines += ["| setting | value |", "| --- | --- |"] + [f"| {k} | `{v}` |" for k, v in meta.items() if v not in ("", None)] + [""]
    missing = sorted(set(units) - {r["unit"] for r in records})
    if missing:
        lines += [f"**{len(missing)} planned rollout(s) produced no result at all:** " + ", ".join(f"`{u}`" for u in missing), ""]

    results: dict[str, Any] = {"meta": meta, "agent": agent, "selection": plan["selection"], "reference_config": config}
    has_prefix = any(u["prefix"] for u in units.values())

    if agent in ("oracle", "nop"):
        matched = [r for r in records if r.get("reward") is not None and float(r["reward"]) == (0 if agent == "nop" else 1)]
        should = "fail every task" if agent == "nop" else "pass every task"
        lines += ["## Result", "", f"These should {should}: **{len(matched)} of {len(records)}** did.", ""]
        results["matched"] = [len(matched), len(records)]
    elif not has_prefix:
        rl, results["result"] = result_lines(records, outcomes, expected, config, model)
        lines += rl

    counts = collections.Counter(outcomes)
    lines += ["## What happened to each rollout", "", "| outcome | rollouts | meaning |", "| --- | ---: | --- |"]
    lines += [f"| {key} | {counts[key]} | {meaning} |" for key, (_, meaning, _) in OUTCOMES.items() if counts[key]]
    lines.append("")
    results["outcomes"] = dict(counts)

    if agent not in ("oracle", "nop", "replay"):
        al, results["api"] = api_lines(records, ref_rows)
        lines += al

    lines += ["## Per task", ""]
    if expected:
        lines += ["| task | this run | published run passed | expected pass rate | steps, this / published | "
                  "output tokens, this / published | minutes, this / published |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    else:
        lines += ["| task | this run | steps | minutes |", "| --- | --- | ---: | ---: |"]
    by_task: dict[str, list[tuple[dict, str]]] = collections.defaultdict(list)
    for r, o in zip(records, outcomes):
        by_task[r["task"]].append((r, o))
    for task in plan["tasks"]:
        rs = by_task.get(task, [])
        marks = "".join(OUTCOMES[o][0] for _, o in rs) or "-"
        steps = med([(r.get("agent") or {}).get("steps") for r, _ in rs])
        minutes = med([((r.get("timings") or {}).get("agent_execution") or 0) / 60 or None for r, _ in rs])
        out_tok = med([sum(((r.get("agent") or {}).get("inference") or {}).get("output_tokens") or []) or None for r, _ in rs])
        if expected:
            e = expected[task]
            lines.append(f"| `{task}` | `{marks}` | {e['passes']} of {e['attempts']} | {e['expected']:.2f} | "
                         f"{fmt(steps)} / {fmt(e['steps'])} | {fmt(out_tok, 'k')} / {fmt(e['output_tokens'], 'k')} | "
                         f"{fmt(minutes)} / {fmt(e['minutes'])} |")
        else:
            lines.append(f"| `{task}` | `{marks}` | {fmt(steps)} | {fmt(minutes)} |")
    lines += ["", "One mark per rollout: " + ", ".join(f"`{mark}` {key}" for key, (mark, _, _) in OUTCOMES.items()) + ".", ""]

    if has_prefix:
        lines += ["## Continuing recorded rollouts", "",
                  "Each of these replayed a recorded rollout's first steps, running its commands for real, "
                  "then let the model take over (or replayed it to the end with no model).", "",
                  "| rollout | the recording | continued from step | this run | tests this run | new steps | how the agent stopped |",
                  "| --- | --- | ---: | --- | ---: | ---: | --- |"]
        buckets: dict[str, list[str]] = collections.defaultdict(list)
        same_grade = []
        for r, o in zip(records, outcomes):
            prefix = units.get(r["unit"], {}).get("prefix") or {}
            if not prefix:
                continue
            steps, total = prefix["steps"], prefix["total_steps"]
            bucket = "replayed to the end" if steps == -1 else f"{round(100 * steps / total) if total else 0}% through"
            buckets[bucket].append(o)
            rec = prefix.get("recorded_reward")
            rec_f2p = prefix.get("recorded_f2p") or [None, None]
            recording = "?" if rec is None else ("passed" if float(rec) >= 1 else "failed")
            if rec_f2p[1] is not None:
                recording += f", {rec_f2p[0]} of {rec_f2p[1]} target tests"
            ag = r.get("agent") or {}
            stopped = {"Submitted": "submitted"}.get(ag.get("exit_status"), f"`{ag['exit_status']}`" if ag.get("exit_status") else "-")
            rw = r.get("rewards") or {}
            ours_tests = f"{rw.get('f2p_passed')} of {rw.get('f2p_total')}" if rw.get("f2p_total") is not None else "-"
            lines.append(f"| `{r['unit']}` | {recording} ({prefix.get('recorded_config')}) | "
                         f"{'end' if steps == -1 else f'{steps} of {total}'} | {o} | {ours_tests} | "
                         f"{(ag.get('steps') or 0) - (ag.get('replayed_steps') or 0)} | {stopped} |")
            if steps == -1 and rec is not None and o in ("passed", "failed tests"):
                same_grade.append((float(rec) >= 1) == (o == "passed"))
        lines += ["", TESTS_NOTE, "", "| continued from | passed | of |", "| --- | ---: | ---: |"]
        for bucket in sorted(buckets, key=lambda b: (b == "replayed to the end", float(b.split("%")[0]) if "%" in b else 0)):
            counted = [o for o in buckets[bucket] if o not in NOT_COUNTED]
            lines.append(f"| {bucket} | {counted.count('passed')} | {len(counted)} |")
        lines.append("")
        replays = [(r["unit"], (r.get("agent") or {}).get("replay")) for r in records if (r.get("agent") or {}).get("replay")]
        if replays:
            lines += ["## Did the replay match the recording?", ""]
            if same_grade:
                lines.append(f"- Replays run to the end that got the same grade as the recording: **{sum(same_grade)} of {len(same_grade)}**")
            diverged = [(u, rp) for u, rp in replays if rp.get("steps_with_different_output")]
            lines.append(f"- Replays where some command printed something different from the recording: {len(diverged)} of {len(replays)}. "
                         "Some of this is normal: timings, file sizes, dates, and files listed in a different order.")
            names = {"numbers": "only numbers differ", "order": "same lines, different order", "content": "different content"}
            for u, rp in diverged[:20]:
                diff = rp["steps_with_different_output"]
                kinds = rp.get("differences") or {}
                detail = "; ".join(f"{names.get(k, k)}: {len(s)}" for k, s in sorted(kinds.items()))
                first = f". First step with different content: {kinds['content'][0]}" if kinds.get("content") else ""
                lines.append(f"  - `{u}`: {len(diff)} of {rp['steps_replayed']} steps"
                             + (f" ({detail}){first}" if detail else f", starting at step {diff[0]}"))
            prompt_diff = sum(1 for _, rp in replays if rp.get("prompt_matches_recording") is False)
            if prompt_diff:
                lines.append(f"- The task prompt differed from the recording in {prompt_diff} replay(s). That's expected: "
                             "the prompt includes a line describing the machine it runs on.")
            lines.append("")

    failures = [(r, o) for r, o in zip(records, outcomes) if o != "passed"]
    if failures:
        lines += ["## Rollouts that didn't pass", ""]
        for r, o in failures:
            ag = r.get("agent") or {}
            head = f"- `{r['unit']}`: {OUTCOMES[o][1]}"
            tests = tests_text(r.get("rewards") or {})
            if tests:
                head += f" ({tests})"
            exc = r.get("exception") or {}
            if o == "api errors" and ag.get("exit_status"):
                head += f". Last error: `{ag['exit_status']}`"
            elif exc.get("type"):
                first = str(exc.get("message") or "").strip().splitlines()
                head += f". Error: `{exc['type']}`" + (f" {first[0][:200]}" if first else "")
            rep = ag.get("repeats") or {}
            if rep.get("longest_streak", 0) >= LOOP_STREAK:
                head += f". **Repeated itself for {rep['longest_streak']} steps starting at step {rep['streak_starts_at']}**"
            if r.get("logs"):
                head += f". Logs: `results-shard-{r['shard']}/{r['logs']}`"
            lines.append(head)
            lines += [f"  - failing: `{test['name'][:160]}`" for test in (r.get("failed_tests") or [])[:3]]
        lines.append("")
        if TESTS_NOTE not in lines:
            lines += [TESTS_NOTE, ""]

    run_id = str(meta.get("run_url", "")).rstrip("/").split("/")[-1] or "<run id>"
    lines += ["## Digging further", "",
              f"- Download a shard's logs and full trajectories: `gh run download {run_id} -n results-shard-<N>`",
              "- Ask a provider for just one step of a trajectory, no container needed: "
              "`python3 -m dswe.probe <provider:model> <logs>/trajectory.json --steps <step>`",
              f"- Continue one of these rollouts from some step, on any provider: run `run.yml` with "
              f"`prefix=trials=run:{run_id}/<rollout> steps=<step>`",
              ""]

    summary = "\n".join(lines)
    (args.out / "summary.md").write_text(summary)
    results["records"] = [{**r, "outcome": o} for r, o in zip(records, outcomes)]
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
