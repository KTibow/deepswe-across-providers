"""collect + report on a synthetic pier job directory. Stdlib only.

Builds two fake trials the way pier and mini-swe-agent lay them out — one
solved with a retried call and a format error, one that died on a provider
error — runs dswe.collect and dswe.report over them, and checks the numbers
the report is supposed to surface. Uses the cached published tables for the
reference expectation.

    python3 tests/test_collect_report.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def response(prompt: int, output: int, cached: int, finish: str = "tool_calls", content: str = "", tool: bool = True) -> dict:
    message = {"role": "assistant", "content": content, "reasoning_content": "thinking"}
    if tool:
        message["tool_calls"] = [{"id": "c", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]
    return {
        "choices": [{"finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": output, "prompt_tokens_details": {"cached_tokens": cached}},
    }


def write_trial(jobs: Path, unit: str, task: str, reward: float | None, messages: list[dict], exit_status: str,
                log: str, exception: dict | None = None) -> None:
    trial = jobs / unit / f"{task[:32]}__abc"
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir()
    result = {
        "task_name": f"datacurve/{task}", "trial_name": trial.name,
        "agent_execution": {"started_at": "2026-09-12T00:00:00Z", "finished_at": "2026-09-12T00:30:00Z"},
        "agent_result": {"n_output_tokens": 999, "cost_usd": 0.5},
        "exception_info": exception,
    }
    if reward is not None:
        result["verifier_result"] = {"rewards": {"reward": reward, "f2p_passed": 3, "f2p_total": 4, "p2p_passed": 9, "p2p_total": 9}}
    (trial / "result.json").write_text(json.dumps(result))
    (trial / "agent" / "mini-swe-agent.trajectory.json").write_text(json.dumps(
        {"info": {"exit_status": exit_status, "mini_version": "2.4.2"}, "messages": messages}))
    (trial / "agent" / "mini-swe-agent.txt").write_text(log)
    (trial / "agent" / "trajectory.json").write_text("{}")


def main() -> None:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: object = "") -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"))
        if not ok:
            failures.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        jobs = root / "jobs"
        solved_task, failed_task = "dasel-html-document-format", "ytt-jsonpath-query-api"
        solved = [
            {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "extra": {"response": response(1000, 50, 0), "timestamp": 100.0}},
            {"role": "tool", "content": "out", "extra": {"timestamp": 101.0}},
            {"role": "user", "content": "format error", "extra": {"interrupt_type": "FormatError",
                                                                  "response": response(1100, 20, 1000, finish="stop", tool=False, content="\u00e2\u0080\u0094")}},
            {"role": "assistant", "content": "", "extra": {"response": response(1200, 80, 1000), "timestamp": 130.0}},
            {"role": "tool", "content": "out", "extra": {"timestamp": 131.0}},
            {"role": "assistant", "content": "", "extra": {"response": response(1300, 30, 1200), "timestamp": 141.0}},
            {"role": "tool", "content": "out", "extra": {"timestamp": 142.0}},
            {"role": "exit", "content": "", "extra": {"exit_status": "Submitted"}},
        ]
        write_trial(jobs, solved_task, solved_task, 1.0, solved, "Submitted",
                    "Retrying <unknown> in 4 seconds as it raised RateLimitError: slow down\n")
        died = solved[:4] + [{"role": "exit", "content": "boom", "extra": {"exit_status": "ServiceUnavailableError"}}]
        write_trial(jobs, failed_task, failed_task, 0.0, died, "ServiceUnavailableError",
                    "Retrying <unknown> in 4 seconds as it raised ServiceUnavailableError: x\n" * 9)
        status = root / "status.jsonl"
        status.write_text("".join(json.dumps({"unit": u, "rc": 0, "seconds": 10, "tail": ""}) + "\n"
                                  for u in (solved_task, failed_task, "never-ran")))

        subprocess.run([sys.executable, "-m", "dswe.collect", "--jobs-dir", str(jobs), "--status", str(status),
                        "--shard", "0", "--out", str(root / "shards" / "results-shard-0" / "results")],
                       cwd=REPO, check=True, capture_output=True)
        shard = json.loads((root / "shards" / "results-shard-0" / "results" / "shard-000.json").read_text())
        recs = {r["unit"]: r for r in shard["records"]}
        inf = recs[solved_task]["agent"]["inference"]
        check("collect: calls include the format-error call", inf["calls"] == 4, inf["calls"])
        check("collect: steps exclude it", recs[solved_task]["agent"]["steps"] == 3, recs[solved_task]["agent"]["steps"])
        check("collect: latency skips the call after a format error", inf["latency_s"] == [10.0], inf["latency_s"])
        check("collect: retries counted", inf["retries"] == {"RateLimitError": 1}, inf["retries"])
        check("collect: C1 characters counted", inf["c1_chars"] == 2, inf["c1_chars"])
        check("collect: finish reasons", inf["finish_reasons"] == {"tool_calls": 3, "stop": 1}, inf["finish_reasons"])
        check("collect: missing unit becomes harness-failure", recs["never-ran"]["state"] == "harness-failure")
        check("collect: trajectory kept", (root / "shards" / "results-shard-0" / "results" / recs[solved_task]["logs"] / "trajectory.json").exists())

        plan = {
            "agent": "crof:glm-5.3", "reference_config": "mini_swe_agent_glm_5_3_max", "selection": "test",
            "tasks": sorted([solved_task, failed_task, "never-ran"]),
            "units": [{"id": t, "task": t, "attempts": 1, "prefix": None} for t in (solved_task, failed_task, "never-ran")],
            "shards": [],
        }
        (root / "plan.json").write_text(json.dumps(plan))
        # "never-ran" isn't a real task; give the report a reference only for the real two.
        plan["tasks"] = [solved_task, failed_task]
        (root / "plan.json").write_text(json.dumps(plan))
        out = subprocess.run([sys.executable, "-m", "dswe.report", "--plan", str(root / "plan.json"),
                              "--results-dir", str(root / "shards"), "--out", str(root / "report")],
                             cwd=REPO, capture_output=True, text=True)
        check("report: exits 0", out.returncode == 0, out.stderr[-2000:])
        summary = (root / "report" / "summary.md").read_text() if out.returncode == 0 else ""
        results = json.loads((root / "report" / "results.json").read_text()) if out.returncode == 0 else {}
        check("report: owners", results.get("owners") == {"solved": 1, "provider": 1, "infrastructure": 1}, results.get("owners"))
        verdict = results.get("verdict") or {}
        check("report: infrastructure excluded from score", verdict.get("scored") == 2 and verdict.get("solved") == 1, verdict)
        check("report: expectation from reference", 1.5 < (verdict.get("expected") or 0) < 2.0, verdict.get("expected"))
        check("report: retries surfaced", "`ServiceUnavailableError` 9" in summary and "`RateLimitError` 1" in summary)
        check("report: format errors surfaced", "format errors: **1**" in summary)
        check("report: per-task marks", f"| `{solved_task}` | `P` |" in summary and f"| `{failed_task}` | `X` |" in summary)
        if failures:
            print(summary)

    if failures:
        sys.exit(f"{len(failures)} check(s) failed")
    print("all checks passed")


if __name__ == "__main__":
    main()
