"""End-to-end check of dswe.replay_model through the real mini-swe-agent CLI.

Needs a Python with mini-swe-agent installed, pointed to by $MINI (the
`mini-swe-agent` executable). No API key, no container: commands run in a
temp dir and the "provider" is a local server that records what it's sent.

    MINI=/path/to/venv/bin/mini-swe-agent python3 tests/test_replay_model.py
"""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def tool_call(i: int, command: str) -> dict:
    return {"tool_call_id": f"call_{i}", "function_name": "bash", "arguments": {"command": command}}


TRAJECTORY = {
    "schema_version": "ATIF-v1.7",
    "agent": {"name": "mini-swe-agent", "model_name": "recorded/model"},
    "steps": [
        {"step_id": 1, "source": "system", "message": "recorded system prompt"},
        {"step_id": 2, "source": "user", "message": "recorded task prompt"},
        {"step_id": 3, "source": "agent", "message": "m0", "reasoning_content": "r0",
         "tool_calls": [tool_call(0, "echo one > f.txt && cat f.txt")],
         "observation": {"results": [{"content": "RECORDED-0"}]}},
        {"step_id": 4, "source": "agent", "message": "m1",
         "tool_calls": [tool_call(1, "cat f.txt"), tool_call(2, "echo two >> f.txt")],
         "observation": {"results": [{"content": "RECORDED-1"}, {"content": "RECORDED-2"}, {"content": "FORMAT-ERR"}]}},
        {"step_id": 5, "source": "agent", "message": "m2", "tool_calls": [tool_call(3, SUBMIT)],
         "observation": {"results": [{"content": ""}]}},
    ],
}


class Recorder(http.server.BaseHTTPRequestHandler):
    bodies: list[dict] = []

    def do_POST(self) -> None:
        Recorder.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        reply = {
            "id": "live", "object": "chat.completion", "created": 1, "model": "live",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "live turn",
                "tool_calls": [{"id": "live_1", "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": SUBMIT})}}]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        pass


def run_mini(workdir: Path, port: int, steps: int, observations: str) -> dict:
    traj_path = workdir / "prefix.json"
    traj_path.write_text(json.dumps(TRAJECTORY))
    out = workdir / "out.json"
    cmd = [
        os.environ["MINI"], "--yolo", "--model=openai/live", "--task=do the thing", f"--output={out}",
        "-c", "mini.yaml", "-c", "agent.cost_limit=0",
        "-c", f"model.model_kwargs.api_base=http://127.0.0.1:{port}/v1",
        "-c", "model.model_class=dswe.replay_model.ReplayModel",
        "-c", f"model.replay_path={traj_path}",
        "-c", f"model.replay_steps={steps}",
        "-c", f"model.replay_observations={observations}",
        "--exit-immediately",
    ]
    env = os.environ | {
        "PYTHONPATH": str(REPO), "OPENAI_API_KEY": "dummy", "MSWEA_CONFIGURED": "true",
        "MSWEA_COST_TRACKING": "ignore_errors", "MSWEA_SILENT_STARTUP": "1",
    }
    proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=120)
    if not out.exists():
        sys.exit(f"mini-swe-agent wrote no trajectory:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return {"stdout": proc.stdout, "traj": json.loads(out.read_text())}


def main() -> None:
    server = http.server.HTTPServer(("127.0.0.1", 0), Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    failures = []

    def check(name: str, ok: bool, detail: object = "") -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if not ok else ""))
        if not ok:
            failures.append(name)

    # Partial replay, recorded observations: the live model must see the recorded history.
    with tempfile.TemporaryDirectory() as tmp:
        Recorder.bodies = []
        result = run_mini(Path(tmp), port, steps=2, observations="recorded")
        check("partial: model called once after the prefix", len(Recorder.bodies) == 1, len(Recorder.bodies))
        sent = Recorder.bodies[0]["messages"] if Recorder.bodies else []
        roles = [m["role"] for m in sent]
        check("partial: history shape", roles == ["system", "user", "assistant", "tool", "assistant", "tool", "tool", "user"], roles)
        if len(sent) == 8:
            check("partial: replayed assistant text and reasoning", sent[2]["content"] == "m0" and sent[2].get("reasoning_content") == "r0", sent[2])
            check("partial: recorded tool output in history", [sent[3]["content"], sent[5]["content"], sent[6]["content"]] == ["RECORDED-0", "RECORDED-1", "RECORDED-2"], sent[3:7])
            check("partial: format-error followup kept", sent[7]["content"] == "FORMAT-ERR", sent[7])
            check("partial: tool call ids carried", sent[3]["tool_call_id"] == "call_0" and sent[6]["tool_call_id"] == "call_2", sent[3])
        check("partial: commands really ran", (Path(tmp) / "f.txt").read_text() == "one\ntwo\n")
        info = result["traj"]["info"]
        check("partial: submitted", info["exit_status"] == "Submitted", info["exit_status"])
        replay = info.get("replay") or {}
        check("partial: replay summary", replay.get("steps_replayed") == 2 and replay.get("steps_with_different_output") == [0, 1], replay)
        check("partial: prompt mismatch detected", replay.get("prompt_matches_recording") is False, replay)
        check("partial: differences labelled", replay.get("differences") == {"content": [0, 1]}, replay.get("differences"))
        check("partial: handoff logged", "[replay] handing off" in result["stdout"], result["stdout"][-500:])

    # Live observations: history holds what our commands printed.
    with tempfile.TemporaryDirectory() as tmp:
        Recorder.bodies = []
        run_mini(Path(tmp), port, steps=2, observations="live")
        sent = Recorder.bodies[0]["messages"] if Recorder.bodies else []
        check("live: our output in history", len(sent) == 7 and '"one\\n"' in str(sent[3]["content"]), sent[3:4])

    # Full replay: the model is never called and the recording's submit ends the run.
    with tempfile.TemporaryDirectory() as tmp:
        Recorder.bodies = []
        result = run_mini(Path(tmp), port, steps=-1, observations="recorded")
        check("full: model never called", Recorder.bodies == [], len(Recorder.bodies))
        check("full: submitted", result["traj"]["info"]["exit_status"] == "Submitted", result["traj"]["info"]["exit_status"])

    server.shutdown()
    if failures:
        sys.exit(f"{len(failures)} check(s) failed")
    print("all checks passed")


if __name__ == "__main__":
    main()
