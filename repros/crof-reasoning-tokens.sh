#!/usr/bin/env bash
# crof.ai glm-5.3 reports a usage.reasoning_tokens that isn't the number of
# reasoning tokens it streamed (0.77x-1.10x in testing), and that can exceed
# usage.completion_tokens when the rest of the reply is short.
#
# Two small streamed requests (~150-320 prompt tokens): one with a tool, one
# without. For each it prints the usage block next to what actually came back:
# the number of reasoning / content deltas (crof streams one token per delta,
# checked against the GLM-5.3 tokenizer) and character counts. The overcount
# varies, so a run doesn't always cross completion_tokens; run it again.
#
#   CROF_KEY=... bash repros/crof-reasoning-tokens.sh
#
# Found 2026-09-12; see NOTES.md, "crof.ai".
set -euo pipefail
KEY=${CROF_KEY:-$(tr -d '\n' < /tmp/crof-key.txt)}
OUT=${OUT:-${TMPDIR:-/tmp}/crof-reasoning-tokens/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"

TASK='In a Go project, error messages print as "ERROR: msg" but should print "ERROR: TypeError: msg". Before acting, think carefully: list at least five different grep strategies for finding where that prefix is added, and for each, the false positives it would hit.'

body() {
  python3 - "$1" "$TASK" <<'PY'
import json, sys
mode, task = sys.argv[1], sys.argv[2]
b = {"model": "glm-5.3", "stream": True, "stream_options": {"include_usage": True},
     "reasoning_effort": "max", "thinking": {"type": "enabled", "clear_thinking": False}}
if mode == "tools":
    b["messages"] = [{"role": "system", "content": "You are a helpful assistant that can interact with a computer."},
                     {"role": "user", "content": task + " Then call the bash tool exactly once with the command `ls`. Write no prose outside the tool call."}]
    b["tools"] = [{"type": "function", "function": {"name": "bash", "description": "Execute a bash command",
                   "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute"}}, "required": ["command"]}}}]
else:
    b["messages"] = [{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": task + " Then reply with only the single word: ls"}]
print(json.dumps(b))
PY
}

for mode in tools notools; do
  body "$mode" > "$OUT/$mode.req.json"
  curl -sS -N https://crof.ai/v1/chat/completions \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    --data @"$OUT/$mode.req.json" > "$OUT/$mode.sse"
  python3 - "$OUT/$mode.sse" "$mode" <<'PY'
import json, sys
path, mode = sys.argv[1], sys.argv[2]
chunks, usage_chunks = [], []
for line in open(path, encoding="utf-8"):
    if line.startswith("data:") and line[5:].strip() != "[DONE]":
        c = json.loads(line[5:])
        if "usage" in c:
            usage_chunks.append(len(chunks))
        chunks.append(c)
if not chunks:
    sys.exit(f"[{mode}] no SSE data: {open(path).read()[:300]}")
r = c = a = ""
nr = nc = 0
finish = None
for ch in chunks:
    for choice in ch.get("choices") or []:
        d = choice.get("delta") or {}
        if d.get("reasoning_content"):
            r += d["reasoning_content"]; nr += 1
        if d.get("content"):
            c += d["content"]; nc += 1
        for tc in d.get("tool_calls") or []:
            a += (tc.get("function") or {}).get("arguments") or ""
        finish = choice.get("finish_reason") or finish
u = chunks[usage_chunks[-1]]["usage"] if usage_chunks else {}
ct, rt = u.get("completion_tokens"), u.get("reasoning_tokens")
print(f"[{mode}] finish={finish}  SSE chunks={len(chunks)}  chunks carrying usage={len(usage_chunks)} (index {usage_chunks})")
print(f"  usage: prompt_tokens={u.get('prompt_tokens')} completion_tokens={ct} reasoning_tokens={rt}")
print(f"  streamed: reasoning deltas={nr} ({len(r)} chars, ~{len(r)/3.5:.0f} tok at 3.5 chars/tok)  "
      f"content deltas={nc} ({len(c)} chars: {c[:20]!r})  tool-call args={len(a)} chars: {a[:40]!r}")
if ct is None or rt is None:
    print("  (no reasoning_tokens in usage)")
    sys.exit()
print(f"  completion_tokens - streamed deltas = {ct - nr - nc:+d}   (small: template tokens like </think>, EOS, tool-call tags)")
print(f"  reasoning_tokens - reasoning deltas = {rt - nr:+d}  ({rt / nr:.3f}x)")
print("  RESULT: " + ("reasoning_tokens > completion_tokens  <-- impossible for a subset count"
                      if rt > ct else f"reasoning_tokens <= completion_tokens this time (margin {ct - rt})"))
PY
done
echo "raw SSE and request bodies: $OUT"
