#!/usr/bin/env bash
# Run a real model through mini-swe-agent on this shard's tasks.
# The API key is passed as a ${VAR} template so pier resolves it from this
# process's environment — it never lands in a command line or a config file.
set -uo pipefail

: "${TASKS:?TASKS must be set}"
: "${MODEL:?MODEL must be set}"
ATTEMPTS="${ATTEMPTS:-1}"
TASK_TIMEOUT="${TASK_TIMEOUT:-10800}"
TASKS_DIR="${TASKS_DIR:-deep-swe/tasks}"
KEY_VAR="${KEY_VAR:-MSWEA_API_KEY}"
API_BASE="${API_BASE:-}"
EXTRA_AGENT_ENV="${EXTRA_AGENT_ENV:-}"
EXTRA_AGENT_KWARGS="${EXTRA_AGENT_KWARGS:-}"
DISK_FLOOR_GB="${DISK_FLOOR_GB:-25}"

if [ -z "${PROVIDER_API_KEY:-}" ]; then
  echo "PROVIDER_API_KEY is empty — set the repository secret before dispatching" >&2
  exit 2
fi

mkdir -p out jobs
: > out/status.jsonl

avail_gb() { df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

args=(--agent mini-swe-agent --model "${MODEL}" --ae "${KEY_VAR}=\${PROVIDER_API_KEY}")
if [ -n "${API_BASE}" ]; then
  args+=(--ae "OPENAI_BASE_URL=${API_BASE}" --ae "OPENAI_API_BASE=${API_BASE}")
fi
for kv in ${EXTRA_AGENT_ENV}; do args+=(--ae "${kv}"); done
for kv in ${EXTRA_AGENT_KWARGS}; do args+=(--ak "${kv}"); done

echo "model=${MODEL} attempts=${ATTEMPTS} base=${API_BASE:-<provider default>} key_var=${KEY_VAR}"

for task in $TASKS; do
  echo "::group::${task}"
  start=$(date +%s)
  timeout --signal=KILL "${TASK_TIMEOUT}" \
    pier run \
      --path "${TASKS_DIR}/${task}" \
      "${args[@]}" \
      --jobs-dir jobs \
      --job-name "${task}" \
      --n-attempts "${ATTEMPTS}" \
      --n-concurrent 1 \
      --yes \
      --quiet \
    > "out/${task}.log" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - start ))
  echo "--- pier exit=${rc} in ${elapsed}s (tail) ---"
  tail -c 3000 "out/${task}.log"

  TASK="${task}" RC="${rc}" SECS="${elapsed}" python3 - >> out/status.jsonl <<'PY'
import json, os
task, rc, secs = os.environ["TASK"], int(os.environ["RC"]), int(os.environ["SECS"])
try:
    tail = open(f"out/{task}.log", errors="replace").read()[-2000:]
except OSError:
    tail = ""
print(json.dumps({"task": task, "rc": rc, "seconds": secs, "tail": tail}))
PY

  echo "::endgroup::"
  if [ "$(avail_gb)" -lt "${DISK_FLOOR_GB}" ]; then
    docker system prune -af --volumes >/dev/null 2>&1 || true
  fi
done

exit 0
