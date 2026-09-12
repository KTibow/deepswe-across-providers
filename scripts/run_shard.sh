#!/usr/bin/env bash
# Replay one shard's tasks, one pier job per task so a single bad task can't
# take the rest of the shard down with it.
set -uo pipefail

: "${TASKS:?TASKS must be set}"
AGENT="${AGENT:-oracle}"
ATTEMPTS="${ATTEMPTS:-1}"
TASK_TIMEOUT="${TASK_TIMEOUT:-3600}"
TASKS_DIR="${TASKS_DIR:-deep-swe/tasks}"
DISK_FLOOR_GB="${DISK_FLOOR_GB:-25}"

mkdir -p out jobs
: > out/status.jsonl

avail_gb() { df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

for task in $TASKS; do
  echo "::group::${task}"
  echo "disk before: $(avail_gb)G free"
  start=$(date +%s)
  timeout --signal=KILL "${TASK_TIMEOUT}" \
    pier run \
      --path "${TASKS_DIR}/${task}" \
      --agent "${AGENT}" \
      --jobs-dir jobs \
      --job-name "${task}" \
      --n-attempts "${ATTEMPTS}" \
      --n-concurrent 1 \
      --yes \
      --quiet \
    > "out/${task}.log" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - start ))
  echo "--- pier exit=${rc} in ${elapsed}s (tail of log) ---"
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

  # Task images are multi-GB and every task pulls its own; reclaim before the
  # runner's disk runs out.
  if [ "$(avail_gb)" -lt "${DISK_FLOOR_GB}" ]; then
    echo "reclaiming docker disk"
    docker system prune -af --volumes >/dev/null 2>&1 || true
    echo "disk after prune: $(avail_gb)G free"
  fi
done

exit 0
