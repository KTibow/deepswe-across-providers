#!/usr/bin/env bash
# Run one shard's units from plan.json, one pier job per unit, so a single bad
# task can't take the rest of the shard down with it.
set -uo pipefail

: "${UNITS:?UNITS must be set}"
PLAN="${PLAN:-plan.json}"
TASKS_DIR="${TASKS_DIR:-deep-swe/tasks}"
TASK_TIMEOUT="${TASK_TIMEOUT:-12600}"
DISK_FLOOR_GB="${DISK_FLOOR_GB:-25}"
# Tasks declare 2 CPUs / 8 GB; a 4-vCPU runner fits two trials at once.
CONCURRENCY="${CONCURRENCY:-1}"
# pier imports dswe.agent from here.
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p out jobs
: > out/status.jsonl

avail_gb() { df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

# Stagger against sibling shards: a whole matrix pulling multi-GB images at
# once is what trips public.ecr.aws's rate limit.
if [ -n "${STAGGER_MAX_SEC:-}" ] && [ "${STAGGER_MAX_SEC}" -gt 0 ]; then
  sleep $(( RANDOM % STAGGER_MAX_SEC ))
fi

record_status() {
  UNIT="$1" RC="$2" SECS="$3" python3 - >> out/status.jsonl <<'PY'
import json, os
unit = os.environ["UNIT"]
try:
    tail = open(f"out/{unit}.log", errors="replace").read()[-2000:]
except OSError:
    tail = ""
print(json.dumps({"unit": unit, "rc": int(os.environ["RC"]), "seconds": int(os.environ["SECS"]), "tail": tail}))
PY
}

for unit in $UNITS; do
  echo "::group::${unit}"
  start=$(date +%s)
  task=$(python3 -m dswe.plan task "${PLAN}" "${unit}")
  if ! python3 -m dswe.plan args "${PLAN}" "${unit}" > "out/${unit}.args" 2> "out/${unit}.log"; then
    cat "out/${unit}.log"
    record_status "${unit}" 2 0
    echo "::endgroup::"
    continue
  fi
  mapfile -t agent_args < "out/${unit}.args"
  if [ -n "${AGENT_TIMEOUT_MULTIPLIER:-}" ]; then
    agent_args+=(--agent-timeout-multiplier "${AGENT_TIMEOUT_MULTIPLIER}")
  fi

  bash scripts/pull_image.sh "${TASKS_DIR}/${task}" || true
  timeout --signal=KILL "${TASK_TIMEOUT}" \
    pier run \
      --path "${TASKS_DIR}/${task}" \
      "${agent_args[@]}" \
      --jobs-dir jobs \
      --job-name "${unit}" \
      --n-concurrent "${CONCURRENCY}" \
      --yes \
      --quiet \
    2>&1 | tee "out/${unit}.log"
  rc=${PIPESTATUS[0]}
  elapsed=$(( $(date +%s) - start ))
  echo "--- pier exit=${rc} in ${elapsed}s ---"
  record_status "${unit}" "${rc}" "${elapsed}"
  echo "::endgroup::"

  # Task images are multi-GB and every task pulls its own.
  if [ "$(avail_gb)" -lt "${DISK_FLOOR_GB}" ]; then
    docker system prune -af --volumes >/dev/null 2>&1 || true
  fi
done

exit 0
