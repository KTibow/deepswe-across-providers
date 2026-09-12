#!/usr/bin/env bash
# Regrade this shard's recorded rollouts: fetch each submission patch, replay it
# into the task environment, let the benchmark's own verifier grade it.
set -uo pipefail

: "${TRIALS:?TRIALS must be set}"
PLAN="${PLAN:-trial-plan.json}"
TASKS_DIR="${TASKS_DIR:-deep-swe/tasks}"
TASK_TIMEOUT="${TASK_TIMEOUT:-3600}"
ATTEMPTS="${ATTEMPTS:-1}"
DISK_FLOOR_GB="${DISK_FLOOR_GB:-25}"

mkdir -p out jobs patches
: > out/status.jsonl

avail_gb() { df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

# Stagger against sibling shards: a whole matrix starting at once is what
# trips the registry rate limit in the first place.
if [ -n "${STAGGER_MAX_SEC:-}" ] && [ "${STAGGER_MAX_SEC}" -gt 0 ]; then
  sleep $(( RANDOM % STAGGER_MAX_SEC ))
fi

TRIALS="${TRIALS}" PLAN="${PLAN}" python3 - > out/shard.tsv <<'PY'
import json, os
plan = json.load(open(os.environ["PLAN"]))
want = set(os.environ["TRIALS"].split())
for trial in plan["trials"]:
    if trial["trial_name"] in want:
        print("\t".join([trial["trial_name"], trial["task_name"], trial["patch_url"]]))
PY

while IFS=$'\t' read -r trial task patch_url; do
  [ -n "${trial:-}" ] || continue
  echo "::group::${trial}"
  echo "task=${task}"
  start=$(date +%s)

  bash scripts/pull_image.sh "${TASKS_DIR}/${task}" || true

  patch="${PWD}/patches/${trial}.patch"
  http=$(curl -sSL -o "${patch}" -w '%{http_code}' "${patch_url}")
  bytes=$(stat -c%s "${patch}" 2>/dev/null || echo 0)
  echo "patch: HTTP ${http}, ${bytes} bytes"

  rc=0
  if [ "${http}" != "200" ] || [ "${bytes}" -eq 0 ]; then
    echo "skipping: submission patch unavailable"
    rc=70
  else
    PYTHONPATH="${PWD}/scripts" timeout --signal=KILL "${TASK_TIMEOUT}" \
      pier run \
        --path "${TASKS_DIR}/${task}" \
        --agent-import-path patch_agent:PatchAgent \
        --ak "patch_path=${patch}" \
        --jobs-dir jobs \
        --job-name "${trial}" \
        --n-attempts "${ATTEMPTS}" \
        --n-concurrent 1 \
        --yes \
        --quiet \
      > "out/${trial}.log" 2>&1
    rc=$?
    echo "--- pier exit=${rc} (tail) ---"
    tail -c 3000 "out/${trial}.log"
  fi

  elapsed=$(( $(date +%s) - start ))
  TASK="${trial}" RC="${rc}" SECS="${elapsed}" HTTP="${http}" BYTES="${bytes}" \
  python3 - >> out/status.jsonl <<'PY'
import json, os
name = os.environ["TASK"]
try:
    tail = open(f"out/{name}.log", errors="replace").read()[-2000:]
except OSError:
    tail = ""
print(json.dumps({
    "task": name, "rc": int(os.environ["RC"]), "seconds": int(os.environ["SECS"]),
    "patch_http": os.environ["HTTP"], "patch_bytes": int(os.environ["BYTES"]),
    "tail": tail,
}))
PY

  echo "::endgroup::"
  if [ "$(avail_gb)" -lt "${DISK_FLOOR_GB}" ]; then
    docker system prune -af --volumes >/dev/null 2>&1 || true
  fi
done < out/shard.tsv

exit 0
