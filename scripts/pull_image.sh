#!/usr/bin/env bash
# Pull a task's prebuilt image, with backoff.
#
# public.ecr.aws rate-limits anonymous pulls per source IP, and a wide matrix of
# GitHub runners trips it: "toomanyrequests: Rate exceeded" surfaces as a docker
# compose failure mid-trial, which looks like a task failure unless you read the
# log. Pulling up front, with retries, keeps an infrastructure hiccup from being
# scored as a model result.
set -uo pipefail

task_dir="${1:?usage: pull_image.sh <task dir>}"
attempts="${PULL_ATTEMPTS:-6}"

image=$(sed -n 's/^[[:space:]]*docker_image[[:space:]]*=[[:space:]]*"\(.*\)"[[:space:]]*$/\1/p' \
        "${task_dir}/task.toml" | head -1)
if [ -z "${image}" ]; then
  echo "[pull] no prebuilt image declared in ${task_dir}/task.toml; pier will build"
  exit 0
fi

if docker image inspect "${image}" >/dev/null 2>&1; then
  echo "[pull] already local: ${image}"
  exit 0
fi

delay=10
for attempt in $(seq 1 "${attempts}"); do
  if docker pull --quiet "${image}" >/dev/null; then
    echo "[pull] ok on attempt ${attempt}: ${image}"
    exit 0
  fi
  if [ "${attempt}" -eq "${attempts}" ]; then break; fi
  wait_for=$(( delay + RANDOM % 15 ))
  echo "[pull] attempt ${attempt} failed, retrying in ${wait_for}s"
  sleep "${wait_for}"
  delay=$(( delay * 2 ))
  if [ "${delay}" -gt 120 ]; then delay=120; fi
done

echo "[pull] could not pull ${image} after ${attempts} attempts" >&2
exit 1
