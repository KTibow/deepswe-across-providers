# Working notes

Things that cost time to find out. Written for two jobs: running the benchmark,
and reusing the rollouts DeepSWE published.

## Where the published data lives

None of this is linked from the repo or the paper; it comes from the SSR payload
on a trial detail page (`https://deepswe.datacurve.ai/data/v1.1/trials/<trial>`,
search the inline `$_TSR.router` script for `artifact_patterns`).

**Aggregate tables** — `https://deepswe.datacurve.ai/artifacts/<release>/<name>.json`:

| name | what |
| --- | --- |
| `trials` | every rollout: model, config, reward, f2p/p2p counts, tokens, cost, error category (51 MB for v1.1, 31,617 rows) |
| `tasks` | the 113 tasks: id, language, repo, base commit |
| `leaderboard-live` | per-config pass@1/pass@4, CIs, cost |
| `v1-delta` | v1 vs v1.1 on shared configs — same rollouts, re-graded |

`leaderboard`, `distribution`, `analysis`, `critiques` all 404.

**Per-rollout artifacts** — base `https://d3ujjcmjq6o8v6.cloudfront.net`,
prefix `<release>/trial-artifacts/<trial_name>/`:

| path | what |
| --- | --- |
| `agent/trajectory.json` | ATIF-format trajectory (pier's own format) |
| `agent/mini-swe-agent.txt` | raw agent log, full message history (~300 KB typical) |
| `artifacts/model.patch` | the diff the rollout submitted |
| `verifier/reward.json` | the grading result |
| `verifier/test-stdout.txt`, `verifier/ctrf.json`, `verifier/run.log` | suite output; `ctrf.json` names each failing test |
| `verifier/reports/*` | whatever each row's `verifier_files` lists |

No auth. Some gotchas:

- **~2.4% are missing.** Of 250 sampled rollouts with `has_model_patch: true`,
  6 return HTTP 403 and one is zero bytes. Confirmed on retry, so it is not
  throttling. Reported upstream as
  [deep-swe#59](https://github.com/datacurve-ai/deep-swe/issues/59). Handle
  missing bodies rather than scoring them 0.
- Trial names are truncated to 32 characters before the `__suffix`, so
  `valibot-recursive-schema-composition` appears as
  `valibot-recursive-schema-composi__FC64HrJ`. Don't reconstruct a trial name
  from a task name.
- `trials.json` is 51 MB. Cache it.
- 8 concurrent fetches was fine; I didn't push further.

## Replaying recorded actions

The trap that cost the most time: **a patch that fails to apply and a model that
did nothing produce identical scores.** Both leave the repo at base state, so the
verifier reports `0/N` fail-to-pass with `N/N` pass-to-pass and a reward of 0.
Nothing in the result distinguishes them.

Task images are built by checking the repo out at the base commit and *then*
running build steps, which can leave tracked files modified in-tree. So applying
a recorded patch onto the image as-is can fail on a file the build touched.
DeepSWE's grader resets exactly the files a patch touches to the base commit
first — mirror that, per file, before every apply attempt:

```bash
paths=$(sed -n 's|^diff --git a/.* b/||p' "$PATCH")
for p in $paths; do git checkout HEAD -- "$p" 2>/dev/null || rm -f "$p"; done
git apply --whitespace=nowarn --binary "$PATCH"
```

`HEAD` is the base commit inside the agent container, so `git checkout HEAD --`
is the right reset. Avoid `git apply --reject` as a fallback: it exits non-zero
while leaving a partial application on disk, which is how half a patch gets
committed and graded.

Log whether the apply succeeded, separately from the score. Otherwise a harness
bug looks exactly like a model failure — that is how a numba rollout in this
repo's first regrade run got recorded as a benchmark disagreement when the fault
was mine.

If you are splicing trajectories rather than patches, the same hazard applies one
level up: the environment state a trajectory assumes at step N has to match what
you reconstruct, or the spliced action lands somewhere it was never written for.

## pier gotchas

- Trial results are `jobs/<job>/<trial>/result.json` — **`result.json`, not
  `results.json`**, despite the docstring in `models/trial/paths.py` saying
  otherwise. A job-level `result.json` sits next to it; it has no `task_name`.
- `--agent oracle` replays the task's reference solution, `--agent nop` submits
  nothing. Neither needs an API key. Both have exact expected scores.
- A custom agent via `--agent-import-path` does **not** receive `task_dir` or
  `trial_paths` — those are passed only when the agent name is literally
  `oracle`. Take what you need through `--ak key=value`.
- Pass secrets as `--ae KEY='${VAR}'`. pier resolves `${VAR}` from its own
  environment, so the value never reaches a command line or a config file. It
  also redacts env keys matching KEY/SECRET/TOKEN when serialising configs, but
  the template is better.
- The agent is installed into a derived image layer at build time, so it gets
  normal network during install. At run time its egress goes through a squid
  proxy allowlisted to domains derived from the model id and from
  `OPENAI_BASE_URL` / `OPENAI_API_BASE` / `ANTHROPIC_BASE_URL` /
  `GEMINI_API_BASE` / `OPENROUTER_API_BASE`.
- `--n-tasks`/`--sample-seed` shuffles in `Path.iterdir()` order, which is
  filesystem order. The same seed can pick different tasks on different
  machines. Sort first.
- Task images come from `public.ecr.aws`, which rate-limits anonymous pulls per
  source IP. A wide runner matrix trips it, and it surfaces mid-trial as a
  `docker compose ... up` RuntimeError that reads like a task failure. Pull up
  front with backoff.

## Known upstream issues that will bite a provider run

These are open against pier and matter specifically when pointing the harness at
a third-party endpoint:

- [pier#7](https://github.com/datacurve-ai/pier/issues/7) — mini-swe-agent
  defaults `openai/*` models to litellm's Responses API, which breaks custom
  chat-completions endpoints. Most cheap providers are chat-completions only.
- [pier#37](https://github.com/datacurve-ai/pier/issues/37) — self-hosted
  endpoints on non-standard ports are blocked by the egress proxy.
- [pier#21](https://github.com/datacurve-ai/pier/issues/21) — mini-swe-agent
  crashing on the first model call with `No module named 'fastapi'` under
  current litellm.
- [pier#13](https://github.com/datacurve-ai/pier/issues/13) —
  `JobStats.n_agent_steps` is never accumulated, so step counts can come back
  empty.
- [deep-swe#81](https://github.com/datacurve-ai/deep-swe/issues/81) — the paper
  says a 9,000 s rollout timeout, the repo says 5,400 s. Set it explicitly.
- [deep-swe#48](https://github.com/datacurve-ai/deep-swe/issues/48) — task
  images are x86-64 only. An arm64 laptop cannot run them; use a runner.

## What it costs

Measured on `ubuntu-latest` (4 vCPU, 16 GB), oracle agent:

| | |
| --- | --- |
| per task, end to end | median 108 s (min 84, max 266) |
| container start, image already pulled | ~1 s |
| grading (both suites + score) | 25–178 s depending on the suite |
| one repeat of a single task | 55 s, of which 41 s is the verifier |
| all 113 tasks across 20 runners | ~11 minutes wall clock |
| task images | ~0.3–2.2 GB compressed each, 30 GB for the set |

Tasks declare 2 CPUs and 8 GB, so two fit on a runner at once —
`-f concurrency=2` roughly halves a repeat run. Prune Docker between tasks or
the disk fills.
