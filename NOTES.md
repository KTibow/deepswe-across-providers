# Working notes

Things that cost time to find out, for running providers against DeepSWE and
for reusing what DeepSWE published.

## Where the published data lives

None of this is linked from the repo or the paper; it comes from the SSR payload
on a trial detail page (`https://deepswe.datacurve.ai/data/v1.1/trials/<trial>`,
search the inline `$_TSR.router` script for `artifact_patterns`). `dswe/published.py`
fetches and caches all of it.

**Aggregate tables** — `https://deepswe.datacurve.ai/artifacts/<release>/<name>.json`:

| name | what |
| --- | --- |
| `trials` | every rollout: model, config, reward, f2p/p2p counts, steps, tokens, cost, duration, error category (51 MB for v1.1, 31,617 rows) |
| `tasks` | the 113 tasks: id, language, repository, base commit |
| `leaderboard-live` | per-config pass@1/pass@4, CIs, medians of cost, steps, tokens, duration |
| `v1-delta` | v1 vs v1.1 on shared configs — same rollouts, re-graded |

**Per-rollout artifacts** — `https://d3ujjcmjq6o8v6.cloudfront.net/<release>/trial-artifacts/<trial_name>/`:

| path | what |
| --- | --- |
| `agent/trajectory.json` | ATIF trajectory: every step's text, reasoning, tool calls, tool output and token usage |
| `agent/mini-swe-agent.txt` | the agent's console log; its header shows the exact config the rollout ran with |
| `artifacts/model.patch` | the diff the rollout submitted |
| `verifier/reward.json`, `ctrf.json`, `test-stdout.txt` | the grade; `ctrf.json` names each failing test |

mini-swe-agent's native `mini-swe-agent.trajectory.json`, `result.json`,
`config.json` and `trial.log` are not published (403).

- **~2.4% of artifacts are missing** (403 or zero bytes), confirmed on retry
  ([deep-swe#59](https://github.com/datacurve-ai/deep-swe/issues/59)).
- **Trial names are truncated** to 32 characters before the `__suffix`. Don't
  build one from a task name.
- `trials` rows have `n_reasoning_tokens` as null for glm-5.3 and some others.

## What the published configs actually ran

Read it off the header of any published `agent/mini-swe-agent.txt`:

```
This is mini-swe-agent version 2.4.2.
Building agent config from specs: ['mini.yaml', 'agent.cost_limit=0',
'model.model_class=litellm', 'model.set_cache_control=default_end',
'model.model_kwargs.extra_body={"reasoning_effort": "max", "thinking":
{"type": "enabled", "clear_thinking": false}}', ...]
```

`providers.toml` copies these per model. Things that differ from what you'd assume:

- **The agent timeout was 5400 s**, not the 10800 s in every `task.toml`
  (`AgentTimeoutError: Agent execution timed out after 5400.0 seconds` in the
  published exceptions; [deep-swe#81](https://github.com/datacurve-ai/deep-swe/issues/81)).
  `run.yml` defaults to `--agent-timeout-multiplier 0.5` to match.
- **mini-swe-agent versions differ per config** (2.4.1 to 2.4.6). pier installs
  whatever `--ak version=` says, so pin it per model.
- **Settings vary by provider, not just model.** glm-5.3-flash went through Z.AI's
  Anthropic-compatible endpoint with `thinking: adaptive` and
  `output_config.effort`; that shape doesn't carry over to an OpenAI-style
  provider, which is why it has no profile yet.
- **Dated checkpoints.** DeepSWE's `deepseek-v4-pro` config ran 2026-08-12 20:32
  to 08-13 02:24 UTC on DeepSeek's API, hours after the V4 Pro GA ("0813") was
  listed; `deepseek-v4-flash` ran 08-05 to 08-06, after the 0731 re-post-train.
  OpenRouter's undated `deepseek/deepseek-v4-pro` and `-flash` ids are the April
  releases, so the dated ids are the matching ones.
- The prompt's `<system_information>` line is the host's `uname`, so a rerun's
  task prompt differs from the recording in that one line.

## mini-swe-agent behaviour that shows up in results

- **Retries.** A failed model call is retried up to 10 times, waiting 4–60 s
  (tenacity, exponential). Each retry logs
  `Retrying <unknown> in 4 seconds as it raised ServiceUnavailableError: ...`
  to `mini-swe-agent.txt`; nothing about it reaches the trajectory. A provider
  can be flaky enough to slow every rollout down while never failing one.
- **When retries run out** the exception class becomes the exit status
  (`ServiceUnavailableError`, `RateLimitError`, `APIConnectionError`, ...).
  Authentication, not-found, context-window and unsupported-params errors abort
  immediately without retrying.
- **Format errors.** A reply without a usable bash tool call isn't kept in the
  history; the agent sends a correction prompt instead, and three in a row end
  the run as `RepeatedFormatError`. A reply cut off at `finish_reason=length`
  gets a different correction prompt.
- **Timestamps.** Each assistant message and tool result carries
  `extra.timestamp`, which is how per-call latency is measured. Format-error
  messages don't.
- The litellm response (usage, finish reason) is saved on every assistant
  message under `extra.response`.

## Spotting loops

Loops often bump a number each turn (`-o /tmp/logs4.md`, then `logs5.md`)
while repeating the same reasoning word for word, so exact matching misses
them. The collector counts a step as a repeat when its commands match one of
the previous five steps' with numbers ignored, and either the commands match
exactly or the text does (numbers ignored). The second condition keeps paging
through a file (`sed -n 1,80p`, `80,160p`) from counting.

Calibration on 30 published glm-5.3 rollouts at Z.AI: 0.8% of steps are
repeats; 26 rollouts have none; one has a 5-step streak, and that one was
legitimately polling a background eslint run (`sleep 29; cat /tmp/lint.log;
pgrep -f eslint`) and passed. So one 5-step streak isn't a loop by itself.
Compare the rate with the reference, and read the steps where a streak starts.

## crof.ai

- OpenAI-compatible chat completions and Responses API at `https://crof.ai/v1`;
  `GET /v1/models` lists ids, quantization and prices.
- **Preserved reasoning works.** `reasoning_content` sent back in assistant
  history is used by the model, and for the same history crof's glm-5.3 counts
  the recorded Z.AI prompt tokens plus a constant 71 (19228 vs 19157, 63521 vs
  63450, 107143 vs 107072). A provider that dropped reasoning from the history
  would come in thousands of tokens short.
- **Prefix caching works** across calls with a shared prefix.
- **Speed**, glm-5.3 at `reasoning_effort=max`, 2026-09-12. A single probe: a
  4.5k-token reply took 152 s, ~7.7 s to first token, ~31 tokens/s; short
  steps took 12–18 s at 63k–107k context. A full rollout of
  `ytt-jsonpath-query-api` (run 34715508369) waited on crof for 29.4 of its 31
  agent minutes, twice the published Z.AI median of 16: typical wait 8.7 s,
  90th percentile 29 s, longest 219 s. The slowest calls were the long
  reasoning replies (12.3k tokens in 219 s, 7.0k in 166 s), so about 40–55
  tokens/s once decoding. Speed matters for scores here because the agent
  timeout is 90 minutes.
- **It slows down as context grows.** Across that rollout's fifths, typical
  prompt 23k → 105k tokens, typical wait 4.8 → 13.6 s, and output tokens per
  second of waiting 52 → 32. Almost all input was cached by then (99%), so
  this is serving long contexts, not re-reading them. Output length explains
  nearly all of the wait (about 4 s plus 18 s per 1,000 output tokens fits 97%
  of it); how much input wasn't cached made no measurable difference.
- **Reasoning tokens depend on which response path crof uses.** Checked
  2026-09-12 with glm-5.3: a request without tools comes back with a `gen-…`
  id and reports `usage.reasoning_tokens` at the top level (not in the
  standard `completion_tokens_details`); a request with a tool comes back as
  `chatcmpl-…` and reports no reasoning count at all. Streamed with a tool it
  reported one, but it was impossible: 4,911 reasoning of 4,563 output tokens.
  mini-swe-agent always sends its bash tool, so in runs crof never reports
  reasoning tokens. The report estimates them from the reasoning's share of
  each reply's characters; on 12 published glm-5.3 trajectories, which report
  both, that estimate was 61.9% against an exact 60.5%.
- **Behaviour per call matched Z.AI** on that rollout: no retries, no format
  errors, 26.1 vs 23.3 garbled characters per 100 calls, typical output 262
  vs 282 tokens, 98.2% vs 97.8% of prompt tokens cached, no repeated steps. It
  did take 92 steps where Z.AI's four rollouts took 60–89, and passed.
- A glm-5.3 reply came back with UTF-8 mis-decoded as Latin-1 (an em dash as
  three characters starting with `â`). That is glm-5.3 at Z.AI too: in 30
  published glm-5.3 trajectories, 29 have it in the model's own output (388
  sequences, next to 23k correctly encoded characters), and tool output has
  none. One rollout greps its files for the bad bytes after writing them. So
  compare the rate per call against the reference, not the raw count; the
  report does.

## Replaying recorded actions

**A patch that fails to apply and a model that did nothing produce identical
scores.** Both leave the repo at base state. Log whether an apply succeeded
separately from the score.

Task images are built by checking the repo out at the base commit and *then*
running build steps, which can leave tracked files modified. DeepSWE's grader
resets exactly the files a patch touches before applying it; anything that
applies a recorded patch has to do the same (see
`archive/2026-09-12-baseline/scripts/patch_agent.py`).

Replaying a whole trajectory (`dswe/replay_model.py`) avoids that class of
problem: the recorded commands run in order, so the environment gets there
the way the recorded agent did. What can still differ is anything the commands
print that depends on time, paths or scheduling. With
`observations=recorded` the model is sent the recorded output regardless;
with `live`, what our run printed.

What "different" looks like in practice: replaying a published glm-5.3
rollout of `ytt-jsonpath-query-api` on a GitHub runner, 20 of 60 steps printed
something other than the recording, and it still graded 103/103 exactly as
published. The differences were `ls -la` sizes and dates (Modal's filesystem
vs the runner's), `find` listing files in another order, and Go test timings
(`0.003s` vs `0.002s`). The replay labels each difference as `numbers`,
`order` or `content`; only `content` is worth reading.

ATIF keeps enough to rebuild the exact message history: agent steps carry
text, reasoning and tool calls, and each observation holds one result per tool
call followed by any format-error prompt sent before the next step.

## pier gotchas

- Trial results are `jobs/<job>/<trial>/result.json` — **`result.json`, not
  `results.json`**. A job-level `result.json` sits next to it with no `task_name`.
- `--agent oracle` replays the reference solution, `--agent nop` submits
  nothing. Neither needs an API key.
- An import-path agent (`--agent-import-path`) gets `--ak` kwargs, `--ae` env as
  `extra_env`, and `model_name`, but **not** `task_dir` or `trial_paths` (only
  an agent literally named `oracle` does). `--ak` values are parsed as JSON.
- An installed agent's `install_spec()` becomes a derived image layer, built
  with normal network and cached by a fingerprint of its steps. At run time the
  agent's egress goes through a squid proxy allowlisted to domains from the
  model id and `OPENAI_BASE_URL` / `OPENAI_API_BASE` / `ANTHROPIC_BASE_URL` /
  `GEMINI_API_BASE` / `OPENROUTER_API_BASE`.
- For `openai/*` models pier's mini-swe-agent defaults to litellm's Responses
  API ([pier#7](https://github.com/datacurve-ai/pier/issues/7)); pass
  `model_class=litellm` for chat completions, as the published configs did.
- Pass secrets as `--ae KEY='${VAR}'`; pier resolves the template from its own
  environment.
- `--n-tasks`/`--sample-seed` shuffles in filesystem order, so the same seed can
  pick different tasks on different machines. `dswe/plan.py` sorts first.
- `public.ecr.aws` rate-limits anonymous pulls per IP; a wide matrix trips it
  mid-trial as a `docker compose up` error. `scripts/pull_image.sh` pulls up
  front with backoff and shards stagger their start.

## Other open upstream issues

- [pier#37](https://github.com/datacurve-ai/pier/issues/37) — self-hosted
  endpoints on non-standard ports are blocked by the egress proxy.
- [pier#21](https://github.com/datacurve-ai/pier/issues/21) — mini-swe-agent
  crashing with `No module named 'fastapi'` under some litellm versions.
- [pier#13](https://github.com/datacurve-ai/pier/issues/13) —
  `n_agent_steps` is never accumulated; the collector counts steps from the
  trajectory instead.
- [deep-swe#48](https://github.com/datacurve-ai/deep-swe/issues/48) — task
  images are x86-64 only; an arm64 laptop can't run them.

## What it costs

Measured on `ubuntu-latest` (4 vCPU, 16 GB) with the oracle agent:

| | |
| --- | --- |
| per task, end to end | median 108 s (min 84, max 266) |
| grading (both suites + score) | 25–178 s depending on the suite |
| all 113 tasks across 20 runners | ~11 minutes wall clock |
| task images | ~0.3–2.2 GB compressed each, 30 GB for the set |

A model run adds the agent's time on top: the published glm-5.3 config's
median rollout was 114 steps and 31 minutes against Z.AI. Tasks declare 2 CPUs
and 8 GB, so two trials fit on a runner (`concurrency=2`).
