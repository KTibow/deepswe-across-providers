# deepswe-across-providers

Measure and debug an inference provider on [DeepSWE](https://deepswe.datacurve.ai/),
Datacurve's benchmark of 113 long-horizon software-engineering tasks, using
GitHub Actions runners as the containers.

DeepSWE publishes every rollout behind its leaderboard, including full
trajectories. That makes a sharper test possible than "run the benchmark and
compare the score": run a model on a provider with exactly the agent settings
of a published config, compare against that config on the same tasks, and when
something looks off, resume a published rollout part way through and see where
the provider's continuation goes wrong.

The harness itself was checked first — reference solutions pass 113/113, empty
submissions fail, and our grader agrees with Datacurve's. That work is in
[archive/2026-09-12-baseline](archive/2026-09-12-baseline).

## Running

Everything is one workflow, `run.yml`. Provider keys are repository secrets
named in `providers.toml`. `CROF_KEY` is set; `OPENROUTER_KEY` deliberately
isn't yet (see [Debugging a provider](#debugging-a-provider)).

```bash
# glm-5.3 on crof, on the 8 tasks the published glm-5.3 config always solved
gh workflow run run.yml -f agent=crof:glm-5.3 -f tasks=subset:glm-5.3-sentinel

# 12 tasks chosen to be as hard as the whole benchmark for glm-5.3, twice each
gh workflow run run.yml -f agent=crof:glm-5.3 -f tasks=subset:glm-5.3-representative -f attempts=2

# take the published glm-5.3 rollout on each sentinel task, replay its first
# half, and let crof finish it
gh workflow run run.yml -f agent=crof:glm-5.3 -f tasks=subset:glm-5.3-sentinel -f "prefix=pick=pass steps=50%"

# resume one rollout at several points (published, or one of ours)
gh workflow run run.yml -f agent=crof:glm-5.3 \
  -f "prefix=trials=ytt-jsonpath-query-api__DjxtPgs steps=25%,50%,75%"

# replay published rollouts end to end with no model: should grade as published
gh workflow run run.yml -f agent=replay -f "prefix=trials=ytt-jsonpath-query-api__DjxtPgs steps=all"

# controls: reference solutions must pass, empty submissions must fail
gh workflow run run.yml -f agent=oracle -f tasks=sample:12:1
gh workflow run run.yml -f agent=nop -f tasks=sample:12:1
```

`tasks` takes `subset:NAME`, `sample:N:SEED`, `all`, or task ids and globs.
`prefix` takes `trials=A,B` (published rollout names) or `pick=pass|fail|any`
(one rollout of the reference config per selected task), plus
`steps=N|N%|all` and optionally `observations=recorded|live`.

Runs use a 5400 s agent timeout by default — what DeepSWE's published runs
used, not the 10800 s in `task.toml` — so timeouts stay comparable.

## What a run tells you

The job summary (also `report/summary.md`, with everything in `results.json`):

1. **Verdict.** Rollouts solved, next to how many the reference config would
   be expected to solve on the same tasks, with a z-score. Each task's
   expected rate blends the reference config's four published rollouts with
   every config's rate on that task, so a task glm-5.3 solved 4/4 expects
   about 0.9, not 1.0.
2. **Outcomes by owner.** Every rollout that didn't pass is put on one of: the
   model (submitted and failed the tests), the provider (a call still failing
   after mini-swe-agent's 10 retries), format errors, agent timeout, or our
   own infrastructure or grading. The last two are left out of the score and
   should be rerun.
3. **Inference health.** From mini-swe-agent's own trajectory for every live
   call: retries by error type, finish reasons, format errors, empty replies,
   C1 control characters (UTF-8 mis-decoded as Latin-1), wait per call, output
   tokens, cache hit share — and median steps, tokens, minutes and cost per
   rollout against the reference config's published rollouts on the same
   tasks.
4. **Per task**, one mark per rollout.
5. For resumed runs, **outcome by resume point**, and for replays, whether
   the replayed commands printed what the recording says they printed.

Each shard's artifact keeps the verifier output and the agent's full
trajectories, so any of our rollouts can be probed or resumed later.

## Debugging a provider

Work from cheapest to most expensive.

**Probe a single step, no container.** Rebuild the exact history a published
(or our own) rollout sent at step K, send it to the provider, and compare:

```bash
python3 -m dswe.probe crof:glm-5.3 abs-module-cache-flags__KeBKjxc --steps 8,40,90 --key-file /tmp/crof-key.txt
# step  40: 12.1s  prompt 63521 (recorded 63450 x1.001)  out 258 ...  format=ok
```

The prompt-token ratio is the quickest fidelity check there is: the same
history through the same tokenizer should cost the same, so a gap means the
provider's chat template renders the history differently — dropped reasoning,
reformatted tool results. Latency and output tokens tell you how long a real
rollout will take.

**Resume a rollout on the provider under test.** If the provider scores low on
a task, resume the reference config's passing rollout at a few points. If
continuations from early on fail but late ones pass, the provider goes wrong
somewhere in between; narrow it with more points. With
`observations=recorded` the model is sent exactly the recorded history, so the
only difference from the recording is who answers.

**Ask the reference provider last, and one call at a time.** Some questions
only the original provider can answer — would it have looped here too? — and
that costs money where crof doesn't. So find the step first: the report flags
rollouts that keep repeating recent commands and says where the streak
starts. Probe that single step on `openrouter:<model>` (pinned to the
original provider) before considering a resumed session there. A full resume
(`prefix=trials=run:<id>/<unit> steps=K` with an OpenRouter agent) replays
every later call and is the expensive last resort. `OPENROUTER_KEY` gets added
once crof results show a specific case that needs it.

## Adding a provider or model

A provider is a base URL, the secret holding its key, and litellm's prefix for
it. A model block copies the settings of DeepSWE's published config for that
model, read off the header of any of its published `agent/mini-swe-agent.txt`
(`Building agent config from specs: [...]`), and maps the model to each
provider's id. See `providers.toml`.

Then pick subsets for it:

```bash
python3 -m dswe.subset kimi-k3 sentinel --n 8
python3 -m dswe.subset kimi-k3 representative --n 12
```

## Layout

```
.github/workflows/run.yml   plan -> sharded pier runs -> report
providers.toml              providers, and each model's published agent settings
subsets/                    committed task subsets, with the stats they were picked by
dswe/plan.py                task selection, resume points, sharding, pier flags per unit
dswe/agent.py               pier's mini-swe-agent, able to start from a recorded trajectory
dswe/replay_model.py        runs in the container: replays the prefix, then calls the model
dswe/atif.py                ATIF trajectory -> the message history mini-swe-agent sent
dswe/collect.py             one shard's pier output -> records with inference metrics
dswe/report.py              records -> verdict, owners, inference health, per task
dswe/probe.py               one step against a provider, no container
dswe/subset.py              sentinel and representative subsets from published data
dswe/published.py           DeepSWE's published tables and trajectories, cached
scripts/run_shard.sh        one pier job per unit, disk reclaimed between
tests/                      local checks: replay model through the real CLI, collect + report
```

Pinned: the benchmark by commit (`DEEPSWE_REF` in `run.yml`), pier by version,
mini-swe-agent by the version each published config used. [NOTES.md](NOTES.md)
has the practical details that cost time to find out.

## Credits

Benchmark and harness are Datacurve's: [deep-swe](https://github.com/datacurve-ai/deep-swe),
[pier](https://github.com/datacurve-ai/pier), [paper](https://arxiv.org/abs/2607.07946).
Agent: [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent).
