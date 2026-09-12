# deepswe-across-providers

Running [DeepSWE](https://deepswe.datacurve.ai/) — Datacurve's benchmark of 113
original, long-horizon software-engineering tasks — in GitHub Actions runners,
so that a model's score can be compared against Datacurve's published numbers.

The point is attribution. If you run some model through a cheap inference
provider and it scores 30% where the leaderboard says 55%, that gap has at
least four possible owners: the provider, the model, the harness, or your
runner. This repo nails down the last two first, then measures the model.

## Establish the baseline before trusting a number

Three checks, none of which call a model, each with a known-correct answer:

| check | workflow | what it proves | correct result |
| --- | --- | --- | --- |
| reference replay | `replay.yml` (`agent=oracle`) | the task images, verifiers and graders work on a GitHub runner | reward **1** on every task |
| empty control | `replay.yml` (`agent=nop`) | tasks aren't passing for free | reward **0** on every task |
| rollout regrade | `regrade.yml` | our verdicts match Datacurve's on *their* recorded submissions | same verdict as published |
| leaderboard recompute | `scripts/analyze_published.py` | the published leaderboard follows from the published rollout table | exact match |

`oracle` runs each task's held-out `solution/solve.sh`; `nop` does nothing. Both
are graded identically — a `[[verifier.collect]]` hook extracts the agent's
commits as `model.patch` and a pristine verifier container applies it alongside
the held-out tests. DeepSWE's fail-to-pass and pass-to-pass whitelists were
*"materialized from the oracle-vs-nop differential"* (`tests/grader.py`), so
these two expectations are exact, not estimates.

The regrade goes further. DeepSWE publishes all 31,617 rollouts behind its
leaderboard and, for 28,815 of them, the `model.patch` the agent actually
submitted. Replaying one of those patches here and grading it is the same
inputs and the same grader on a different machine, so any disagreement is a
reproduction difference and nothing else.

## Then measure the model

```bash
# set the key once
gh secret set PROVIDER_API_KEY

# 10 tasks, one rollout each, scored against published results on those tasks
gh workflow run bench.yml -f model="openrouter/qwen/qwen3-coder" -f n_tasks=10

# a cheap provider behind an OpenAI-compatible endpoint
gh workflow run bench.yml \
  -f model="openai/some-model" \
  -f api_base="https://api.example.com/v1" \
  -f n_tasks=10 -f attempts=3

# the same tasks a published config saw, to compare like for like
gh workflow run bench.yml -f model=... -f n_tasks=20 -f seed=0 \
  -f reference_configs="mini_swe_agent_claude_opus_5_high"
```

Smoke it with one task before spending anything:

```bash
gh workflow run bench.yml -f model="openrouter/..." -f tasks="igel-persist-feature-schema"
```

`mini-swe-agent`'s own knobs go through `agent_kwargs`, space separated —
`reasoning_effort=high`, `cost_limit=5`, `model_class=...`. pier installs the
agent into a derived image layer at build time, so the agent's own egress stays
limited to the provider domain implied by the model id or `api_base`.

`bench.yml` drives `mini-swe-agent` — the same scaffold every leaderboard entry
used — and reports two rates: **strict** (an errored rollout is a failure, which
is what you want when the provider is what's under test) and **DeepSWE policy**
(infrastructure and provider errors dropped, which is how published numbers are
computed). The key is passed as a `${PROVIDER_API_KEY}` template, so it never
reaches a command line, a config file, or an artifact.

## Baseline runs

```bash
gh workflow run replay.yml -f n_tasks=12 -f shards=12      # reference replay
gh workflow run replay.yml -f agent=nop -f n_tasks=12      # empty control
gh workflow run regrade.yml -f n_trials=10 -f balance=true # rollout regrade
gh workflow run replay.yml -f n_tasks=0 -f shards=20       # all 113 tasks
```

Results land in the job summary and a `report` artifact (`summary.md`,
`results.json`). Findings from the runs done so far are in
[`FINDINGS.md`](FINDINGS.md).

## What will bite you

- **Small subsets are noisy.** Per-task pass rates across published rollouts run
  from 2.5% (`obsidian-linter-auto-table-of-contents`) to 92%
  (`true-myth-iterable-collection-combinators`). A 10-task subset can swing a
  model's apparent score by tens of points, so keep `seed` fixed and compare
  against the published reference *on the same tasks* — which is what
  `aggregate_bench.py` does.
- **Grading version matters more than it looks.** DeepSWE v1 scored by exit
  code, v1.1 by test node id. Re-grading the *same* rollouts moved individual
  configs by up to 6 points and individual tasks by up to 67, while the pooled
  rate barely moved. The arXiv paper's numbers are v1; the live leaderboard is
  v1.1.
- **The error policy is a choice.** DeepSWE excludes provider errors, timeouts
  at the infrastructure level, and grading errors, and does not resample them.
  If you are testing a provider, that policy hides exactly what you are looking
  for. Read the strict rate.
- **Subset sampling isn't portable.** pier's `--n-tasks/--sample-seed` shuffles
  tasks in `Path.iterdir()` order, which is filesystem order, so the same seed
  can select different tasks on different machines. `scripts/plan.py` sorts
  first, so `(seed, n_tasks)` names one fixed subset anywhere.

## Layout

```
.github/workflows/replay.yml    oracle/nop: plan -> sharded replay -> report
.github/workflows/regrade.yml   published rollouts: select -> regrade -> compare
.github/workflows/bench.yml     a real model through mini-swe-agent -> score
scripts/plan.py                 deterministic subset selection + sharding
scripts/select_trials.py        pick published rollouts that have a patch
scripts/patch_agent.py          pier agent that replays a recorded submission
scripts/run_shard.sh            one pier job per task, disk reclaimed between
scripts/run_trials_shard.sh     same, fetching each rollout's patch first
scripts/run_bench_shard.sh      same, driving mini-swe-agent against a provider
scripts/collect.py              trial results + verifier logs -> compact JSON
scripts/aggregate.py            oracle/nop replay -> summary.md
scripts/aggregate_trials.py     ours vs published, with a confusion matrix
scripts/aggregate_bench.py      model score vs published, same tasks, with CIs
scripts/analyze_published.py    recompute the leaderboard from published data
```

Everything is pinned: the benchmark by commit SHA (`deepswe_ref`), the harness
by version (`pier_version`), the runner by image (`ubuntu-latest`). Tasks
declare 2 CPUs / 8 GB / 20 GB disk, so a 4-vCPU runner runs one at a time and
Docker is pruned between tasks — each task pulls its own multi-GB image from
`public.ecr.aws`.

## Credits

Benchmark and harness are Datacurve's: [deep-swe](https://github.com/datacurve-ai/deep-swe),
[pier](https://github.com/datacurve-ai/pier), [paper](https://arxiv.org/abs/2607.07946).
