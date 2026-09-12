# deepswe-across-providers

Reproducing [DeepSWE](https://deepswe.datacurve.ai/) — Datacurve's benchmark of
113 original, long-horizon software-engineering tasks — inside GitHub Actions
runners, with no model in the loop.

## What is being reproduced

DeepSWE ships tasks in the [Harbor](https://www.harborframework.com/docs/tasks)
format and is run with [Pier](https://github.com/datacurve-ai/pier). Pier has
two agents that need no LLM at all:

| agent | what it does | correct score |
| --- | --- | --- |
| `oracle` | uploads the task's held-out `solution/`, runs `solve.sh`, commits the result | reward **1** on every task |
| `nop` | does nothing at all | reward **0** on every task |

Grading is identical either way: a `[[verifier.collect]]` hook extracts the
agent's commits as `model.patch`, and a pristine verifier container applies that
patch plus the held-out `test.patch` and checks each task's fail-to-pass and
pass-to-pass node-id whitelists.

Those whitelists are, in DeepSWE's own words, *"materialized from the
oracle-vs-nop differential"* (`tests/grader.py`). So the expected score is not
an estimate — replaying the reference actions **must** produce 1.0, and
replaying nothing **must** produce 0.0. Any deviation is a difference between
the environment Datacurve built the benchmark in and the environment you are
running it in, which is exactly what this repo measures.

This is deliberately *not* a leaderboard run. A leaderboard entry
(`--agent mini-swe-agent --model ...`) measures a model; the replay measures the
benchmark.

## Running it

Actions → **DeepSWE replay** → *Run workflow*, or:

```bash
# five deterministically sampled tasks, one runner each
gh workflow run replay.yml -f n_tasks=5 -f shards=5

# named tasks
gh workflow run replay.yml -f tasks="tomlkit-toml-table-converters termenv-preserve-ansi-resets"

# the whole benchmark across 20 runners
gh workflow run replay.yml -f n_tasks=0 -f shards=20

# flakiness: same task five times
gh workflow run replay.yml -f tasks="katex-multicolumn-array-spans" -f attempts=5

# the negative control
gh workflow run replay.yml -f agent=nop -f n_tasks=5
```

Each run produces a job summary and a `report` artifact containing
`summary.md` (per-task table + a *Differences* section with failing node ids and
verifier stdout) and `results.json`.

### Layout

```
.github/workflows/replay.yml   plan -> sharded replay -> report
scripts/plan.py                deterministic subset selection + sharding
scripts/run_shard.sh           one pier job per task, disk reclaimed between
scripts/collect.py             trial results + verifier logs -> compact JSON
scripts/aggregate.py           shards -> summary.md / results.json
```

Everything is pinned: the benchmark by commit SHA (`deepswe_ref`), the harness
by version (`pier_version`), the runner by image (`ubuntu-latest`).

## Notes on reproducibility

**Subset selection.** Pier's `--n-tasks/--sample-seed` shuffles tasks in
`Path.iterdir()` order (`pier/models/job/config.py`), which is filesystem
order, not sorted order. The same seed can therefore select different tasks on
different machines. `scripts/plan.py` sorts before shuffling so a `(seed,
n_tasks)` pair names one fixed subset anywhere.

**Runner sizing.** DeepSWE tasks declare 2 CPUs / 8 GB RAM / 20 GB disk. A
GitHub-hosted `ubuntu-latest` runner is 4 vCPU / 16 GB, so tasks run one at a
time per shard, and Docker is pruned between tasks because each task pulls its
own multi-GB image from `public.ecr.aws`.

## Findings

See [`FINDINGS.md`](FINDINGS.md).
