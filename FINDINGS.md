# Findings

What the baseline runs and the published data say, as of 2026-09-12.
Benchmark pinned at `datacurve-ai/deep-swe@0b9fabb`, harness `datacurve-pier==0.3.1`,
runner `ubuntu-latest`, data release `v1.1`.

## 1. The benchmark runs correctly on a GitHub runner

Replaying each task's reference solution (`--agent oracle`) and grading it with
the held-out verifier:

| run | tasks | expected | result |
| --- | ---: | --- | --- |
| [smoke, one task per language](https://github.com/KTibow/deepswe-across-providers/actions/runs/34709187966) | 5 | reward 1 each | **5/5** |
| [seed 0 sample](https://github.com/KTibow/deepswe-across-providers/actions/runs/34709631628) | 12 | reward 1 each | **12/12** |
| [empty control (`nop`)](https://github.com/KTibow/deepswe-across-providers/actions/runs/34710135561) | 12 | reward 0 each | **12/12** |
| [full sweep](https://github.com/KTibow/deepswe-across-providers/actions/runs/34711567315) | **113** | reward 1 each | **113/113** |

The two runs bracket the benchmark exactly. On the same 12 tasks, `oracle`
passes every fail-to-pass and pass-to-pass node (`44/44` f2p and `2738/2738` p2p
on `adaptix-name-mapping-aliases`, and so on across all five languages), while
`nop` scores `0/44` f2p and `2738/2738` p2p — every fail-to-pass test fails at
base, every pass-to-pass test passes at base, on every task. That is the
differential the benchmark's whitelists were built from, reproduced on a
GitHub runner.

Cost of the environment itself: **median 108 s per task** end to end (image pull,
apply, full test suite, grade), min 84 s, max 266 s. With the image already
pulled, container start is ~1 s and grading is 25–178 s depending on the suite.
That is the floor a model run adds its own thinking time to.

### Every reference solution grades correctly in v1.1

The full 113-task sweep passes — 34/34 Go, 34/34 Python, 35/35 TypeScript, 5/5
JavaScript, 5/5 Rust, every f2p and p2p node.

That settles an open report against v1.0. [deep-swe#30](https://github.com/datacurve-ai/deep-swe/issues/30)
found **105/113** under the old exit-code grading: 6 tasks broken by unpinned
dependency drift (`narwhals-rolling-window-suite` on polars 1.40, `skrub-duration-encoding`,
`dateutil-rfc5545-timezone-interop`, `langchain-request-coalescing`,
`fd-deterministic-multi-key-sorting`, `igel-persist-feature-schema`) and 2 where
the reference solution genuinely failed (`mnamer-daemon-watch-lifecycle`,
`helm-unified-manifest-stream`). All 8 pass here under v1.1:

| task | f2p | p2p |
| --- | ---: | ---: |
| `narwhals-rolling-window-suite` | 103/103 | 10093/10093 |
| `skrub-duration-encoding` | 130/130 | 2784/2784 |
| `dateutil-rfc5545-timezone-interop` | 67/67 | 2035/2035 |
| `langchain-request-coalescing` | 50/50 | 232/232 |
| `fd-deterministic-multi-key-sorting` | 43/43 | 109/109 |
| `igel-persist-feature-schema` | 24/24 | 2/2 |
| `mnamer-daemon-watch-lifecycle` | 51/51 | 319/319 |
| `helm-unified-manifest-stream` | 5/5 | 2/2 |

Node-id whitelists plus rebuilt `-v1.1` images fixed the class where one
unrelated broken baseline test zeroed a correct solution
([deep-swe#17](https://github.com/datacurve-ai/deep-swe/issues/17)).

## 2. Our verdicts match Datacurve's on their own rollouts

Taking rollouts DeepSWE published, fetching the `model.patch` each one actually
submitted, replaying it here and grading it
([run](https://github.com/KTibow/deepswe-across-providers/actions/runs/34709626013)):

| published \ ours | pass | fail |
| --- | ---: | ---: |
| **pass** | 4 | 0 |
| **fail** | 0 | 5 |

**9/9 agreement**, and not just on the verdict — the node counts line up exactly
on the partial failures too (`anko-default-function-arguments` 1/2 f2p here,
`[1, 2]` published; `testem-per-launcher-reports` 64/65 here, `[64, 65]`
published; `clack-async-autocomplete-options` 72/82 both). The grader is
deterministic across machines.

The tenth rollout produced no verdict, for an infrastructure reason worth
knowing about — see below.

## 3. The published leaderboard follows from the published data

`scripts/analyze_published.py` recomputes every leaderboard row from the 31,617
published rollouts: **all 70 configs reproduce exactly**, pass@1 and pass@4.
No task is unsolvable (every one of the 113 was solved by someone) and none is
free (none was solved by everyone).

## 4. Things that will cost you a run if you don't know them

### One f2p test grades some submissions nondeterministically

`langchain-request-coalescing`, replayed 10 times from an identical patch on an
identical runner image:

| input replayed | reward 1 | reward 0 |
| --- | ---: | ---: |
| a published rollout's `model.patch` (claude-sonnet-5) | **2** | **8** |
| the task's reference solution | **10** | 0 |

The reference is stable because it never races. It dedups the input list
synchronously before dispatching:

```python
keys = [_make_coalesce_key(inp) for inp in inputs]
unique_inputs = [inputs[i] for i in unique_indices]   # ["hello", "world"]
unique_results = self.bound.batch(unique_inputs, ...)  # exactly 2 calls
```

`call_count == 2` is then true by construction. The rollout's implementation
instead delegates to `Runnable.batch`, which runs `self.invoke` per item
concurrently, so the two `"hello"` items coalesce only if the second reaches the
in-flight registry before the first completes — a race, and the submission's own
comment says so:

```python
# `Runnable.batch` runs `self.invoke` (our coalescing-aware override)
# per item, concurrently, while preserving positional order.
return Runnable.batch(self, inputs, config, ...)
```

The prompt's requirement is unconditional — *"Batch methods coalesce per-item
and preserve positional order"* — so the rollout does not reliably meet the
spec, and grading it 0 is correct. The defect in the test cuts the other way:
without the barrier, a non-compliant implementation passes about 1 run in 5.
Adding the barrier its siblings use would make the check deterministic and
stricter, not more lenient.

The test that moves is benchmark-authored (it comes from the task's own
`tests/test.patch`, not upstream langchain) and it is on the fail-to-pass
whitelist:

```python
def test_batch_per_item_coalescing() -> None:
    inner = _Blocking()
    coalesced = inner.with_coalesce()
    def do_batch() -> None:
        results[0] = coalesced.batch(["hello", "hello", "world"])
    t = threading.Thread(target=do_batch)
    t.start()
    inner.release()          # races the worker thread
    t.join(timeout=10)
    assert inner.call_count == 2
```

`call_count == 2` holds only if both `"hello"` items are in flight at once.
Nine of the 14 threaded tests in the same file sleep before releasing the
blocked runnable; five do not, and three of those five assert a coalescing
invariant (`test_batch_per_item_coalescing`,
`test_batch_as_completed_coalesced_yield_together`,
`test_stream_late_joiner_gets_all_chunks`). So an implementation that submits
batch items one at a time can still coalesce when scheduling cooperates, and
the test scores it differently run to run.

How much of the task's published data this touches: of 257 scored rollouts, 73
are exactly one f2p test short. Sampling 40 of their published `ctrf.json`
files:

| failing test | rollouts |
| --- | ---: |
| `test_batch_per_item_coalescing` | 28/40 |
| `test_callbacks_fire_for_joined_callers` | 9/40 |
| `test_stats_after_operations` | 2/40 |
| `test_coalesce_clear_cancels_sync_waiters` | 1/40 |

Related open report: [deep-swe#45](https://github.com/datacurve-ai/deep-swe/issues/45)
(`prometheus-transactional-reload-status` carries 21 `TestQueryLog` entries in
its pass-to-pass whitelist — an upstream-known flaky test unrelated to the
task). Those entries are still in the v1.1 config; the task passed 82/82 in our
sweep, so it did not flake that run.

**For provider comparison:** repeat a borderline task before reading anything
into it. A single rollout's verdict on a task like this carries real variance
that has nothing to do with which provider served the tokens.

### Registry rate limits, not task failures

The one rollout that produced no verdict failed with:

```
main Error toomanyrequests: Rate exceeded
Error response from daemon: toomanyrequests: Rate exceeded
```

`public.ecr.aws` rate-limits anonymous pulls per source IP, and a 17-runner
matrix all pulling multi-GB task images at once trips it. It surfaces as a
`docker compose ... up` RuntimeError mid-trial, which looks like a task failure
in any summary that doesn't read the log. `scripts/pull_image.sh` now pulls each
image up front with exponential backoff and shards stagger their start.

**If you are benchmarking a provider, this class of error is the one that will
fool you.** Read the strict rate, and read the error categories.

### Grading version moves scores more than you'd guess

DeepSWE v1 scored by exit code; v1.1 scores by test node id. Datacurve re-graded
the *same* rollouts under both:

- pooled pass rate: 0.5346 → 0.5340 (essentially unchanged)
- individual configs: up to **6 points**, and rank order changes
  (`gpt_5_5_medium` +5.97, `gpt_5_4_xhigh` −3.76, `claude_opus_4_8_xhigh` −3.38)
- individual tasks: up to **67 points** (`vulture-persistent-analysis-cache`
  0.833 → 0.167; `narwhals-rolling-window-suite` 0.333 → 1.000)

The arXiv paper reports v1 numbers (gpt-5.5 at 70.0%); the live leaderboard is
v1.1 (the same config at 67.0%). Compare against one or the other, not a mix.

### Task difficulty spread makes small subsets noisy

Per-task pass rate across all published rollouts ranges from **2.5%**
(`obsidian-linter-auto-table-of-contents`) to **91.7%**
(`true-myth-iterable-collection-combinators`). Mean per-language:

| language | mean task pass rate | tasks |
| --- | ---: | ---: |
| go | 0.614 | 34 |
| python | 0.545 | 34 |
| typescript | 0.514 | 35 |
| javascript | 0.503 | 5 |
| rust | 0.487 | 5 |

A 10-task subset can move an apparent score by tens of points, which is why
`aggregate_bench.py` compares against published results *restricted to the same
tasks* rather than against the headline leaderboard number.

The seed alone is worth ±8 points at 12 tasks:

| seed | subset rate | vs full benchmark |
| ---: | ---: | ---: |
| 0 | 60.3% | +5.1 |
| 1 | 55.7% | **+0.5** |
| 4 | 46.7% | −8.5 |
| 8 | 60.9% | +5.7 |

`scripts/pick_subset.py` scans seeds and names the representative one — **seed 1
at 12 tasks** is the closest to the full benchmark's 55.2%.

### The exclusion policy hides provider failures

DeepSWE drops rollouts that hit provider errors, infrastructure timeouts or
grading errors, and does not resample them — 155 of 31,617 (0.49%), concentrated
in a few configs. That is the right call for ranking models and the wrong one
for judging a provider, since a `model_routing_404` or a `provider_timeout` is
exactly the failure you're shopping for. Published error categories:

`agent_timeout` 116, `model_routing_404` 73, `verifier_timeout` 38,
`provider_timeout` 35, `unclassified_exception` 5, `upstream_provider_error` 3,
`context_window_exceeded` 1, `rate_limit` 1.

### The HuggingFace mirror is still v1.0

`datacurve/deep-swe` on HuggingFace is gated, but its public file listing shows
what it contains:

| | HF mirror | GitHub v1.1 |
| --- | ---: | ---: |
| `tests/config.json` (node-id whitelists) | 0 | 113 |
| `tests/grader.py` | 0 | 113 |
| Dockerfiles | 113 | 226 (separate verifier env) |

Last modified 2026-06-02; the v1.1 task changes landed 2026-08-26. Pulling the
dataset from HuggingFace gets you exit-code grading — the regime with the
false-negative class above, and scores that differ from the live leaderboard.
Clone the GitHub repo at a pinned SHA instead, which is what these workflows do.

### Two smaller ones

- **Subset sampling isn't portable.** pier's `--n-tasks/--sample-seed` shuffles
  in `Path.iterdir()` order — filesystem order, not sorted — so the same seed
  can pick different tasks on different machines.
  `scripts/plan.py` sorts first.
- **A language tag is wrong.** `prometheus-transactional-reload-status` is
  tagged `typescript` in `manifest.json` while its repo is `prometheus/prometheus`
  (its sister task `prometheus-typed-label-sorting` is correctly tagged `go`).
  Matches [deep-swe#53](https://github.com/datacurve-ai/deep-swe/issues/53);
  it skews any per-language breakdown, including the table above.
- **~2.4% of published submissions aren't downloadable.** Of 250 sampled
  rollouts with `has_model_patch: true`, 6 return HTTP 403 from the artifact
  CDN (confirmed on retry, not throttling), and one is zero bytes. The regrade
  runner records these as skipped rather than scoring them 0. Reported upstream
  as [deep-swe#59](https://github.com/datacurve-ai/deep-swe/issues/59).
