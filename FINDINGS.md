# Findings

What the provider runs have shown so far. Each run's full report is in
[`runs/`](runs); the harness checks behind these numbers are in
[archive/2026-09-12-baseline](archive/2026-09-12-baseline/FINDINGS.md).

## crof.ai, glm-5.3 (2026-09-12)

Compared against DeepSWE's published `mini_swe_agent_glm_5_3_max` run (Z.AI's
own API, 69.0% over the full benchmark), with the same agent settings and the
same 90-minute agent timeout.

### It solves tasks like the published run

| run | tasks | passed | published run would pass |
| --- | --- | ---: | --- |
| [smoke](runs/2026-09-12-crof-glm-5.3-smoke.md) | `ytt-jsonpath-query-api` | 1 of 1 | 0.9 |
| [sentinel](runs/2026-09-12-crof-glm-5.3-sentinel.md) | 8 tasks the published run always passed | 8 of 8 | 7.3 (normal range 5.8–8.0) |

The sentinel tasks can only show a provider doing worse, which it didn't.
They can't show it doing better; the representative subset is for that.

### Per call, it behaves like Z.AI

Across the sentinel run's 669 model calls, next to the published run's 32
rollouts on the same tasks:

| per call | crof | Z.AI |
| --- | ---: | ---: |
| replies with no usable command | 0.1% | 0.0% |
| garbled characters per 100 calls | 16.0 | 16.7 |
| output tokens, typical / 90th percentile | 238 / 1,568 | 198 / 1,294 |
| prompt tokens served from cache | 98.1% | 98.2% |
| steps repeating a recent step | 0.6% | 0.7% |

Per rollout, the medians are within a few percent on output tokens (54k vs
56k), reasoning share, uncached input (118k vs 119k) and largest prompt (114k
vs 114k), and crof took fewer steps (87 vs 100). For the same history crof's
tokenizer counts the recorded prompt tokens plus a constant 71, so it renders
the conversation, including preserved reasoning, the way Z.AI does. No
retries, no API errors.

### It is slower, and slower still under load

- Agent time per rollout was 1.73x Z.AI's (median 36 vs 21 minutes). Nearly
  all of it is waiting on the model.
- Output tokens per second of waiting, time to first token included: 43 with
  one rollout running, 30 with eight rollouts running at once. The same task
  took 31 minutes both times, but its speed went from 43 to 34.
- Within a rollout, speed falls as context grows: 37 tokens/s in the first
  fifth of steps (typical prompt 24k tokens), 25–30 in the rest (61k–110k),
  even though 98–99% of the input is cached by then.
- None of the 9 rollouts came near the 90-minute timeout (longest 43 minutes),
  but a slower task under heavier load could.

### Its usage accounting is off

- `usage.reasoning_tokens` doesn't match the reasoning it returns (0.77x–1.10x
  when checked with the GLM-5.3 tokenizer), and can exceed
  `completion_tokens`, which does count reasoning. One command reproduces it;
  see [`repros/crof-reasoning-tokens.sh`](repros/crof-reasoning-tokens.sh).
- With a `tools` key and no streaming, which is how agents call it, there's
  no reasoning count at all. Without tools the response has a different id
  shape (`gen-…` instead of `chatcmpl-…`).
- `completion_tokens` and cost fields look right. Cost per rollout was $0.53
  at crof's prices against $2.23 at Z.AI's.
