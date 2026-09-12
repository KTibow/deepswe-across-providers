# Baseline, 2026-09-12

Before measuring any provider, this repo checked that DeepSWE runs correctly on
a GitHub runner at all. Those checks are done and their conclusions are in
[FINDINGS.md](FINDINGS.md): 113/113 reference solutions pass, 12/12 empty
submissions fail, our grader agrees with Datacurve's on their own submitted
patches, and the published leaderboard follows from the published rollouts.

The workflows and scripts that produced those runs are kept here as they were.
They no longer run from this directory; the links in FINDINGS.md point at the
original Actions runs. The live pipeline replaced them:

| then | now |
| --- | --- |
| `replay.yml` (oracle / nop) | `run.yml` with `agent=oracle` or `agent=nop` |
| `regrade.yml` (apply a published patch) | `run.yml` with `agent=replay`, which replays the whole published trajectory rather than just its final patch |
| `bench.yml` (a model through mini-swe-agent) | `run.yml` with `agent=<provider>:<model>` |
| `pick_subset.py` | `python3 -m dswe.subset` |
| `analyze_published.py` | not replaced; the leaderboard check was a one-off |
