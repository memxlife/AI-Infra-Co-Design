# EPFL: agent evolution and execution history

Four twenty-round experiments · 11 October 2026 · [中文](comparison.zh-CN.md) · [Run instructions](../README.md) · [Seed and best recipes](../recipes/README.md)

## Problem setup

### Task and search space

We search for an **ABC logic-synthesis recipe that reduces mapped standard-cell area while preserving circuit functionality and meeting fixed delay limits** on a selected subset of the EPFL benchmarks. Each input is a circuit in AIG format. A candidate is a JSON file containing one ordered list of **1–40 synthesis commands**. The same list is applied unchanged to every circuit in every split; there are no circuit-specific branches or recipes.

The search can change the order, repetition and allowed options of `balance`, `rewrite`, `refactor` and `resub`, including their whitelisted `-z` variants and resubstitution cut sizes `-K 6/8/10/12/16`. The evaluator fixes input loading, technology mapping (`map`), measurement and equivalence checking. It uses ABC and the Nangate45 cell library. Area and delay are measured after logic synthesis and technology mapping; physical placement and routing are outside this experiment. The exact [search space and scoring rules](../examples/epfl_synthesis/protocol.py) and [evaluator](../examples/epfl_synthesis/evaluate.py) are shared by all four arms.

### Data, baseline and objective

| Split | Circuits | Use |
| --- | --- | --- |
| Training | adder, bar, max, arbiter, i2c, priority | Feedback during recipe search and formal candidate ranking |
| Validation | sin, cavlc, dec | Evaluate the locked final training winner |
| Test | square, multiplier, voter | Evaluate the same locked final training winner |

Every run starts with the same ten-command `resyn2` seed recipe and the same initial agent guidance. For each circuit, its frozen `resyn2` result supplies both the reference area and the maximum allowed delay. A candidate must prove combinational equivalence (CEC), return finite area/delay metrics, and satisfy that circuit's delay cap. Synthesis, measurement in a fresh ABC process, equivalence checking and result acceptance share **30 seconds per circuit, single-threaded**.

For an evaluated circuit set C, a feasible recipe P scores:

`score_C(P) = (100 / |C|) × sum over c in C of (resyn2_area_c / candidate_area_c)`.

If **any** circuit fails equivalence, the time budget, measurement validity or its delay cap, the **whole set scores zero**. Thus training averages six ratios and each heldout split averages three; `resyn2` scores 100. Higher is better. A ten-point score gain is not a ten-percent area reduction, because the objective averages inverse area ratios.

### Search and evaluation procedure

Each arm runs **20 rounds × 2 serial workers**, using gpt-6-astra/xhigh with at most **24 turns and 1,200 seconds per rollout**. Each worker receives up to four previous solver recipes with formal feedback, plus the guidance/history context determined by its arm. It may run up to **three full six-circuit training evaluations during optimize**, inspect the measurements, revise the recipe and select its submission. After the worker finishes, the framework evaluates that submission again for its formal training score. Both workers' inputs are selected before the round, so the second does not receive the first worker's new results from that round.

The forty completed workers therefore account for up to **120 local full-training evaluations**, **40 formal candidate evaluations** and **one formal seed evaluation**. “Training” names the circuits used for search feedback; the language model's weights remain fixed. The evolving objects are recipes and, in A/D, agent guidance. Local trial feedback is available before submission, which is why saved optimize logs can be valuable to later workers.

The final recipe is chosen solely by the highest formal training score, including seed, with earliest generation breaking ties. It is locked before final validation/test evaluation; those results do not guide further search. Identical previously measured recipes reuse their heldout results, including failures; a new winner is measured once per heldout split. These are designer-seen comparison sets. The four arms vary guidance evolution and access to prior execution history under this common task and budget; the configuration section below specifies the differences.

## Results

All four runs completed twenty rounds and forty formal candidates. **D had the
highest training score, 117.865078**, compared with B's 114.890255, A's 110.070247
and C's 103.390170. B retained the slightly higher test score: 102.447260 versus
D's 102.175127. Each configuration was run once.

The labels use **agent** for the guidance called `optimizer` in EvE, and **worker**
for a single rollout:

- **A: Evolving agent + sampled history**
- **B: Fixed agent + all history**
- **C: Fixed agent + no history**
- **D: Evolving agent + all history**

| Measure | A: Evolving agent + sampled history | B: Fixed agent + all history | C: Fixed agent + no history | D: Evolving agent + all history |
| --- | --- | --- | --- | --- |
| Rounds included | 20/20 | 20/20 | 20/20 | 20/20 |
| Best training score after round 5 | 106.682500 | 106.526298 | 103.262182 | 105.331594 |
| Best training score after round 20 | 110.070247 | 114.890255 | 103.390170 | **117.865078** |
| Round 5→20 score gain | +3.387747 | +8.363957 | +0.127988 | **+12.533484** |
| Five-round winner: validation / test | 0 / 102.656734 | 0 / 102.656734 | Not measured / Not measured | Not measured / Not measured |
| Final winner: validation / test | 0 / 0 | 0 / 102.447260 | 0 / 0 | 0 / 102.175127 |
| Final training winner: mean per-circuit area reduction | 7.6324% | 11.0924% | 3.0516% | 11.6349% |
| Final training winner: summed area reduction | 3.3494% | 4.6092% | 1.1400% | 4.3502% |

The training score averages inverse area ratios. Mean area reduction is
`mean(100 × (1 − candidate_area / baseline_area))`; summed area reduction uses
the sums of circuit areas. Score gains and area reductions are different quantities.

## Training trajectories

![Best formal training score by round](figures/training-trajectories.png)

[Vector SVG](figures/training-trajectories.svg). Each curve is the maximum over
seed and all formal submissions through that round, combining both workers.
Round 0 is seed, with score 100; the y-axis starts at 99. The figures contain
training scores only.

![Best formal training score by equivalent tokens](figures/training-vs-equivalent-tokens.png)

[Vector SVG](figures/training-vs-equivalent-tokens.svg). Each point is a completed
round; its x-value accumulates both workers' token usage. We use
[EvE Appendix A.1](https://arxiv.org/html/2605.09018v4#A1):

`equivalent tokens = cached input + 2 × fresh input + 12 × output`

Native input already includes cached tokens, so this is also
`2 × input − cache + 12 × output`. Reasoning tokens are already included in output.
This is the paper's comparison scale, not the model's actual price; ABC compute
is separate.

| Equivalent-token cost at the plotted endpoint | A: 20 rounds | B: 20 rounds | C: 20 rounds | D: 20 rounds |
| --- | ---: | ---: | ---: | ---: |
| Completed workers, summed by round | 39,438,854 | 37,304,414 | 23,446,352 | 37,986,238 |
| Retained interrupted attempt | 0 | 499,012 | 0 | 0 |
| Cumulative cost plotted | **39,438,854** | **37,803,426** | **23,446,352** | **37,986,238** |

The last cumulative native counter in each session was summed across sessions
and workers; optimizer CSV usage was not added again. B's stopped attempt adds
499,012 equivalent tokens once between rounds 5 and 6 without creating a score.
C's earlier cancelled and deleted attempt is outside the plotted run; its
unavailable cost is not assumed to be zero.

## Configuration differences

All fields in the table are under `loop`. Configuration files are
[A](../configs/eve/epfl_A.yaml), [B](../configs/eve/epfl_B.yaml),
[C](../configs/eve/epfl_C.yaml) and [D](../configs/eve/epfl_D.yaml).
[Common settings](../configs/eve/epfl_common.yaml) and the
[sampling configuration](../configs/eve/loop/epfl_synthesis.yaml) are shared.

| Configuration | A: Evolving agent + sampled history | B: Fixed agent + all history | C: Fixed agent + no history | D: Evolving agent + all history |
| --- | --- | --- | --- | --- |
| `produce_optimizer_in_phase2` | 2 | 0 | 0 | **2** |
| `n_optimizer_examples_phase2` | 4 | 4 | 0 | **100** |
| `optimizer_log_inheritance` | current | current | none | **none** |
| Optimizer-reference sampler | `PerWorkerExcludeAllWorkingSampler` | Same as A | Same as A, but no references | **`PerWorkerExcludeSelfSampler`** |
| `n_solver_examples_phase2` | 4 | 4 | 4 | 4 |
| `n_workers_phase2` / `n_parallel_phase2` | 2 / 1 | 2 / 1 | 2 / 1 | 2 / 1 |
| `max_iterations` (target) | 20 | 20 | 20 | 20 |

`produce_optimizer_in_phase2` controls admission of edited guidance into the
agent population. At zero, only the original seed remains, even if a worker
edits its own local guidance. At two, up to two changed guidance trees can enter
per round.

`n_optimizer_examples_phase2` controls references supplied to workers. A count
of four includes the worker's own agent plus up to three sampled alternatives.
Each reference includes guidance, logs and score. B's single agent is supplied
once with its accumulated history. C's zero removes the reference directory,
including its own agent's history, while retaining working guidance and solver
references.

`optimizer_log_inheritance=current` gives a new agent the execution that created
it, without its parent's full older history. `none` starts a new agent with empty
logs. In both cases, each completed execution is appended to the agent actually
used. These settings govern a new entry's initial logs, separately from reference
access.

D uses `PerWorkerExcludeSelfSampler`: the sampler excludes only the worker's
own agent, which EvE then adds back. A's sampler excludes all working agents,
so raising only the reference count can still omit the other worker's agent.
D's limit of 100 covers the maximum population of 41. Gathering all entries
provides each preceding execution under its owner without copying it into new
entries. It also supplies all guidance texts, making D a change to guidance
coverage as well as history coverage.

The [public EvE reference sampler](https://github.com/scaling-group/eve/blob/main/src/scaling_evolve/algorithms/eve/populations/samplers/optimizer_examples.py)
and [Phase 2 workflow](https://github.com/scaling-group/eve/blob/main/src/scaling_evolve/algorithms/eve/workflow/phase2.py)
implement these mechanisms.

## Why execution history matters here

**EPFL workers can obtain complete training scores during optimize and then
revise the recipe.** A worker may run three complete six-circuit evaluations,
inspect area, delay and equivalence results, and choose its submission. The
framework evaluates that submission again after the rollout. The
[task instructions](../examples/epfl_synthesis/TASK.md) and
[formal evaluation entrypoint](../configs/eve/evaluation/epfl_synthesis/evaluation.sh)
use the same evaluator and training circuits.

B's round 17 worker 2 tested three prefixes on an established thirty-command
recipe and knew their full scores before submitting:

| Local trial | Prepended command | Six-circuit training score |
| --- | --- | ---: |
| 1 | `resub -K 6` | 113.413549 |
| 2 | `resub -K 8` | 113.501943 |
| 3 | `resub -K 10` | 114.335994 |

It selected `resub -K 10`; formal evaluation also scored 114.335994, improving
on the previous 109.760312. Its local note described trying a small resubstitution
step before the established flow once suffix edits plateaued.

An optimize execution record can contain submitted and unsubmitted recipes,
local measurements, failed scores, research notes and subsequent formal
evaluation. Solver references supply the selected recipe and formal feedback,
without automatically transferring the other local trials. At the start of
round 20, B's seed held 38 completed-worker records containing 114 local full
training evaluations, including 25 zeros, and 38 formal evaluations.

B therefore couples fixed guidance with accumulated measured experience.
Later workers can consult successful changes and failed alternatives. B and D
performed well in this setting, while C evaluated one identical failed recipe
nine times in rounds 13–19. This supports the usefulness of retaining evaluation
experience in these observed runs, without isolating history as the sole cause.

The timing of feedback differs from ICON in the EvE paper. Its
[worker instructions](https://github.com/scaling-group/eve/blob/main/configs/eve/optimizer/icon/prompt/ENTRYPOINT.md)
prohibit worker-driven eval-edit loops; the
[check runner](https://github.com/scaling-group/eve/blob/main/configs/eve/optimizer/icon/immutable/check_runner/check.sh)
defaults to 50 steps, while the
[formal evaluation](https://github.com/scaling-group/eve/blob/main/configs/eve/evaluation/icon/evaluation.sh)
uses 2,000 training steps. EPFL permits complete objective queries during optimize.
Its logs can thus carry immediately reusable scores for alternatives that were
never formally submitted. Log sharing is already part of EvE; the distinctive
feature here is when complete feedback becomes available.

## What workers could access each round

Both workers' references are selected before the round starts. Serial execution
does not give worker 2 access to worker 1's new results from that same round.
All arms retain up to four solver references and within-rollout trial feedback.
B and C always start from original guidance; their edited local guidance files
do not become later workers' initial guidance.

The table counts distinct prior execution subtrees supplied through agent
references, shown as worker 1 / worker 2. It counts neither local evaluations nor
agent entries.

| Round | A: Evolving agent + sampled history W1 / W2 | B: Fixed agent + all history W1 / W2 | C: Fixed agent + no history W1 / W2 | D: Evolving agent + all history W1 / W2 |
| --- | ---: | ---: | ---: | --- |
| 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 2 | 2 / 2 | 2 / 2 | 0 / 0 | 2 / 2 |
| 3 | 4 / 4 | 4 / 4 | 0 / 0 | 4 / 4 |
| 4 | 6 / 6 | 6 / 6 | 0 / 0 | 6 / 6 |
| 5 | 7 / 7 | 8 / 8 | 0 / 0 | 8 / 8 |
| 6 | 6 / 6 | 10 / 10 | 0 / 0 | 10 / 10 |
| 7 | 6 / 6 | 12 / 12 | 0 / 0 | 12 / 12 |
| 8 | 5 / 5 | 14 / 14 | 0 / 0 | 14 / 14 |
| 9 | 7 / 7 | 16 / 16 | 0 / 0 | 16 / 16 |
| 10 | 8 / 4 | 18 / 18 | 0 / 0 | 18 / 18 |
| 11 | 4 / 7 | 20 / 20 | 0 / 0 | 20 / 20 |
| 12 | 4 / 7 | 22 / 22 | 0 / 0 | 22 / 22 |
| 13 | 4 / 5 | 24 / 24 | 0 / 0 | 24 / 24 |
| 14 | 4 / 6 | 26 / 26 | 0 / 0 | 26 / 26 |
| 15 | 5 / 4 | 28 / 28 | 0 / 0 | 28 / 28 |
| 16 | 6 / 4 | 30 / 30 | 0 / 0 | 30 / 30 |
| 17 | 7 / 7 | 32 / 32 | 0 / 0 | 32 / 32 |
| 18 | 8 / 5 | 34 / 34 | 0 / 0 | 34 / 34 |
| 19 | 7 / 7 | 36 / 36 | 0 / 0 | 36 / 36 |
| 20 | 8 / 6 | 38 / 38 | 0 / 0 | 38 / 38 |

A's history is partial and can differ between workers. B and D supply all
preceding completed workers' histories. C supplies none of those agent logs,
but still evolves its solver population. All D workers from round 2 onward
queried historical content before local evaluation. Availability is distinct
from complete reading: workers can filter or summarize files, and tool output
can be truncated. A historical-sounding note in B does not demonstrate direct
inheritance of its author's local guidance file.

## Shared sampling

| Sampling site | Shared configuration |
| --- | --- |
| Working optimizer | `RankSoftmaxSampler`, temperature 1, replacement `auto` |
| Solver references | `RankExponentialSumSampler`, score weight 1 / temperature 1, no replacement |
| Solver prefill | `UniformSampler` over solver references, no replacement |
| Additional optimizer references | `PerWorkerExcludeAllWorkingSampler`, rank-softmax base / temperature 1 / no replacement; C takes none; D uses `PerWorkerExcludeSelfSampler` |
| Produced optimizers | `RankExponentialSumSampler`, score weight 1 / temperature 1, no replacement; B/C admit none |

All arms use gpt-6-astra/xhigh, 24 turns and 1,200 seconds per rollout. Each of
the forty workers used three local training evaluations: 120 per arm, plus
forty formal candidate evaluations and one formal seed evaluation. Worker web
search and multi-agent execution were disabled.

## Later improvements and heldout results

D reached 105.331594 at round 5, 112.304555 at round 10, 116.321806 at round 12
and 117.865078 at round 18 / worker 2; rounds 19–20 added no new best. Its final
improvement replaced a small-cut resubstitution with a rewrite, with local scores
117.327500, 117.661363 and 117.865078 across three alternatives.

B plateaued at 109.760312 in rounds 9–16 before its round-17 prefix change and
finished at 114.890255. C plateaued in rounds 8–18, then reached 103.390170 after
adding another K6 resubstitution step in round 19.

All 48 final circuit measurements proved equivalence and completed within 30
seconds. The zero heldout scores came from delay violations: all arms exceeded
the validation caps on sin and cavlc; A exceeded the test cap on square, and C
on square and multiplier. B and D passed all test caps.

### Cumulative best formal training score

| Round | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| 0 | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| 1 | 100.124174 | 100.124174 | 100.000000 | 100.000000 |
| 2 | 100.124174 | 100.124174 | 102.621234 | 100.786556 |
| 3 | 101.434898 | 105.324724 | 103.262182 | 100.786556 |
| 4 | 104.083601 | 106.526298 | 103.262182 | 100.824982 |
| 5 | 106.682500 | 106.526298 | 103.262182 | 105.331594 |
| 6 | 106.682500 | 107.537503 | 103.262182 | 105.376849 |
| 7 | 107.720346 | 109.223500 | 103.325944 | 106.375825 |
| 8 | 107.720346 | 109.223500 | 103.325944 | 106.506241 |
| 9 | 108.639811 | 109.760312 | 103.325944 | 106.506241 |
| 10 | 108.906939 | 109.760312 | 103.325944 | 112.304555 |
| 11 | 108.906939 | 109.760312 | 103.325944 | 113.025348 |
| 12 | 108.954163 | 109.760312 | 103.325944 | 116.321806 |
| 13 | 108.954163 | 109.760312 | 103.325944 | 117.304183 |
| 14 | 108.954163 | 109.760312 | 103.325944 | 117.304183 |
| 15 | 108.954163 | 109.760312 | 103.325944 | 117.327500 |
| 16 | 108.954163 | 109.760312 | 103.325944 | 117.549218 |
| 17 | 108.954163 | 114.335994 | 103.325944 | 117.549218 |
| 18 | 109.207130 | 114.335994 | 103.325944 | 117.865078 |
| 19 | 110.053049 | 114.822709 | 103.390170 | 117.865078 |
| 20 | 110.070247 | 114.890255 | 103.390170 | 117.865078 |

## Circuit measurements

Each cell is **area / delay in ps / acceptance time in seconds**; bold cells violate the fixed delay cap. All 48 final circuit measurements proved equivalence and completed within 30 seconds. Baseline areas and caps match across arms; values come from original batches, with no extra evaluation.

### train

| Circuit | Baseline area | Delay cap (ps) | A | B | C | D |
| --- | ---: | ---: | --- | --- | --- | --- |
| adder | 882.06 | 4081.94 | 846.15 / 3720.63 / 0.446 | 718.20 / 4067.68 / 0.681 | 775.92 / 4072.39 / 0.449 | 718.20 / 4067.68 / 0.553 |
| bar | 2199.82 | 1334.94 | 2159.39 / 1329.74 / 0.672 | 2159.39 / 1329.74 / 1.225 | 2199.82 / 1334.94 / 0.833 | 2149.28 / 1316.67 / 0.769 |
| max | 2256.21 | 6387.22 | 2070.54 / 5189.61 / 0.707 | 2070.54 / 5189.61 / 1.040 | 2256.21 / 6387.22 / 0.887 | 2148.22 / 5462.00 / 0.769 |
| arbiter | 6837.80 | 1644.07 | 6837.80 / 1644.07 / 1.800 | 6837.80 / 1644.07 / 2.960 | 6837.80 / 1644.07 / 1.766 | 6837.80 / 1644.07 / 2.352 |
| i2c | 778.58 | 361.30 | 771.67 / 337.50 / 0.495 | 748.79 / 349.12 / 0.739 | 731.77 / 361.30 / 0.616 | 771.13 / 307.23 / 0.553 |
| priority | 601.69 | 2646.51 | 416.56 / 1493.90 / 0.500 | 396.61 / 1229.00 / 0.674 | 600.10 / 2635.58 / 0.550 | 341.81 / 1196.43 / 0.488 |

### validation

| Circuit | Baseline area | Delay cap (ps) | A | B | C | D |
| --- | ---: | ---: | --- | --- | --- | --- |
| sin | 4134.97 | 4390.96 | **4231.26 / 4522.08 / 1.790** | **4191.63 / 4653.42 / 3.758** | **4133.64 / 4413.54 / 2.209** | **4153.32 / 4424.04 / 2.001** |
| cavlc | 425.33 | 347.55 | **421.08 / 378.66 / 0.452** | **420.28 / 390.00 / 0.714** | **424.80 / 349.07 / 0.548** | **426.40 / 355.08 / 0.570** |
| dec | 254.03 | 190.63 | 254.03 / 190.63 / 0.436 | 254.03 / 190.63 / 0.552 | 254.03 / 190.63 / 0.502 | 254.03 / 190.63 / 0.442 |

### test

| Circuit | Baseline area | Delay cap (ps) | A | B | C | D |
| --- | ---: | ---: | --- | --- | --- | --- |
| square | 12559.99 | 4158.49 | **12311.81 / 4181.33 / 9.421** | 12369.53 / 4151.10 / 12.038 | **12360.49 / 4166.82 / 10.188** | 12583.93 / 3892.43 / 10.155 |
| multiplier | 18698.20 | 5268.03 | 18927.50 / 5226.12 / 6.212 | 18935.48 / 5192.23 / 11.688 | **18607.23 / 5269.14 / 5.904** | 18753.27 / 5125.87 / 7.126 |
| voter | 9969.68 | 1298.20 | 9398.84 / 1215.37 / 4.237 | 9312.66 / 1180.28 / 6.015 | 9383.68 / 1224.33 / 3.315 | 9316.65 / 1270.51 / 4.096 |

## Cost and interpretation

| Native tokens for forty completed candidates | A | B | C | D |
| --- | --- | --- | --- | --- |
| Input | 32,766,125 | 31,242,631 | 18,620,312 | 31,443,553 |
| Output | 348,505 | 321,180 | 269,368 | 360,821 |
| Cache (already included in input) | 30,275,456 | 29,035,008 | 17,026,688 | 29,230,720 |

Cache is included in input. B also used input 393,300 / output 3,757 / cache
332,672 in its stopped attempt. A/B resumed after five rounds; C/D completed
continuously. The scripts in this package launch fresh twenty-round runs.

D ended with the highest training score at 37.99M equivalent tokens, compared
with B's 37.80M including interruption, A's 39.44M and C's 23.45M. B retained the
slightly higher test score. Both full-history conditions reuse evaluation
experience produced inside optimize. The experiment has one run per condition;
D also broadens guidance coverage and C removes a reference directory. These
results do not separately identify the causal effects of guidance evolution and
history access.
