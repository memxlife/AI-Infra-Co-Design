# EPFL: agent evolution and execution history

[English report](reports/comparison.en.md) · [中文报告](reports/comparison.zh-CN.md)

Search for one ABC logic-synthesis recipe that reduces mapped circuit area
while preserving equivalence and satisfying per-circuit delay caps. The same
recipe is used on all circuits. Four twenty-round runs compare:

| Arm | Agent guidance | Execution history | Final training score |
| --- | --- | --- | ---: |
| A | Evolving | Sampled | 110.070247 |
| B | Fixed | All prior completed workers | 114.890255 |
| C | Fixed | None supplied | 103.390170 |
| D | Evolving | All prior completed workers | 117.865078 |

Here, **agent** means the guidance called `optimizer` in EvE; a **worker** is
one rollout. The reports explain problem setup, configuration, the training
and equivalent-token curves, and validation/test results.

## Seed and best recipes

[Saved recipes](recipes/README.md) include the original `resyn2` seed and the
best formal-training solver from each of A–D. D has the highest training score,
117.865078. Each solver is a small JSON command list.

## Install and prepare

Requirements: Git, [uv](https://docs.astral.sh/uv/), a C/C++17 compiler and
`make`, plus an authenticated `codex` executable on PATH with access to the
configured model. On macOS, use Xcode Command Line Tools.

From the repository root:

```bash
cd eve-experiments/epfl
bash setup.sh
bash prepare.sh
```

`setup.sh` clones the public [EvE repository](https://github.com/scaling-group/eve)
into the ignored `.eve/` directory, adds this task's configs and code, and
installs the Python environment. `prepare.sh` downloads ABC, Nangate45 and
the six training circuits, builds single-threaded ABC and measures the
`resyn2` baselines. Prepare once, then use the same training data for A–D.
Re-running `setup.sh` copies edits to the task configs/code into the checkout.

## Run the four configurations

Preview a resolved configuration without calling a model or ABC:

```bash
bash run.sh D --cfg job --resolve
```

Run each command to completion before starting the next:

```bash
bash run.sh A
bash run.sh B
bash run.sh C
bash run.sh D
```

Each command starts a fresh 20-round run with two sequential workers per
round. EvE creates a separate run directory under `.eve/.runs/`; its console
output identifies that directory. These commands use model and ABC compute.
The original A/B runs were extended from five rounds; the commands here run
all twenty continuously with the same experimental settings.

| `loop` setting | A | B | C | D |
| --- | --- | --- | --- | --- |
| `produce_optimizer_in_phase2` | 2 | 0 | 0 | 2 |
| `n_optimizer_examples_phase2` | 4 | 4 | 0 | 100 |
| `optimizer_log_inheritance` | current | current | none | none |
| Reference sampler | Exclude all working | Exclude all working | Exclude all working | Exclude self |

Config files: [A](configs/eve/epfl_A.yaml), [B](configs/eve/epfl_B.yaml),
[C](configs/eve/epfl_C.yaml), [D](configs/eve/epfl_D.yaml).
[Common settings](configs/eve/epfl_common.yaml) use gpt-6-astra/xhigh,
24 turns and 1,200 seconds per rollout, up to four solver references, and
at most three local full-training evaluations per worker. The shared
[sampling configuration](configs/eve/loop/epfl_synthesis.yaml),
[initial guidance](configs/eve/optimizer/epfl_synthesis/initial_guidance/strategy.md),
[task instructions](examples/epfl_synthesis/TASK.md) and
[worker prompt](configs/eve/optimizer/epfl_synthesis/prompt/ENTRYPOINT.md)
are included.

In D, `none` starts a new agent entry with empty logs; completed executions
remain on the agent that performed them. `PerWorkerExcludeSelfSampler`,
with EvE adding the worker's own entry back, supplies every existing agent
and its history. The limit of 100 covers the maximum population of 41.
Both workers receive their references before the round starts.

## Evaluate a completed run

Substitute the run directory printed by EvE:

```bash
bash heldout.sh "$PWD/.eve/.runs/eve/epfl-synthesis/<run-id>"
```

This selects the highest formal training score, including seed, with the
earliest entry breaking ties; writes `best_recipe.json`; prepares validation
and test baselines; and evaluates that recipe on each split. Existing heldout
result files are reused. Run it after all twenty rounds have completed.
Results are saved under the run's `heldout/validation` and `heldout/test`.
A zero score can mean a delay-cap violation even when equivalence passed.

## Files

- `reports/`: bilingual four-arm comparison and PNG/SVG figures.
- `recipes/`: original seed and the best solver from each arm.
- `configs/eve/`: four arm configs and shared task/worker settings.
- `examples/epfl_synthesis/`: task definition, resource preparation and evaluator.
- `setup.sh`, `prepare.sh`, `run.sh`, `heldout.sh`: executable workflow.

The package contains the reports and runnable task, without original run
directories, worker logs or bulk result exports. New runs store their output
under the ignored `.eve/` checkout.

Task code is adapted from Scaling Group's EPFL experiment and is distributed
under the [Apache License 2.0](https://github.com/scaling-group/eve/blob/main/LICENSE). ABC, EPFL benchmarks and Nangate45
are downloaded with their accompanying licenses during preparation.
