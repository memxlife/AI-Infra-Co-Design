# Seed and best solver recipes

Each JSON file is a complete solver: one ordered list of ABC synthesis commands.
The seed is the original ten-command `resyn2` recipe used by all four runs.
Each arm's winner is selected by its highest formal training score across all
twenty rounds, including seed; the earliest generated candidate breaks ties.

| Recipe | Configuration | Commands | Training score | Round / worker |
| --- | --- | ---: | ---: | --- |
| [Seed](seed.json) | Original resyn2 | 10 | 100.000000 | — |
| [A](best_A.json) | Evolving agent + sampled history | 27 | 110.070247 | 20 / 1 |
| [B](best_B.json) | Fixed agent + all history | 32 | 114.890255 | 20 / 1 |
| [C](best_C.json) | Fixed agent + no history | 14 | 103.390170 | 19 / 1 |
| [D](best_D.json) | Evolving agent + all history | 40 | 117.865078 | 18 / 2 |

**D has the highest formal training score across the four runs.** Worker numbers
above start at 1. Validation and test results are in the [English report](../reports/comparison.en.md)
and [中文报告](../reports/comparison.zh-CN.md).

The files use the same `{"commands": [...]}` format accepted by the
[evaluator](../examples/epfl_synthesis/evaluate.py). These saved winners are for
inspection and reuse; the four experiment configurations continue to start from
the original seed.
