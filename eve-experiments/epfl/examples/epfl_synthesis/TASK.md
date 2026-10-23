# Search for a logic synthesis recipe

Edit only `recipe.json`: a JSON object with one key, `commands`, containing a list
of 1–40 ABC synthesis steps. The same recipe runs on every circuit. Optimize
mapped standard-cell area without exceeding the frozen resyn2 baseline delay on
any training circuit. The score is the mean of `100 * baseline_area / area`;
any invalid output, unproved equivalence, timeout or delay violation makes the
whole evaluation score zero. Higher is better; baseline is 100.

Allowed steps (exact strings):

- `balance`, `rewrite`, `rewrite -z`, `refactor`, `refactor -z`
- `resub`, `resub -z`
- `resub -K 6`, `resub -K 8`, `resub -K 10`, `resub -K 12`, `resub -K 16`
- each of those five `resub -K N` forms with ` -z` appended

The evaluator fixes `strash`, mapping with `map`, the Nangate45 typical library,
independent `stime` measurement and combinational equivalence checking (CEC).
No file paths, shell commands, aliases, settings or mapping commands are allowed
in the recipe. Each circuit has a shared 30-second wall budget for synthesis,
mapping, measurement, CEC and result acceptance. Input staging and hash validation
are excluded and identical for all recipes. ABC is compiled without pthreads.
An output only counts when it is fully verified before the parent deadline.

From the agent workspace, use the fixed repository evaluator:

```bash
python -m examples.epfl_synthesis.evaluate --recipe solver/recipe.json --data solver/train --output logs/optimize/trial_1
```

Run at most three local evaluations per rollout. Save every attempted recipe with
its results. At least one check must cover all six training circuits. Update
`guidance/strategy.md` with useful search advice; do not memorize benchmark answers.
Read the sampled candidates and guidance before proposing a change. Inspect all
per-circuit results, including a failed delay cap, before drawing conclusions.
Do not inspect files outside this workspace or access held-out data. Do not launch
other agents, download tools, change the evaluator or run parallel measurements.
