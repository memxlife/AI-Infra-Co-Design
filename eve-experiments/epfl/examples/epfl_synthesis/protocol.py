"""Fixed search space and scoring rules for the EPFL pilot."""

import math

ABC_COMMIT = "a3001b72edc5de22442e942165487fddf150e3d0"
SECONDS = 30.0
SPLITS = {
    "train": ["adder", "bar", "max", "arbiter", "i2c", "priority"],
    "validation": ["sin", "cavlc", "dec"],
    "test": ["square", "multiplier", "voter"],
    "calibration": ["ctrl", "int2float", "router"],
}
# Expand aliases explicitly; ABC runs with startup files disabled (-s).
RESYN2 = [
    "balance",
    "rewrite",
    "refactor",
    "balance",
    "rewrite",
    "rewrite -z",
    "balance",
    "refactor -z",
    "rewrite -z",
    "balance",
]
ALLOWED = frozenset(
    [
        "balance",
        "rewrite",
        "rewrite -z",
        "refactor",
        "refactor -z",
        "resub",
        "resub -z",
        "resub -K 6",
        "resub -K 8",
        "resub -K 10",
        "resub -K 12",
        "resub -K 16",
        "resub -K 6 -z",
        "resub -K 8 -z",
        "resub -K 10 -z",
        "resub -K 12 -z",
        "resub -K 16 -z",
    ]
)


def validate_recipe(value):
    if not isinstance(value, dict) or set(value) != {"commands"}:
        raise ValueError('Recipe must contain exactly one field: "commands"')
    commands = value["commands"]
    if not isinstance(commands, list) or not 1 <= len(commands) <= 40:
        raise ValueError("Use 1 to 40 commands")
    if any(not isinstance(command, str) or command not in ALLOWED for command in commands):
        raise ValueError("Command outside the fixed synthesis allowlist")
    return commands


def score_case(area, delay, reference_area, delay_cap):
    values = (area, delay, reference_area, delay_cap)
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Metrics must be finite and positive")
    # ABC prints delay to two decimals. No additional performance slack is allowed.
    if delay > delay_cap + 1e-9:
        return 0.0
    return 100.0 * reference_area / area
