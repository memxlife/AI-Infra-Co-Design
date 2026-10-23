"""Evaluate a fixed ABC recipe; only the parent owns files, commands and deadlines."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import yaml

from examples.epfl_synthesis.protocol import SECONDS, score_case, validate_recipe

REPO = Path(__file__).resolve().parents[2]
METRIC = re.compile(r"Area\s*=\s*([0-9.eE+-]+).*?Delay\s*=\s*([0-9.eE+-]+)")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Hash mismatch: {path}")


def parse_metrics(text):
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    found = METRIC.findall(clean)
    if len(found) != 1:
        raise ValueError("Expected exactly one ABC stime area/delay record")
    area, delay = map(float, found[0])
    if not all(math.isfinite(x) and x > 0 for x in (area, delay)):
        raise ValueError("Invalid area/delay")
    return area, delay


def equivalent(text):
    # ABC can exit zero after a command fails, so require affirmative evidence.
    return bool(re.search(r"Networks are equivalent\b", text)) and not re.search(
        r"NOT EQUIVALENT|not equivalent|verification failed", text, re.I
    )


def abc_call(abc, directory, commands, deadline, log):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Shared circuit deadline exhausted")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with log.open("w") as stream:
        process = subprocess.Popen(
            [str(abc), "-s", "-c", commands],
            cwd=directory,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            process.wait(timeout=remaining)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if time.monotonic() >= deadline:
        raise TimeoutError("Result arrived after shared circuit deadline")
    if process.returncode:
        raise RuntimeError(f"ABC failed with status {process.returncode}")
    return log.read_text(errors="replace")


def run_circuit(abc, library, circuit, commands, output, seconds=SECONDS):
    """Stage immutable inputs first. Synthesis, remapping checks and CEC share time."""
    validate_recipe({"commands": commands})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {"accepted": False, "equivalent": False, "error": None}
    with tempfile.TemporaryDirectory(prefix="epfl-abc-") as tmp:
        directory = Path(tmp)
        shutil.copyfile(circuit, directory / "input.aig")
        shutil.copyfile(library, directory / "cells.lib")
        start = time.monotonic()
        deadline = start + seconds
        try:
            # All command fragments are fixed or exact allowlist members.
            abc_call(
                abc,
                directory,
                "read_lib cells.lib; read_aiger input.aig; strash; "
                + "; ".join(commands)
                + "; map; write_blif mapped.blif",
                deadline,
                output / "synthesis.log",
            )
            mapped = directory / "mapped.blif"
            if not mapped.is_file() or mapped.stat().st_size == 0:
                raise ValueError("ABC did not produce a mapped netlist")
            text = abc_call(
                abc,
                directory,
                "read_lib cells.lib; read_blif mapped.blif; topo; stime",
                deadline,
                output / "measurement.log",
            )
            area, delay = parse_metrics(text)
            proof = abc_call(
                abc,
                directory,
                "read_lib cells.lib; cec input.aig mapped.blif",
                deadline,
                output / "equivalence.log",
            )
            if not equivalent(proof):
                raise ValueError("Equivalence was not proved")
            # Include artifact read/hash in acceptance deadline; log persistence is outside it.
            netlist = mapped.read_bytes()
            mapped_hash = hashlib.sha256(netlist).hexdigest()
            elapsed = time.monotonic() - start
            if elapsed >= seconds:
                raise TimeoutError("Verified result arrived after deadline")
            result.update(
                accepted=True,
                equivalent=True,
                area=area,
                delay=delay,
                accepted_seconds=elapsed,
                netlist_sha256=mapped_hash,
            )
            (output / "mapped.blif").write_bytes(netlist)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_seconds"] = time.monotonic() - start
    result["budget_seconds"] = seconds
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def evaluate(recipe, data, tools, output):
    commands = validate_recipe(json.loads(Path(recipe).read_text()))
    data, tools, output = Path(data), Path(tools), Path(output)
    manifest = json.loads((data / "manifest.json").read_text())
    toolchain = json.loads((tools / "toolchain.json").read_text())
    abc = (tools / toolchain["abc"]).resolve()
    library = tools / toolchain["library"]
    checked(abc, toolchain["abc_sha256"])
    checked(library, toolchain["library_sha256"])
    rows = []
    for entry in manifest["circuits"]:
        circuit = data / f"{entry['name']}.aig"
        checked(circuit, entry["sha256"])
        row = run_circuit(abc, library, circuit, commands, output / entry["name"])
        row.update(name=entry["name"], reference_area=entry["area"], delay_cap=entry["delay"])
        row["delay_feasible"] = row["accepted"] and row["delay"] <= row["delay_cap"] + 1e-9
        row["score"] = (
            score_case(row["area"], row["delay"], entry["area"], entry["delay"])
            if row["accepted"]
            else 0.0
        )
        rows.append(row)
    valid = bool(rows) and all(row["accepted"] and row["delay_feasible"] for row in rows)
    return dict(
        score=statistics.mean(row["score"] for row in rows) if valid else 0.0,
        valid=valid,
        summary="100 × baseline area / area; all delay caps and equivalence checks required",
        candidate_sha256=digest(recipe),
        data_manifest_sha256=digest(data / "manifest.json"),
        toolchain=toolchain,
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tools", default=str(REPO / ".runs/epfl-synthesis-tools"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = evaluate(args.recipe, args.data, args.tools, output)
    except Exception:
        result = dict(score=0.0, valid=False, error=traceback.format_exc())
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k not in ("rows", "toolchain")}
    (output / "score.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    print(json.dumps(summary))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
