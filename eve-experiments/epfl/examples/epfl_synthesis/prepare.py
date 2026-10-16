"""Download task assets, build ABC locally, and prepare resyn2 baselines."""

import argparse
import datetime
import json
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from examples.epfl_synthesis.evaluate import REPO, digest, run_circuit
from examples.epfl_synthesis.protocol import ABC_COMMIT, RESYN2, SECONDS, SPLITS

SOURCE = Path(__file__).resolve().parent


def fetch_assets(tools, splits):
    urls = json.loads((SOURCE / "assets.json").read_text())
    requested = {name for split in splits for name in SPLITS[split]}
    tools.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, url in urls.items():
        if name.endswith(".aig") and Path(name).stem not in requested:
            continue
        dest = tools / name
        if not dest.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            temporary = tools / (name + ".download")
            temporary.write_bytes(payload)
            temporary.rename(dest)
        records[name] = {"sha256": digest(dest)}
    return records


def build_abc(tools):
    directory = tools / f"abc-{ABC_COMMIT}"
    binary = directory / "abc"
    args = ["make", "-j2", "ABC_USE_NO_READLINE=1", "ABC_USE_NO_PTHREADS=1", "ABC_MAKE_NO_DEPS=1"]
    # Apple CLT installations can place libc++ headers only inside the SDK.
    if platform.system() == "Darwin":
        sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        headers = Path(sdk) / "usr/include/c++/v1"
        if headers.is_dir():
            args.append(f"CXX=clang++ -isystem {headers}")
    if not binary.exists():
        if not directory.exists():
            with tarfile.open(tools / "abc.tar.gz") as archive:
                archive.extractall(tools, filter="data")
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        with (tools / f"build-{stamp}.log").open("w") as log:
            subprocess.run(args, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
    record = dict(
        abc=str(binary.relative_to(tools)),
        abc_sha256=digest(binary),
        abc_source_commit=ABC_COMMIT,
        build_command=args,
        library="NangateOpenCellLibrary_typical.lib",
        library_sha256=digest(tools / "NangateOpenCellLibrary_typical.lib"),
        platform=platform.platform(),
        machine=platform.machine(),
        python=platform.python_version(),
        threading="ABC compiled with ABC_USE_NO_PTHREADS=1",
    )
    manifest = tools / "toolchain.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        for key in ("abc_sha256", "library_sha256", "abc_source_commit"):
            if previous[key] != record[key]:
                raise ValueError(
                    "Toolchain changed; preserve the old experiment before preparing a new one"
                )
        return previous
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    return record


def prepare(tools, data, task, splits):
    tools, data, task = tools.resolve(), data.resolve(), task.resolve()
    lock = fetch_assets(tools, splits)
    chain = build_abc(tools)
    protocol_hash = digest(SOURCE / "protocol.py")
    evaluator_hash = digest(SOURCE / "evaluate.py")
    for split in splits:
        entries = []
        destination = data / split
        destination.mkdir(parents=True, exist_ok=True)
        for name in SPLITS[split]:
            circuit = tools / f"{name}.aig"
            target = data / "baselines" / name
            record_path = target / "reference.json"
            identity = dict(
                circuit_sha256=lock[circuit.name]["sha256"],
                abc_sha256=chain["abc_sha256"],
                library_sha256=chain["library_sha256"],
                protocol_sha256=protocol_hash,
                evaluator_sha256=evaluator_hash,
            )
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record["identity"] != identity:
                    raise ValueError(
                        f"Baseline identity changed for {name}; retain existing evidence"
                    )
                result = record["result"]
            else:
                if (target / "result.json").exists():
                    raise ValueError(
                        f"Incomplete attempt at {target}; inspect it before a new attempt"
                    )
                print(f"Baseline {split}/{name}", flush=True)
                result = run_circuit(
                    tools / chain["abc"], tools / chain["library"], circuit, RESYN2, target
                )
                record_path.write_text(
                    json.dumps(dict(identity=identity, result=result), indent=2) + "\n"
                )
            if not result["accepted"]:
                raise RuntimeError(f"Baseline failed for {name}: {result['error']}; logs retained")
            shutil.copyfile(circuit, destination / circuit.name)
            entries.append(
                dict(
                    name=name,
                    sha256=identity["circuit_sha256"],
                    area=result["area"],
                    delay=result["delay"],
                    baseline_identity=identity,
                )
            )
        manifest = dict(split=split, seconds=SECONDS, baseline_commands=RESYN2, circuits=entries)
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if "train" in splits:
        if task.exists():
            raise FileExistsError(f"Task already exists; keep it frozen: {task}")
        task.mkdir(parents=True)
        shutil.copytree(data / "train", task / "train")
        shutil.copyfile(SOURCE / "TASK.md", task / "README.md")
        (task / "recipe.json").write_text(json.dumps(dict(commands=RESYN2), indent=2) + "\n")
    print(json.dumps(dict(task=str(task), prepared_splits=splits)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools", type=Path, default=REPO / ".runs/epfl-synthesis-tools")
    parser.add_argument("--data", type=Path, default=REPO / ".runs/epfl-synthesis-data")
    parser.add_argument("--task", type=Path, default=REPO / ".runs/epfl-synthesis-task")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["calibration", "train"])
    args = parser.parse_args()
    prepare(args.tools, args.data, args.task, args.splits)


if __name__ == "__main__":
    main()
