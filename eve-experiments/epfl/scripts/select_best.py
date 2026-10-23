"""Select the best formal training recipe, including seed; earlier entries win ties."""

import argparse
import json
import sqlite3
from pathlib import Path

import yaml


def select_best(run_root):
    # EvE updates created_at when refreshing an entry; rowid preserves insertion order.
    with sqlite3.connect((run_root / "solver_lineage.db").as_uri() + "?mode=ro", uri=True) as db:
        entries = db.execute(
            "SELECT entry_id, files_ref_json, score_ref_json FROM eve_population_entries "
            "ORDER BY rowid"
        ).fetchall()
    best = None
    for entry_id, files_json, score_json in entries:
        score_ref = json.loads(score_json)
        score = yaml.safe_load((run_root / "artifacts" / score_ref["location"]).read_text())[
            "score"
        ]
        if best is None or score > best["score"]:
            files_ref = json.loads(files_json)
            files = json.loads((run_root / "artifacts" / files_ref["location"]).read_text())
            best = dict(entry_id=entry_id, score=score, recipe=files["recipe.json"])
    if best is None:
        raise ValueError("The run has no formally evaluated candidates")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    best = select_best(root)
    output = root / "best_recipe.json"
    output.write_text(best.pop("recipe"))
    print(json.dumps(dict(**best, recipe=str(output))))


if __name__ == "__main__":
    main()
