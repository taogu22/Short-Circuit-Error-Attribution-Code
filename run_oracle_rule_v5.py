import csv
import json
import traceback
from pathlib import Path

from config import N_TASKS
from regenerate_own_pipeline_v5 import stage_reasoning_demo

"""
Step 2 of the oracle-substitution (o4-mini) workflow, split out of
extend_oracle_pipeline_v5.py so editing descriptions and running rule
induction on them are two separate, independently-runnable steps.

This script does NOT generate or correct any descriptions. It REQUIRES
images/taskXX/taskXX_oracle_v5.csv to already exist (created by
extend_oracle_pipeline_v5.py, then hand-edited by you against the true
grid) -- if a task's file is missing, it's skipped with a message instead
of being auto-generated.

Unlike extend_oracle_pipeline_v5.py, re-running a task here OVERWRITES its
row in results/TAE_results_oracle_v5.csv automatically -- no need to
manually delete the old row first. This is meant for a tight edit -> rerun
-> check -> edit-again loop:

  1. Edit images/taskXX/taskXX_oracle_v5.csv by hand.
  2. Narrow TASK_RANGE below to just that task (or leave it covering
     everything you've edited) and run this script.
  3. Check the printed rule / results/TAE_results_oracle_v5.csv.
  4. Repeat.

Writes results/TAE_results_oracle_v5.csv (same file extend_oracle_pipeline_v5.py
writes, same columns: task_id, rule, raw_response, error). Rewrites the
whole file each time so overwriting is simple; task order is sorted by
task_id.
"""

TASK_RANGE = range(N_TASKS)
INCLUDE_RAW_IMAGES = True
OUTPUT_PATH = Path("results") / "TAE_results_oracle_v5.csv"
FIELDNAMES = ["task_id", "rule", "raw_response", "error"]


def load_oracle_v5_descriptions(task_id):
    path = Path("images") / task_id / f"{task_id}_oracle_v5.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["train_id"], r["input/output"]): r["description"] for r in rows}


def load_existing_results() -> dict:
    """task_id -> row dict, preserving whatever's already in OUTPUT_PATH
    (rows for tasks not touched this run are kept as-is)."""
    if not OUTPUT_PATH.exists():
        return {}
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["task_id"]: r for r in rows if r.get("task_id")}


def write_results(results: dict):
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for task_id in sorted(results):
            writer.writerow(results[task_id])


def main():
    with open("name_to_id.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    task_id_to_name = {row["task_id"]: row["original_filename"] for row in rows}

    results = load_existing_results()

    for i in TASK_RANGE:
        task_id = f"task{i:02d}"
        oracle_path = Path("images") / task_id / f"{task_id}_oracle_v5.csv"
        if not oracle_path.exists():
            print(f"{task_id}: skipped, {oracle_path} doesn't exist yet (run extend_oracle_pipeline_v5.py first)")
            continue

        print("==================")
        print(f"Working on {task_id}...")
        try:
            task_name = task_id_to_name[task_id]
            with open(Path("MiniARC") / (task_name + ".json")) as tf:
                task = json.load(tf)

            per_d = {k: v for k, v in load_oracle_v5_descriptions(task_id).items() if k[0] != "test"}
            rule, raw = stage_reasoning_demo(task_id, task, per_d, INCLUDE_RAW_IMAGES)
            results[task_id] = {"task_id": task_id, "rule": rule, "raw_response": raw, "error": ""}
            write_results(results)
            print(f"  oracle rule (o4-mini): {rule}")

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"{task_id}: FAILED - {err_msg}")
            traceback.print_exc()
            results[task_id] = {"task_id": task_id, "rule": "", "raw_response": "", "error": err_msg}
            write_results(results)

    print("Done. Results in", OUTPUT_PATH)


if __name__ == "__main__":
    main()
