import csv
import json
import traceback
from pathlib import Path

from render_all_tasks import render_grid
from config import N_TASKS
from regenerate_own_pipeline_v5 import stage_reasoning_test

"""
The e_reasoningstage_t experiment: swap in BOTH the oracle-corrected rule
(from taskXX_oracle_v5.csv's descriptions -> run_oracle_rule_v5.py's rule
induction) and the oracle test description (perceptionstage_t's would-be
output, also from taskXX_oracle_v5.csv) and run only the rule-application
step (reasoningstage_t) on o4-mini. Unlike per_d/per_t/rea_d, there's no
LLM-judge here -- reasoningstage_t's output is a grid, so its correctness
is checked directly against ground truth, exactly per the Experimental
Setup section: "check the resulting grid directly against ground truth to
get e_reasoningstage_t."

Inputs (both must already exist -- this script does no generation):
  - images/taskXX/taskXX_oracle_v5.csv: oracle demo descriptions (per_d)
    AND the oracle test description (per_t), all hand-reviewed/edited.
  - results/TAE_oracle_rules.csv: the "summarised_rule" column, keyed by
    task_id -- the hand-corrected rules (originally from VLS_o4_v2.csv,
    then manually fixed for the tasks that were wrong) used as the oracle
    reasoningstage_d output here.

If either is missing for a task, that task is skipped with a message.

1 - (fraction correct) over these tasks is e_reasoningstage_t.

Writes results/TAE_results_oracle_rea_t_v5.csv (task_id, predicted_grid,
ground_truth, is_correct, application, raw_response, error). Rewrites the
whole file each run (like run_oracle_rule_v5.py) so re-running a task
after further edits just overwrites its row -- no manual deletion needed.
"""

TASK_RANGE = range(N_TASKS)
INCLUDE_RAW_IMAGES = True
OUTPUT_PATH = Path("results") / "TAE_results_oracle_rea_t_v5.csv"
FIELDNAMES = ["task_id", "predicted_grid", "ground_truth", "is_correct", "application", "raw_response", "error"]


def load_oracle_v5_descriptions(task_id):
    path = Path("images") / task_id / f"{task_id}_oracle_v5.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["train_id"], r["input/output"]): r["description"] for r in rows}


def load_oracle_rules(path=Path("results") / "TAE_oracle_rules.csv") -> dict:
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("task_id")]
    return {r["task_id"]: r["summarised_rule"] for r in rows if r.get("summarised_rule")}


def load_existing_results() -> dict:
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

    oracle_rules = load_oracle_rules()
    results = load_existing_results()

    n_correct = 0
    n_total = 0

    for i in TASK_RANGE:
        task_id = f"task{i:02d}"
        oracle_path = Path("images") / task_id / f"{task_id}_oracle_v5.csv"

        if task_id not in oracle_rules:
            print(f"{task_id}: skipped, no oracle rule in TAE_oracle_rules.csv yet")
            continue
        if not oracle_path.exists():
            print(f"{task_id}: skipped, {oracle_path} doesn't exist yet")
            continue

        print("==================")
        print(f"Working on {task_id}...")
        try:
            task_name = task_id_to_name[task_id]
            with open(Path("MiniARC") / (task_name + ".json")) as tf:
                task = json.load(tf)

            oracle_desc = load_oracle_v5_descriptions(task_id)
            per_d = {k: v for k, v in oracle_desc.items() if k[0] != "test"}
            per_t = oracle_desc[("test", "input")]
            rea_d_rule = oracle_rules[task_id]

            application, predicted_grid, raw = stage_reasoning_test(
                task_id, task, per_d, rea_d_rule, per_t, INCLUDE_RAW_IMAGES
            )
            ground_truth = task["test"][0]["output"]
            is_correct = predicted_grid == ground_truth

            results[task_id] = {
                "task_id": task_id,
                "predicted_grid": predicted_grid,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "application": application,
                "raw_response": raw,
                "error": "",
            }
            write_results(results)

            try:
                image_path = Path("images") / task_id / f"{task_id}_test_output_oracle_v5.png"
                render_grid(predicted_grid, image_path)
            except Exception as render_err:
                print(f"{task_id}: predicted grid was invalid, skipped rendering ({render_err})")

            n_total += 1
            n_correct += int(is_correct)
            print(f"  is_correct={is_correct}")

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"{task_id}: FAILED - {err_msg}")
            traceback.print_exc()
            results[task_id] = {
                "task_id": task_id, "predicted_grid": "", "ground_truth": "",
                "is_correct": False, "application": "", "raw_response": "", "error": err_msg,
            }
            write_results(results)

    if n_total:
        print(f"\n{n_correct}/{n_total} correct -> e_reasoningstage_t (WRONG rate) = {1 - n_correct / n_total:.1%}")
    print("Done. Results in", OUTPUT_PATH)


if __name__ == "__main__":
    main()
