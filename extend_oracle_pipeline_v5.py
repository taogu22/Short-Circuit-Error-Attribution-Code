import csv
import json
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# o4-mini reasoning (both stages), matching regenerate_own_pipeline_v5.py --
# this is the piece that actually needs to change now that v5 is the
# baseline own-pipeline: short-circuit's M_i/M is computed on a
# perception+reasoning-o4-mini pipeline, so oracle substitution's e_i must
# be computed on the SAME reasoning model, or the two aren't measuring the
# same pipeline.
from regenerate_own_pipeline_v5 import stage_reasoning_demo

"""
Oracle substitution's reasoningstage_d arm, redone on o4-mini reasoning to
match v5 (the now-current own-pipeline baseline).

Background: extend_oracle_pipeline.py (and the original 10-task oracle
pilot, results/two_stage_results_oracle_imagetext.csv) both ran the
oracle-substituted rule-induction call on gpt-4o, because the own-pipeline
they were meant to pair with (v3/v4) also used gpt-4o for reasoning. Now
that the own-pipeline baseline is v5 (o4-mini for perception AND
reasoning), those gpt-4o-based oracle numbers are no longer paired with
the right pipeline -- e_reasoningstage_d has to be measured on the same
model that produces v5's M_i/M.

Own oracle-description file (taskXX_oracle_v5.csv), separate from
taskXX_oracle_descriptions.csv: the descriptions here are meant to be
hand-reviewed/edited (per the "strong model proposes, human revises"
methodology in the paper), so this arm needs its own file that won't be
silently overwritten by, or confused with, the original gpt-4o-era oracle
pilot's shared oracle_descriptions.csv.

Workflow:
  1. Run this script. For any task without a taskXX_oracle_v5.csv yet, it
     generates one (o4-mini proposes a correction, given the model's own
     v5 description as a style reference and the true grid as ground
     truth), then immediately runs o4-mini rule-induction on it and
     appends the result to TAE_results_oracle_v5.csv.
  2. Open taskXX_oracle_v5.csv, review/edit the proposed descriptions by
     hand against the true grid.
  3. Delete that task's row from TAE_results_oracle_v5.csv (so it isn't
     skipped as "already done") and re-run this script -- since
     taskXX_oracle_v5.csv now exists, it will be read from disk (your
     edited version) instead of regenerated, and rule-induction will be
     re-run on your edited descriptions.

Writes to results/TAE_results_oracle_v5.csv. Does NOT touch
TAE_results_oracle_imagetext.csv, TAE_results_oracle_extended.csv, or any
gpt-4o-era oracle file.
"""

from config import N_TASKS

load_dotenv()
client = OpenAI()

ORACLE_MODEL = "o4-mini"  # matches generate_oracle_descriptions.py's correction model
TASK_RANGE = range(N_TASKS)
INCLUDE_RAW_IMAGES = True
OUTPUT_PATH = Path("results") / "TAE_results_oracle_v5.csv"

# Verbatim from generate_oracle_descriptions.py / extend_oracle_pipeline.py.
CORRECTION_PROMPT = """You are writing an accurate natural-language description of
an ARC puzzle grid. For reference, another (vision) model previously wrote a
description of this grid while looking at a rendered image of it -- but its
description may contain inaccuracies: wrong shapes, wrong positions, wrong color
values, missed or invented objects, and so on. Do not treat that description as
authoritative; it is only a rough starting point for style and level of detail.

You are given the EXACT ground-truth grid (as a matrix of integers, row by row,
0-indexed from the top-left) instead of an image. Your job is to write a description
of this grid that is fully accurate against the true grid, covering:
- the shape and location of the objects in the grid
- any additional patterns in the grid
- the dominant colors in the grid
using this color-to-number mapping (refer to colors by these integers):
0 = black
1 = blue
2 = red
3 = green
4 = yellow
5 = grey
6 = fuchsia
7 = orange
8 = teal
9 = brown/maroon

Prioritize accuracy over preserving the original description's wording -- feel free
to rewrite it completely if that's what it takes to correctly describe the true grid.
Do not add commentary about what was corrected, do not mention the transformation
rule or other grids in the task, and do not add information that isn't about this
single grid. Respond with ONLY the description text (no JSON, no markdown headers
required, plain natural language, similar length/style to the reference description).
"""


def clean_response(text):
    text = text.strip()
    text = text.removeprefix("```").removesuffix("```")
    return text.strip()


def correct_description(original_description, true_grid):
    grid_str = "\n".join(str(row) for row in true_grid)
    content = [
        {"type": "text", "text": CORRECTION_PROMPT},
        {"type": "text", "text": f"Reference description (may be inaccurate):\n{original_description}"},
        {"type": "text", "text": f"True grid (ground truth):\n{grid_str}"},
    ]
    response = client.chat.completions.create(
        model=ORACLE_MODEL,
        messages=[{"role": "user", "content": content}],
        reasoning_effort="medium",
    )
    return clean_response(response.choices[0].message.content)


def generate_oracle_v5_descriptions(task_id, task):
    """Style/length reference is v5's own description (o4-mini's own
    perception), not v3's -- keeps the reference consistent with which
    own-pipeline this oracle arm is paired with."""
    own_desc_path = Path("images") / task_id / f"{task_id}_descriptions_v5.csv"
    with open(own_desc_path, encoding="utf-8") as f:
        own_rows = list(csv.DictReader(f))

    true_grids = {}
    for i, pair in enumerate(task["train"]):
        true_grids[(str(i), "input")] = pair["input"]
        true_grids[(str(i), "output")] = pair["output"]
    true_grids[("test", "input")] = task["test"][0]["input"]

    out_path = Path("images") / task_id / f"{task_id}_oracle_v5.csv"
    oracle = {}
    with open(out_path, "w", newline="", encoding="utf-8") as of:
        writer = csv.writer(of)
        writer.writerow(["train_id", "input/output", "description"])
        for row in own_rows:
            key = (row["train_id"], row["input/output"])
            if key not in true_grids:
                continue
            corrected = correct_description(row["description"], true_grids[key])
            writer.writerow([key[0], key[1], corrected])
            oracle[key] = corrected
            print(f"    oracle description {key}: done")
    return oracle


def load_existing_oracle_v5_descriptions(task_id):
    path = Path("images") / task_id / f"{task_id}_oracle_v5.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["train_id"], r["input/output"]): r["description"] for r in rows}


def already_done_task_ids() -> set:
    if not OUTPUT_PATH.exists():
        return set()
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["task_id"] for r in rows if r.get("task_id")}


def main():
    with open("name_to_id.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    task_id_to_name = {row["task_id"]: row["original_filename"] for row in rows}

    skip_ids = already_done_task_ids()
    if skip_ids:
        print(f"Resuming: {len(skip_ids)} task(s) already in {OUTPUT_PATH}, skipping those.")

    file_exists = OUTPUT_PATH.exists()
    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        if not file_exists:
            writer.writerow(["task_id", "rule", "raw_response", "error"])
            writer.writerow([])

        for i in TASK_RANGE:
            task_id = f"task{i:02d}"
            if task_id in skip_ids:
                continue
            print("==================")
            print(f"Working on {task_id}...")

            oracle_path = Path("images") / task_id / f"{task_id}_oracle_v5.csv"
            try:
                task_name = task_id_to_name[task_id]
                with open(Path("MiniARC") / (task_name + ".json")) as tf:
                    task = json.load(tf)

                if oracle_path.exists():
                    print(f"  {oracle_path} already exists -- reusing it (your edits included), not regenerating.")
                    per_d = {k: v for k, v in load_existing_oracle_v5_descriptions(task_id).items() if k[0] != "test"}
                else:
                    oracle = generate_oracle_v5_descriptions(task_id, task)
                    per_d = {k: v for k, v in oracle.items() if k[0] != "test"}

                rule, raw = stage_reasoning_demo(task_id, task, per_d, INCLUDE_RAW_IMAGES)
                writer.writerow([task_id, rule, raw, ""])
                writer.writerow([])
                out_f.flush()
                print(f"  oracle rule (o4-mini): {rule}")

            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                print(f"{task_id}: FAILED - {err_msg}")
                traceback.print_exc()
                writer.writerow([task_id, "", "", err_msg])
                writer.writerow([])
                out_f.flush()

    print("Oracle substitution (o4-mini reasoning, task01-30) completed. Results in", OUTPUT_PATH)


if __name__ == "__main__":
    main()
