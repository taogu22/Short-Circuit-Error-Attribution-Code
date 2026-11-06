import base64
import csv
import json
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from render_all_tasks import render_grid

"""
Full own-pipeline run on o4-mini: perception AND reasoning both switched
from gpt-4o to o4-mini (v4 only swapped the perception prompt and kept
gpt-4o everywhere; this version swaps the model for all four stages).

This is a different kind of run from v3/v4/oracle: those all held the
reasoning-stage model fixed at gpt-4o (matching Wang et al.'s own choice
for Mini-ARC, Table 1) so that any measured difference was attributable
only to perception's input, not to which model is doing the reasoning.
Swapping the whole own-pipeline to o4-mini means v5's task-level
correctness and M_i/M numbers are no longer directly comparable to
Wang et al.'s case study on the same footing -- treat this as a
supplementary check on whether a different model's own-perception is any
more reliable, not as a replacement for the gpt-4o results used
elsewhere in the paper.

Both stages use o4-mini's reasoning_effort="medium" instead of
temperature=0 (o-series models reject/ignore temperature), matching the
convention already used for o4-mini calls elsewhere in this codebase
(judge_stages.py, generate_oracle_descriptions.py).

Writes to images/taskXX/taskXX_descriptions_v5.csv and
results/TAE_results_v5.csv, same as before. Re-point judge_stages.py at
DESCRIPTIONS_SUFFIX = "descriptions_v5" and RESULTS_PATH =
TAE_results_v5.csv (already the active config) to judge this arm.
"""

from config import N_TASKS

load_dotenv()
client = OpenAI()

MODEL = "o4-mini"  # now used for perception AND reasoning
TASK_RANGE = range(N_TASKS)
INCLUDE_RAW_IMAGES = True

RESULT_PATH = Path("results") / "TAE_results_v5.csv"

PERCEPTION_PROMPT_V5 = """You are a precise visual perception system for ARC puzzles. Your task is to provide an accurate, structured description of a single grid to help downstream reasoning models discover transformation rules.

Look at the image carefully before writing anything. Do not describe what a "typical" ARC grid of this kind might look like -- check the actual image and report only what is really there. In particular:
- Determine the grid's exact dimensions (number of rows and columns) from the image itself.
- For each object, check its exact shape, size, and position (e.g. which rows/columns it occupies) against the image rather than estimating.
- Check each color against the image rather than assuming a color from context; note that 0 corresponds to the background color (and does not itself form objects).

Then write the following in natural language:
- The shape and location of the objects in the grid. Some objects can be large or overlapping with each other.
- Any additional patterns in the grid.
- The dominant colors in the grid.

The grids use the following color-to-number mapping. Do not use color names, only refer to colors by their integer value:
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

Respond with ONLY the description text (no JSON, no markdown headers), in natural language, similar length/style to a careful, accurate grid description."""

# Same wording as RULE_INDUCTION_PROMPT / RULE_APPLICATION_PROMPT_TEMPLATE
# in two_stage_experiment_v3_split.py -- copied here (not imported) so this
# file can call them with o4-mini's parameters instead of gpt-4o's, without
# touching the shared module that v3/v4/oracle all still depend on.

RULE_INDUCTION_PROMPT = """Below are descriptions of several demonstration examples (input and output pairs). Infer the common transformation rule from the demonstrations.

The grids use the following color-to-number mapping. Do not use color names, only refer to colors by their integer value:
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

Your answer should consist of one part, organised using the format {"rule": "..."}:
- The transformation rule you discovered based on the example inputs and outputs, in a concise way.

Respond with ONLY the JSON object described above. Do not include markdown formatting or any other extra text."""


RULE_APPLICATION_PROMPT_TEMPLATE = """Below are descriptions of several demonstration examples (input and output pairs), followed by a description of a test input. You have already inferred the following transformation rule from the demonstrations:

Rule: {rule}

Apply this rule to the test input.

All grids (both input and output) are exactly 5x5.

The grids use the following color-to-number mapping. When you write the output grid, every cell must be one of these integers (not a color name):
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

Your answer should consist of two parts, organised using the format {{"application": "...", "grid": [[...]]}}:
- First, the way you apply the transformation rule to the test input grid.
- Second, your answer of the output grid given the test input grid and the transformation rule, as a JSON 2D array of integers.

Respond with ONLY the JSON object described above. Do not include markdown formatting or any other extra text."""


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_block(image_path: Path) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"},
    }


def clean_json_response(text):
    text = text.strip()
    text = text.removeprefix("```json").removeprefix("```")
    text = text.removesuffix("```")
    return text.strip()


def call_model(content, max_retries=2):
    """Same empty-response retry/diagnostic behavior as
    two_stage_experiment_v3_split.py's call_model, but with
    reasoning_effort instead of temperature=0 (o4-mini rejects/ignores
    temperature)."""
    last_diag = None
    for attempt in range(max_retries + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            reasoning_effort="medium",
        )
        choice = response.choices[0]
        text = choice.message.content
        if text is not None and text.strip():
            return text
        refusal = getattr(choice.message, "refusal", None)
        last_diag = f"finish_reason={choice.finish_reason!r}, refusal={refusal!r}"
        print(f"    empty response (attempt {attempt + 1}/{max_retries + 1}): {last_diag}")
        if attempt < max_retries:
            time.sleep(1.5)
    raise RuntimeError(f"Model returned empty content after {max_retries + 1} attempts ({last_diag})")


def describe(image_path: Path) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PERCEPTION_PROMPT_V5},
                    image_block(image_path),
                ],
            }
        ],
        reasoning_effort="medium",
    )
    text = response.choices[0].message.content
    return text.strip() if text else ""


def generate_v5_descriptions(task_id: str, task: dict) -> dict:
    """Generates v5 descriptions for every demo pair + the test input, from
    the rendered images only (no ground truth). Writes to
    images/taskXX/taskXX_descriptions_v5.csv (same schema as
    descriptions_v3.csv/descriptions_v4.csv) and returns the same dict
    shape used everywhere else in this codebase: {(train_id_str,
    "input"/"output"): text} plus ("test", "input") -> text."""
    train_size = len(task["train"])
    out_path = Path("images") / task_id / f"{task_id}_descriptions_v5.csv"
    descriptions = {}
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["train_id", "input/output", "description"])

        for train_id in range(train_size):
            tid = str(train_id)
            input_desc = describe(Path("images") / task_id / f"{task_id}_train{train_id}_input.png")
            writer.writerow([train_id, "input", input_desc])
            descriptions[(tid, "input")] = input_desc

            output_desc = describe(Path("images") / task_id / f"{task_id}_train{train_id}_output.png")
            writer.writerow([train_id, "output", output_desc])
            descriptions[(tid, "output")] = output_desc

        test_desc = describe(Path("images") / task_id / f"{task_id}_test_input.png")
        writer.writerow(["test", "input", test_desc])
        descriptions[("test", "input")] = test_desc

    return descriptions


def load_existing_v5_descriptions(task_id: str) -> dict:
    path = Path("images") / task_id / f"{task_id}_descriptions_v5.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["train_id"], r["input/output"]): r["description"] for r in rows}


def stage_reasoning_demo(task_id: str, task: dict, per_d: dict, include_raw_images: bool):
    """o4-mini version of two_stage_experiment_v3_split.py's function of
    the same name -- same prompt, same call shape, different model/params."""
    train_size = len(task["train"])
    content = [{"type": "text", "text": RULE_INDUCTION_PROMPT}]

    for train_id in range(train_size):
        tid = str(train_id)
        content.append({"type": "text", "text": f"Demo {train_id} input image description: {per_d[(tid, 'input')]}"})
        if include_raw_images:
            content.append(image_block(Path("images") / task_id / f"{task_id}_train{train_id}_input.png"))
        content.append({"type": "text", "text": f"Demo {train_id} output image description: {per_d[(tid, 'output')]}"})
        if include_raw_images:
            content.append(image_block(Path("images") / task_id / f"{task_id}_train{train_id}_output.png"))

    raw_text = call_model(content)
    parsed = json.loads(clean_json_response(raw_text))
    rule = parsed["rule"]
    return rule, raw_text


def stage_reasoning_test(task_id: str, task: dict, per_d: dict, rea_d_rule: str, per_t: str, include_raw_images: bool):
    """o4-mini version of two_stage_experiment_v3_split.py's function of
    the same name. Re-shows demo descriptions/images for the same reason
    as the original: without them, a rule like "swap the L-shaped pattern"
    has no referent for whoever applies it."""
    train_size = len(task["train"])
    content = [{"type": "text", "text": RULE_APPLICATION_PROMPT_TEMPLATE.format(rule=rea_d_rule)}]

    for train_id in range(train_size):
        tid = str(train_id)
        content.append({"type": "text", "text": f"Demo {train_id} input image description: {per_d[(tid, 'input')]}"})
        if include_raw_images:
            content.append(image_block(Path("images") / task_id / f"{task_id}_train{train_id}_input.png"))
        content.append({"type": "text", "text": f"Demo {train_id} output image description: {per_d[(tid, 'output')]}"})
        if include_raw_images:
            content.append(image_block(Path("images") / task_id / f"{task_id}_train{train_id}_output.png"))

    content.append({"type": "text", "text": f"Test input description: {per_t}"})
    if include_raw_images:
        content.append(image_block(Path("images") / task_id / f"{task_id}_test_input.png"))

    raw_text = call_model(content)
    try:
        parsed = json.loads(clean_json_response(raw_text))
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Could not parse rule-application response as JSON ({e}). "
            f"Raw response was: {raw_text!r}"
        ) from e
    application = parsed["application"]
    predicted_grid = parsed["grid"]
    return application, predicted_grid, raw_text


def already_done_task_ids() -> set:
    if not RESULT_PATH.exists():
        return set()
    with open(RESULT_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["task_id"] for r in rows if r.get("task_id")}


def main():
    with open("name_to_id.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    task_id_to_name = {row["task_id"]: row["original_filename"] for row in rows}

    skip_ids = already_done_task_ids()
    if skip_ids:
        print(f"Resuming: {len(skip_ids)} task(s) already in {RESULT_PATH}, skipping those.")

    file_exists = RESULT_PATH.exists()
    with open(RESULT_PATH, "a", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        if not file_exists:
            writer.writerow(["task_id", "predicted_grid", "ground_truth_grid", "is_correct", "rule", "application", "raw_response", "error"])
            writer.writerow([])

        for i in TASK_RANGE:
            task_id = f"task{i:02d}"
            if task_id in skip_ids:
                continue
            print("==================")
            print(f"Working on {task_id}...")
            try:
                task_name = task_id_to_name[task_id]
                with open(Path("MiniARC") / (task_name + ".json")) as tf:
                    task = json.load(tf)

                v5_path = Path("images") / task_id / f"{task_id}_descriptions_v5.csv"
                if v5_path.exists():
                    print(f"  {v5_path} already exists -- reusing it, not regenerating.")
                    own_descriptions = load_existing_v5_descriptions(task_id)
                else:
                    own_descriptions = generate_v5_descriptions(task_id, task)

                per_d = {k: v for k, v in own_descriptions.items() if k[0] != "test"}
                per_t = own_descriptions[("test", "input")]

                rea_d_rule, rea_d_raw = stage_reasoning_demo(task_id, task, per_d, INCLUDE_RAW_IMAGES)
                print(f"  rea_d rule: {rea_d_rule}")
                rea_t_application, predicted_grid, rea_t_raw = stage_reasoning_test(
                    task_id, task, per_d, rea_d_rule, per_t, INCLUDE_RAW_IMAGES
                )

                ground_truth = task["test"][0]["output"]
                is_correct = predicted_grid == ground_truth

                combined_raw = f"RULE_CALL: {rea_d_raw}\n---\nAPPLICATION_CALL: {rea_t_raw}"
                writer.writerow([
                    task_id, predicted_grid, ground_truth, is_correct,
                    rea_d_rule, rea_t_application, combined_raw, "",
                ])
                writer.writerow([])
                out_f.flush()

                try:
                    image_path = Path("images") / task_id / f"{task_id}_test_output_v5.png"
                    render_grid(predicted_grid, image_path)
                except Exception as render_err:
                    print(f"{task_id}: predicted grid was invalid, skipped rendering ({render_err})")

                print(f"{task_id}: done, is_correct={is_correct}")

            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                print(f"{task_id}: FAILED - {err_msg}")
                traceback.print_exc()
                writer.writerow([task_id, None, None, False, "", "", "", err_msg])
                writer.writerow([])
                out_f.flush()

    print("v5 own-pipeline run completed. Results in", RESULT_PATH)


if __name__ == "__main__":
    main()
