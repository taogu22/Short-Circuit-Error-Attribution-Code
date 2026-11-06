import numpy as np
import base64
import json
import csv
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from render_all_tasks import render_grid

"""
We implement the vision-language synergy.
Inputs: grids from MiniARC without rendered as images first.
Stored output: ["task_id", "predicted_grid", "ground_truth_grid", "is_correct", "summarised_rule", "raw_application_response"]
"""

from config import N_TASKS

load_dotenv()

client = OpenAI()

MODEL = "o4-mini"


RULE_SUM_PROMPT = """I will now provide you with several input and output images about 2D grids. You need to summarize the grid-changing rule from it. Output the rule you learned within \\boxed{}."""

RULE_APP_PROMPT = """ I will provide you with several input and output matrices. You need to find the matrix-changing rule from it and apply it to the new input. Put the output matrix within \\boxed{{}}. Here is a possible rule for your reference. {suggested_rule}
Note that the rule is described in color and each color represents a value in the matrix: [0:black; 1:blue; 2:red; 3:green; 4:yellow; 5:grey; 6:pink; 7:orange; 8:light blue; 9:brown]. You need to first check the correctness of the rule based on the examples. If the rule is correct, apply it to the new input. Otherwise, summarize a new rule and apply it to the new input.
The matrix inside \\boxed{{}} must be written as a JSON-style nested list of integers, e.g. \\boxed{{[[0, 1, 2], [3, 4, 5]]}}. Do not use LaTeX matrix notation such as bmatrix or pmatrix.
"""

def encode_image(image_path: Path) -> str:
    """turn an image into base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def clean_json_response(text):
    """
    clean up the response
    """
    text = text.strip()
    text = text.removeprefix("```json").removeprefix("```")
    text = text.removesuffix("```")
    return text.strip()

def main():

    with open("name_to_id.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # dict mapping task id to task name
    task_id_to_name = {}
    for row in rows:
        task_id, task_name = row['task_id'], row['original_filename']
        task_id_to_name[task_id] = task_name

    RESULT_PATH = Path("results") / "VLS_o4_v2.csv"
    with open(RESULT_PATH, 'w', newline="", encoding="utf-8") as rf:
        writer = csv.writer(rf)
        writer.writerow(["task_id", "predicted_grid", "ground_truth_grid", "is_correct", "summarised_rule", "raw_application_response"])

        for i in range(N_TASKS):
        # for i in range(1): # test with one task
            task_id = f"task{i:02d}" # task0i
            print("==================")
            print(f"Working on {task_id}...")
            task_name = task_id_to_name[task_id]
            task_path = Path("MiniARC") / (task_name + ".json")
            with open(task_path) as f:
                task = json.load(f)
            train_size = len(task['train']) # number of samples in the task

            """
            stage VISION: collect the content for rule summarization
            """
            content_sum = [{"type": "text", "text": RULE_SUM_PROMPT}]
            content_app = [{"type": "text", "text": RULE_APP_PROMPT}]
            
            # add all training data to content
            for train_id in range(train_size):
                print(f"working on {train_id=}")

                # add the train input
                content_sum.append({"type": "text", "text": f"Example input {train_id}:"})
                train_input_path = Path("images") / task_id / f"{task_id}_train{train_id}_input.png"
                content_sum.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(train_input_path)}"}})

                content_app.append({"type": "text", "text": f"Example input {train_id}:"})
                train_input_grid = str(task['train'][train_id]['input'])
                content_app.append({"type": "text", "text": train_input_grid})

                # add the train output
                content_sum.append({"type": "text", "text": f"Example output {train_id}:"})
                train_output_path = Path("images") / task_id / f"{task_id}_train{train_id}_output.png"
                content_sum.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(train_output_path)}"}})

                content_app.append({"type": "text", "text": f"Example output {train_id}:"})
                train_output_grid = str(task['train'][train_id]['output'])
                content_app.append({"type": "text", "text": train_output_grid})

            # add the test input to content
            content_app.append({"type": "text", "text": "New input:"})
            test_input_grid = str(task['test'][0]['input'])
            content_app.append({"type": "text", "text": test_input_grid})
            
            response_rule = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": content_sum,
                    }
                ],
                reasoning_effort = 'medium',  
            )

            answer_text = response_rule.choices[0].message.content
            # print(f"{answer_text}")
            sum_rule = answer_text
            print("Rule summarised.")
            

            """
            stage TEXT: apply the rule
            """
            content_app[0]["text"] = content_app[0]["text"].format(suggested_rule=sum_rule)

            response_app = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": content_app,
                    }
                ],
                reasoning_effort = 'medium'
            )

            predicted_grid_raw = response_app.choices[0].message.content
            # turn raw data into grid
            print(repr(predicted_grid_raw))
            match = re.search(r"\\boxed\{(.*)\}", predicted_grid_raw, re.DOTALL)
            if match is None:
                print(f"{task_id}: failed to parse boxed content")
                print(predicted_grid_raw)
                writer.writerow([task_id, None, None, False, sum_rule, predicted_grid_raw])
                continue
            try:
                predicted_grid = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                print(f"{task_id}: boxed content is not valid JSON")
                print(repr(match.group(1)))
                writer.writerow([task_id, None, None, False, sum_rule, predicted_grid_raw])
                continue
            ground_truth_grid = task['test'][0]['output']
            is_correct = (predicted_grid == ground_truth_grid)

            # ["task_id", "predicted_grid", "ground_truth_grid", "is_correct", "summarised_rule"]

            writer.writerow([task_id, predicted_grid, ground_truth_grid, is_correct, sum_rule, predicted_grid_raw])
            
            print("Rule applied.")

            # Render the predicted grid
            image_path = Path("images") / task_id / f"{task_id}_test_output_VLS_o4_v2.png"
            image_grid = np.array(predicted_grid, dtype=np.uint8)
            render_grid(image_grid, image_path)

    print("Experiment completed.")

