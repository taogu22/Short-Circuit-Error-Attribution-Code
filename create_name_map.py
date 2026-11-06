import csv
from pathlib import Path
import random

"""
Make a map task_id -> task_name, and store in name_to_id.csv.
This samples 50 of the 149 Mini-ARC tasks with seed 22; the experiments in the paper
use the first 30 (task00-task29), which is what the shipped name_to_id.csv contains.
(The order of Path.glob depends on the file system, so we ship the resulting file
rather than relying on re-running this script.)
"""

random.seed(22)

json_folder = Path('MiniARC')
all_task_names = [f.stem for f in json_folder.glob("*.json")]
# f.name keeps .json
print(all_task_names)

sampled_tasks = random.sample(all_task_names, 50)
print(f"{sampled_tasks=}")

with open("name_to_id.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["task_id", "original_filename"])

    for i, file_name in enumerate(sampled_tasks):
        task_id = f"task{i:02d}"
        writer.writerow([task_id,file_name])
