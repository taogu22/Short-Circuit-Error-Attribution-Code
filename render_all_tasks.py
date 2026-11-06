"""
Render every task listed in name_to_id.csv to PNG images under images/taskXX/:
all demonstration inputs/outputs and the test input (the test output is the
ground truth and is not rendered). Requires the Mini-ARC json files in MiniARC/.

    python render_all_tasks.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Same palette as the official Mini-ARC interface
PALETTE = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]
CMAP = ListedColormap(PALETTE)

JSON_FOLDER = Path("MiniARC")
IMAGES_DIR = Path("images")
MANIFEST_PATH = Path("name_to_id.csv")


def render_grid(grid, save_path):
    """Render an integer matrix as a colored grid image."""
    grid = np.array(grid, dtype=np.uint8)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=9)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(save_path, dpi=150)  # same rendering as used in the experiments (no tight_layout)
    plt.close(fig)  # free the figure


def main():
    IMAGES_DIR.mkdir(exist_ok=True)

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)  
        rows = list(reader)

    for row in rows:
        task_id = row["task_id"]
        original_filename = row["original_filename"]
        task_dir = IMAGES_DIR / task_id
        task_dir.mkdir(exist_ok=True)
        with open(JSON_FOLDER / (original_filename + ".json")) as jf:
            task = json.load(jf)
        for i, pair in enumerate(task["train"]):
            render_grid(pair["input"], task_dir / f"{task_id}_train{i}_input.png")
            render_grid(pair["output"], task_dir / f"{task_id}_train{i}_output.png")
        render_grid(task["test"][0]["input"], task_dir / f"{task_id}_test_input.png")

        print(f"{task_id} <- {original_filename}: rendered, {len(task['train'])} demo pairs")

    print(f"\nDone: {len(rows)} tasks rendered under {IMAGES_DIR}/")


if __name__ == "__main__":
    main()
