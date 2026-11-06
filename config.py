"""Shared experiment size. Override with the N_TASKS environment variable, e.g.
    N_TASKS=10 python regenerate_own_pipeline_v5.py
Tasks are task00 ... task{N_TASKS-1}; name_to_id.csv must contain at least N_TASKS rows.
"""
import os

N_TASKS = int(os.environ.get("N_TASKS", 30))
