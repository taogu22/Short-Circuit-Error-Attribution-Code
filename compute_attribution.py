import csv
from pathlib import Path

"""
Computes short-circuit shares M_i/M from results/judgments_plain.csv
(produced by judge_stages.py against results/TAE_results_plain.csv).

Walks each failing task's judged stages in the paper's canonical order --
perceptionstage_d, reasoningstage_d, perceptionstage_t, reasoningstage_t --
and attributes the failure to the FIRST stage judged WRONG, matching Wang
et al.'s own earliest-failing-step protocol (Sec 3.4) and this paper's
short-circuit definition (Sec 3.2). If per_d, rea_d, and per_t are all
judged CORRECT but the task still failed overall, the failure is
attributed to reasoningstage_t by elimination (we don't judge rea_t
directly here; it's whatever's left once the first three are ruled out).

Also reports e_perceptionstage_d and e_perceptionstage_t: since
perception's input (the raw image) is always correct by construction, no
oracle substitution is needed for these two -- the plain pipeline's own
per_d/per_t WRONG rate over the same N tasks already IS the intrinsic
error rate for those two stages. (e_reasoningstage_d needs the oracle-pilot
run judged separately -- see judge_stages.py's oracle mode; not computed
here.)
"""

JUDGMENTS_PATH = Path("results") / "judgments_v5.csv"
# v4 own-pipeline judgments (gpt-4o, accuracy-first prompt): Path("results") / "judgments_v4.csv"
# Original v3 own-pipeline judgments: Path("results") / "judgments_plain.csv"

STAGE_ORDER = ["per_d", "rea_d", "per_t", "rea_t"]
STAGE_LABELS = {
    "per_d": "Perception (Demo)",
    "rea_d": "Reasoning (Inductive)",
    "per_t": "Perception (Test)",
    "rea_t": "Reasoning (Deductive)",
}


def load_judgments():
    with open(JUDGMENTS_PATH, encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("task_id")]


def main():
    rows = load_judgments()

    bad_rows = [r for r in rows if r["per_d_verdict"] not in ("CORRECT", "WRONG")]
    if bad_rows:
        print(f"WARNING: {len(bad_rows)} row(s) look like judging errors, not CORRECT/WRONG "
              f"verdicts (e.g. task_id={bad_rows[0]['task_id']!r}, "
              f"per_d_verdict={bad_rows[0]['per_d_verdict']!r}). Excluding them below.")
    rows = [r for r in rows if r not in bad_rows]

    # Short-circuit attribution, restricted to tasks that actually failed
    # end to end (should be all of them for the task01-30 pool, but don't
    # assume it -- a task judged as failing every stage but somehow correct
    # overall would be a real anomaly worth looking at by hand, not silently
    # attributing it to something).
    failing = [r for r in rows if r["pipeline_is_correct"].strip().lower() != "true"]
    unexpected_pass = [r for r in rows if r["pipeline_is_correct"].strip().lower() == "true"]
    if unexpected_pass:
        print(f"NOTE: {len(unexpected_pass)} task(s) in this pool actually succeeded "
              f"end to end ({[r['task_id'] for r in unexpected_pass]}) -- excluded from "
              f"the M_i/M denominator, since short-circuit attribution is only defined "
              f"over failures.")

    counts = {s: 0 for s in STAGE_ORDER}
    attributions = {}
    for r in failing:
        verdicts = {"per_d": r["per_d_verdict"], "rea_d": r["rea_d_verdict"], "per_t": r["per_t_verdict"]}
        earliest_wrong = next((s for s in ["per_d", "rea_d", "per_t"] if verdicts[s] == "WRONG"), None)
        stage = earliest_wrong if earliest_wrong is not None else "rea_t"
        counts[stage] += 1
        attributions[r["task_id"]] = stage

    M = len(failing)
    print(f"\n=== Short-circuit attribution (N={M} failing tasks) ===")
    for s in STAGE_ORDER:
        pct = 100 * counts[s] / M if M else 0
        print(f"  {STAGE_LABELS[s]:25s} M={counts[s]:2d}  {pct:5.1f}%")
    print(f"  {'TOTAL':25s} M={sum(counts.values()):2d}  {100*sum(counts.values())/M if M else 0:5.1f}%")

    print("\nPer-task attribution:")
    for task_id, stage in attributions.items():
        print(f"  {task_id}: {STAGE_LABELS[stage]}")

    # e_perceptionstage_d / e_perceptionstage_t: computed over ALL judged
    # tasks in this pool (not just the failing ones) -- perception can be
    # wrong on a task that still ends up correct overall (e.g. a lucky
    # self-correction downstream), and that still counts toward its own
    # intrinsic error rate.
    n_all = len(rows)
    e_per_d = sum(1 for r in rows if r["per_d_verdict"] == "WRONG") / n_all if n_all else 0
    e_per_t = sum(1 for r in rows if r["per_t_verdict"] == "WRONG") / n_all if n_all else 0
    print(f"\n=== Perception intrinsic error rates (N={n_all}, no oracle needed) ===")
    print(f"  e_perceptionstage_d = {e_per_d*100:.1f}%")
    print(f"  e_perceptionstage_t = {e_per_t*100:.1f}%")


if __name__ == "__main__":
    main()
