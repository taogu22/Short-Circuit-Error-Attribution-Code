# Short-Circuit Error Attribution Fails for Bottleneck Evaluation: code

Minimal code to reproduce the Mini-ARC experiment in the paper: short-circuit
error attribution ($M_i/M$) versus oracle substitution ($e_i$) on the
four-stage perception-reasoning pipeline of Wang et al. (2026), with OpenAI
o4-mini in all four roles and N = 30 Mini-ARC tasks (task00-task29).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your own OpenAI API key in .env
```

All scripts are run from this directory. Every model call uses `o4-mini` with
`reasoning_effort="medium"`. Prompts are in the scripts (also listed in
Appendix C of the paper). Runs call the OpenAI API and cost money; outputs can
differ from ours because the model is not deterministic.

`MiniARC/` holds the 30 Mini-ARC task files used (see `NOTICE`), and
`name_to_id.csv` maps `taskXX` to the file name.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 0. Render grids to images | `python render_all_tasks.py` | `images/taskXX/*.png` |
| 1. Run the pipeline (percept_d, reason_d, percept_t, reason_t) | `python regenerate_own_pipeline_v5.py` | `images/taskXX/taskXX_descriptions_v5.csv`, `results/TAE_results_v5.csv` |
| 2. LLM-judge the stages | `python judge_stages.py` | `results/judgments_v5.csv` |
| 3. Short-circuit shares $M_i/M$ and $e$ for the two perception stages | `python compute_attribution.py` | printed table |

Step 3 walks each failed task through the stages in order and blames the first
one judged wrong ($M_i/M$, over the $M$ failing tasks). The perception stages
always see the raw image, so their error rate over all $N$ tasks is already
$e_{\text{percept}_d}$ and $e_{\text{percept}_t}$.

## Oracle substitution

| Step | Command | Output |
|---|---|---|
| 4a. Generate oracle descriptions (o4-mini rewrites each description given the ground-truth grid) | `python extend_oracle_pipeline_v5.py` | `images/taskXX/taskXX_oracle_v5.csv` |
| 4b. **Review and edit** each `taskXX_oracle_v5.csv` by hand against the true grid | (manual) | edited `taskXX_oracle_v5.csv` |
| 4c. Rule induction on the oracle descriptions | `python run_oracle_rule_v5.py` | `results/TAE_results_oracle_v5.csv` |
| 4d. Judge the rules: set the oracle block in `judge_stages.py` (`JUDGE_REA_D = True`, `JUDGE_PER_D = JUDGE_PER_T = False`, `RESULTS_PATH = results/TAE_results_oracle_v5.csv`) and run it | `python judge_stages.py` | WRONG rate = $e_{\text{reason}_d}$ |
| 5a. Oracle rules for the test stage: rule summarization from the demo images | `python VLS_o4_v2.py` | `results/VLS_o4_v2.csv` |
| 5b. **Check by hand** the rules, correct the wrong ones, and save them as `results/TAE_oracle_rules.csv` | (manual) | `results/TAE_oracle_rules.csv` |
| 5c. Run only $\text{reason}_t$ with the oracle rule and oracle test description; the output grid is checked against ground truth by exact match | `python run_oracle_rea_t_v5.py` | `results/TAE_results_oracle_rea_t_v5.csv`; prints $e_{\text{reason}_t}$ |

Steps 4b and 5b are human-in-the-loop (as described in the paper).

## Outputs

No precomputed results are included. All outputs are written to `results/` and
`images/` (rendered PNGs and per-task description CSVs) when you run the
steps above. Steps 4b and 5b are manual: after running 4a and 5a, edit the
generated files yourself, as described in the paper.

## Citation

TODO (add after the paper is public).
