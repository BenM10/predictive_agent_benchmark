# Agent Benchmark Scoring Framework

This document defines the evaluation criteria used to compare agent tools across benchmark tasks.

Each task is evaluated using five criteria. Each criterion is scored from **0–2**, giving a maximum score of **10 per task**.

---

## Metric Weighting

All evaluation criteria are equally weighted.

The purpose of the benchmark is exploratory rather than predictive, aiming to provide a transparent comparison of agent performance across multiple dimensions. Equal weighting avoids introducing subjective assumptions about the relative importance of criteria such as code quality, statistical validity, or efficiency.

Instead, each criterion contributes equally to the total score, while qualitative discussion in the report provides additional interpretation of trade-offs between metrics.

---

## Evaluation Criteria

| Criterion | Score | Description |
|-----------|------|-------------|
| Correctness | 0–2 | Whether the generated code runs successfully and completes the required task |
| Statistical Validity | 0–2 | Whether appropriate data science practices are followed (e.g. correct splits, no leakage, correct evaluation metrics) |
| Code Quality | 0–2 | Readability, structure, and maintainability of the generated code |
| Insight Quality | 0–2 | Whether meaningful insights or interpretations are produced (relevant for EDA tasks) |
| Efficiency | 0–2 | Number of iterations required to obtain a working solution |

Maximum Score per Task: **10**

---

## Iteration Tracking

Iterations are recorded separately from the score.

| Tool | Task | Initial Prompt | Follow-up Prompts | Total Iterations |
|-----|-----|-----|-----|-----|
| Claude | Task 1 | ✓ | 0 | 1 |
| Codex | Task 1 | ✓ | 2 | 3 |
| Antigravity | Task 1 | ✓ | 0 | 1 |

This allows efficiency comparisons between tools.

---

## Notes on Evaluation

- All tools receive **identical prompts** for each task.
- All experiments are conducted on the **same dataset**.
- Outputs and generated code are stored in the `results/` directory.
- Scores are assigned collaboratively by the project team.
