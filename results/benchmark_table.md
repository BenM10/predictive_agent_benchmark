# Benchmark Results

This table records the evaluation scores for each agent across the benchmark tasks.

Each criterion is scored from **0–2**, giving a maximum score of **10 per task**.

---

## Task 1 — Dataset Ingestion

| Tool | Correctness | Statistical Validity | Code Quality | Insight Quality | Efficiency | Total |
|-----|-----|-----|-----|-----|-----|-----|
| Antigravity | 2 | 2 | 1 | 1 | 2 | 8 |
| Claude | 2 | 2 | 1.5 | 1 | 1.5 | 8 |
| Codex | 1 | 1 | 1 | 1 | 1 | 5 |

---

## Task 2 — Exploratory Data Analysis

| Tool | Correctness | Statistical Validity | Code Quality | Insight Quality | Efficiency | Total |
|-----|-----|-----|-----|-----|-----|-----|
| Antigravity | 1.5 | 1.5 | 1 | 1 | 2 | 7 |
| Claude | 2 | 2 | 2 | 1.5 | 1.5 | 9 |
| Codex | 1 | 1 | 1 | 0.5 | 1 | 4.5 |

---

## Task 3 — Baseline Model

| Tool | Correctness | Statistical Validity | Code Quality | Insight Quality | Efficiency | Total |
|-----|-----|-----|-----|-----|-----|-----|
| Antigravity | 1.5 | 1.5 | 1.5 | 1 | 2 | 7.5 |
| Claude | 2 | 2 | 2 | 1.5 | 1.5 | 9 |
| Codex | 0.5 | 0.5 | 1 | 0.5 | 1 | 3.5 |

---

## Task 4 — Model Improvement

| Tool | Correctness | Statistical Validity | Code Quality | Insight Quality | Efficiency | Total |
|-----|-----|-----|-----|-----|-----|-----|
| Antigravity | 1.5 | 1.5 | 1.5 | 1 | 2 | 7.5 |
| Claude | 2 | 2 | 2 | 2 | 0.5 | 8.5 |
| Codex | 0.5 | 0.5 | 1 | 0.5 | 1 | 3.5 |

---

## Task 5 — Data Leakage Detection

| Tool | Correctness | Statistical Validity | Code Quality | Insight Quality | Efficiency | Total |
|-----|-----|-----|-----|-----|-----|-----|
| Antigravity | 1.5 | 1.5 | 1.5 | 1 | 2 | 7.5 |
| Claude | 2 | 2 | 2 | 1.5 | 2 | 9.5 |
| Codex | 1 | 1 | 1.5 | 1 | 1.5 | 6 |

---

## Overall Summary

| Tool | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Average Score | 
|-----|-----|-----|-----|-----|-----|-----|
| Antigravity | 8 | 8 | 6.5 | 5 | 10 | **7.5** |
| Claude | 10 | 10 | 9.5 | 7.5 | 7 | **8.8** |
| Codex | 4 | 4 | 5.5 | 3.5 | 5.5 | **4.5** |

---

# Iteration Tracking

Iterations are recorded seperately from the score, but have played a part in the *efficiency* scoring.
Recorded for each task is the number of follow up promtps required to achieve an appropriate standard of work. 

| Tool | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|-----|-----|-----|-----|-----|-----|
| Antigravity | 0 | 0 | 2 | 1 | 2 |
| Claude | 0 | 0 | 0 | 0 | 0 |
| Codex | 0 | 0 | 2 | 1 | 2 |
