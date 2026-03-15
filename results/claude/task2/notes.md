# Task 2 – Exploratory Data Analysis
**Agent:** Claude (claude-sonnet-4-6)
**Date:** 2026-03-15

---

## Dataset Used
`data/processed/Claude_clean.csv` — 129,487 rows × 24 columns

---

## Key Findings

### Class Balance
| Class | Count | % |
|-------|-------|---|
| neutral or dissatisfied | 73,225 | 56.6% |
| satisfied | 56,262 | 43.4% |

Mild class imbalance. Models should account for this (e.g. class weighting or stratified splits).

---

### Categorical Feature Patterns (Fig 2)
- **Type of Travel:** Business travellers are far more satisfied (~58%) than personal travellers (~24%). This is the strongest categorical signal.
- **Class:** Business class passengers are most satisfied (~69%); Eco (~19%) and Eco Plus (~24%) are much less so.
- **Customer Type:** Loyal customers are substantially more satisfied (~48%) than disloyal (~24%).
- **Gender:** Negligible difference between male and female satisfaction rates.

---

### Age Distribution (Fig 3)
- Satisfied passengers skew slightly older (peak ~40–55), while dissatisfied passengers are more concentrated at younger ages (~20–35) and very young passengers (<18).

---

### Flight Distance (Fig 4)
- Satisfied passengers tend to fly longer distances. Short-haul flights (<500 miles) are more associated with dissatisfaction.

---

### Service Ratings (Fig 5)
- Across all 14 service categories, satisfied passengers consistently rate every service higher.
- The largest gaps are in **Online boarding**, **Inflight entertainment**, and **Seat comfort**.
- Even the smallest gap services (e.g. **Departure/Arrival time convenient**) still show meaningful differences.

---

### Feature Correlations with Satisfaction (Figs 6 & 7)

| Feature | Correlation |
|---------|------------|
| Online boarding | +0.502 |
| Inflight entertainment | +0.398 |
| Seat comfort | +0.349 |
| On-board service | +0.322 |
| Leg room service | +0.313 |
| Cleanliness | +0.307 |
| Flight Distance | +0.298 |
| Inflight wifi service | +0.283 |
| Arrival Delay in Minutes | -0.102 |
| Departure Delay in Minutes | -0.097 |

- **Online boarding** is the single strongest numeric predictor.
- **Delays** are weakly negatively correlated — delays alone are not the primary driver of dissatisfaction.
- Service rating features dominate over demographic or trip-type features in linear correlation.

---

## Modelling Implications
1. **Online boarding, Inflight entertainment, Seat comfort** — prioritise these as features.
2. **Type of Travel** and **Class** are strong categorical predictors — include as encoded features.
3. Class imbalance is mild (~57/43 split) — monitor precision/recall not just accuracy.
4. Most service ratings are moderately inter-correlated (Fig 6) — consider dimensionality reduction or regularised models to handle multicollinearity.
5. Delay columns have low linear correlation but may have non-linear effects — worth including in tree-based models.

---

## Figures Produced

| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | `fig_01_satisfaction_distribution.png` | Class distribution bar chart |
| Fig 2 | `fig_02_satisfaction_by_categorical.png` | Satisfaction rate by 4 categorical features |
| Fig 3 | `fig_03_age_distribution_by_satisfaction.png` | Age histogram by satisfaction group |
| Fig 4 | `fig_04_flight_distance_by_satisfaction.png` | Flight distance histogram by satisfaction group |
| Fig 5 | `fig_05_service_ratings_by_satisfaction.png` | Mean service ratings grouped bar chart |
| Fig 6 | `fig_06_correlation_heatmap.png` | Lower-triangle correlation heatmap |
| Fig 7 | `fig_07_feature_correlations_with_satisfaction.png` | Ranked feature correlations with target |

All figures saved to `figures/Claude/`.
