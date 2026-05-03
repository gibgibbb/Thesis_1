# Logistic Regression for CA Fire Spread Prediction — Comprehensive Report

**Author:** Kristian Lemuel W. Diaz
**Thesis:** CA + ML Fire Spread Prediction in Lapu-Lapu City, Cebu
**Model responsibility:** Logistic Regression (LR) — partner Kent handles Random Forest (RF)
**Date:** 2026-04-11

---

## 1. Context and Objective

This thesis uses a Cellular Automata (CA) engine to simulate fire spread across a raster grid representing Lapu-Lapu City, Cebu. The CA assigns each cell a state (non-burnable, not-yet-burning, ignited, blazing, extinguished) and advances fire spread based on environmental factors and neighbor states.

The goal of the ML component is to replace the CA's rule-based ignition formula with a trained model that predicts the probability of ignition for each cell, given its features. Two models are compared: Logistic Regression (this report) and Random Forest (Kent's responsibility).

**Thesis performance targets:** F1 >= 0.80, Recall >= 0.80.

---

## 2. Features (9 Original)

The ML model receives 9 features per cell, assembled by `feature_pipeline.py`:

| # | Feature | Type | Source | Range |
|---|---------|------|--------|-------|
| 1 | slope_risk | float | Raster (DEM-derived) | 0.0 – 1.0 |
| 2 | proximity_risk | float | Raster (distance to roads/boundaries) | 0.0 – 1.0 |
| 3 | building_presence | binary | Raster (building footprints) | 0.0 or 1.0 |
| 4 | material_risk | float | Raster (building material classification) | 0.0 – 1.0 |
| 5 | wind_speed | float | Simulation config | 5.0 – 25.0 km/h |
| 6 | wind_sin | float | Derived from wind direction | -1.0 – 1.0 |
| 7 | wind_cos | float | Derived from wind direction | -1.0 – 1.0 |
| 8 | neighbor_burning_count | int | Dynamic (counted at each timestep) | 1 – 8 (filtered) |
| 9 | composite_flammability | float | building_presence x material_risk | 0.0 – 1.0 |

**Target variable:** `Ignited` (1 = cell ignited this timestep, 0 = did not).

**Training data filter:** Only "susceptible" cells are included — cells where `neighbor_burning_count > 0`. This matches what the CA engine actually evaluates. Including cells far from fire (count = 0) would make neighbor_burning_count a trivial separator and produce misleadingly perfect scores.

---

## 3. Model Architecture

**Pipeline:** `StandardScaler` -> `LogisticRegression` (sklearn)

The scaler is necessary because LR is sensitive to feature magnitudes (wind_speed ranges 5-25 while building_presence is 0-1). The pipeline wraps both into a single object with `predict_proba()`, which is required by `automata_engine.py` for CA integration.

**How LR works:**
1. Computes a weighted sum of features: `score = w1*f1 + w2*f2 + ... + bias`
2. Passes the score through a sigmoid function: `probability = 1 / (1 + e^(-score))`
3. If probability > threshold, predict "ignite"

The sigmoid makes the output a smooth probability (the S-curve), but the underlying decision is based on a linear (additive) combination of features. LR cannot represent interactions between features (e.g., "steep slope AND wooden building = extra dangerous") — it can only add their individual contributions.

---

## 4. Timeline of Work

### 4.1 Initial Exploration and Setup

- Cloned Kent's `rf-training-8params-test` branch (9 features including material_risk)
- Set up Python virtual environment with dependencies (pandas, scikit-learn, rasterio, etc.)
- Verified raster files load correctly (5489x6896 grid, EPSG:32651)
- Analyzed Kent's synthetic dataset: 4,252 susceptible rows, 462 positives (~10.9%)

### 4.2 Critical Finding: Wind Was Broken

**Discovery:** The CA engine (`automata_engine.py`) computed `wind_multiplier = 1.0 + 0.20 = 1.2` — a constant. Wind speed and direction had zero effect on fire spread. The 3 wind features (wind_speed, wind_sin, wind_cos) were recorded in training data but carried no information.

**Evidence:** LR coefficients for all wind features were exactly 0.000.

**Fix implemented:** Replaced the flat multiplier with a directional 3x3 wind kernel (`_compute_wind_kernel()`):
- Each of the 8 neighbor positions gets a weight based on wind alignment
- Upwind neighbors (fire traveling WITH the wind toward the candidate cell) get weight > 1.0
- Downwind neighbors (fire traveling AGAINST the wind) get weight closer to 0.05
- Scaled by `speed_factor = speed_kmh / 10.0`

**Physics grounding:** Alexandridis et al. (2011) and Gao et al. (2008) both model wind as a directional multiplier on spread probability in CA fire models.

### 4.3 Before/After Wind Fix Comparison

Trained LR on Kent's original dataset (before fix) and on a 40-scenario regenerated dataset (after fix):

| Metric | Before Fix (4,252 rows) | After Fix (2,091 rows) |
|--------|:-----------------------:|:----------------------:|
| Precision | 0.206 | 0.069 |
| Recall | 0.391 | 0.750 |
| F1 | 0.270 | 0.127 |
| AUC-ROC | 0.669 | 0.581 |

**Wind coefficients:**
| Feature | Before Fix | After Fix |
|---------|:----------:|:---------:|
| wind_speed | 0.000 | +0.096 |
| wind_sin | 0.000 | +0.060 |
| wind_cos | 0.000 | +0.027 |

**Conclusion:** Wind fix confirmed working — coefficients moved from zero to non-zero. F1 dropped because the after-fix dataset had only 102 positives (not enough data), not because the fix was wrong.

### 4.4 Expanded Dataset Generation

Regenerated data with larger configuration:
- 10 ignition points (was 5)
- 100 timesteps (was 60)
- 5 runs per wind config (was 1)
- 5 wind speeds x 8 directions x 5 runs = **200 total simulation runs**

**Result:** 21,079 rows with 1,095 positives (5.19%). Well-distributed across wind speeds.

### 4.5 LR Baseline on Expanded Dataset

| Metric | Value | Thesis Target |
|--------|:-----:|:-------------:|
| Precision | 0.077 | — |
| Recall | 0.470 | >= 0.80 |
| F1 | 0.133 | >= 0.80 |
| AUC-ROC | 0.630 | — |

F1 barely improved from the 40-scenario run (0.127 -> 0.133) despite 10x more data. This suggested the bottleneck was model architecture, not data quantity.

### 4.6 Path A: Hyperparameter Tuning (Optuna)

**Method:** 100 trials of Bayesian optimization (Optuna TPE sampler) with 5-fold stratified cross-validation on training data. Test set locked and never seen by Optuna.

**Search space:** C (0.0001-100, log), penalty (l1/l2/elasticnet), solver (compatible with penalty), class_weight (balanced or custom 1x-20x).

**Grounding:**
- Hyperparameter optimization: Bergstra & Bengio (2012), Akiba et al. (2019)
- Cross-validation: Stone (1974), Kohavi (1995)
- Bias-variance tradeoff: Hastie, Tibshirani & Friedman (2009)

**Result:**
| Metric | Baseline (defaults) | Optuna-tuned |
|--------|:-------------------:|:------------:|
| F1 | 0.133 | 0.129 |
| Recall | 0.470 | 0.379 |
| AUC-ROC | 0.630 | 0.629 |

CV F1 = 0.1375, Test F1 = 0.1294, gap = +0.008 (minimal, no overfitting).

**Best hyperparameters:** penalty=l2, solver=lbfgs, C=0.908, class_weight={0:1.0, 1:13.94}. The defaults were already near-optimal.

**Conclusion:** Tuning cannot fix a representational limitation. The bottleneck is not hyperparameters.

### 4.7 Path B: Feature Engineering + Optuna

Added 7 physics-informed interaction features:

| Feature | Formula | Justification |
|---------|---------|---------------|
| slope_x_neighbors | slope_risk x neighbor_burning_count | Fire travels uphill, compounds with neighbor presence (Alexandridis 2011) |
| wind_x_neighbors | wind_speed x neighbor_burning_count | Wind fans flames from neighbors (Gao 2008) |
| slope_x_building | slope_risk x building_presence | Buildings on slopes more vulnerable |
| slope_x_wind | slope_risk x wind_speed | Terrain-wind channeling effect |
| proximity_x_building | proximity_risk x building_presence | Wildland-urban interface risk |
| proximity_x_neighbors | proximity_risk x neighbor_burning_count | Accessibility to fire |
| neighbors_squared | neighbor_burning_count^2 | Nonlinear converging heat effect |

**Results (all 4 LR variants):**

| Variant | F1 | Recall | Precision | AUC-ROC |
|---------|:--:|:------:|:---------:|:-------:|
| 9 features, defaults | 0.133 | 0.470 | 0.077 | 0.630 |
| 9 features, Optuna (Path A) | 0.129 | 0.379 | 0.078 | 0.629 |
| 16 features, defaults (Path B-1) | 0.135 | 0.489 | 0.078 | 0.631 |
| 16 features, Optuna (Path B-2) | 0.134 | 0.489 | 0.078 | 0.629 |

**Notable coefficient findings from Path B Optuna (ElasticNet):**
- proximity_x_building was the strongest new feature (+0.124) — urban-interface interaction is real
- neighbors_squared replaced raw neighbor_burning_count (zeroed by L1) — nonlinear term is more informative
- 4 of 7 engineered features were zeroed out as redundant

**Conclusion:** Feature engineering did not break the ceiling. F1 remained at ~0.13 across all variants.

---

## 5. Key Findings

### 5.1 Wind Fix Validation
The directional wind kernel fix was confirmed working. Wind feature coefficients went from exactly zero (broken) to non-zero (functional) and remained non-zero across all subsequent experiments with larger datasets.

### 5.2 LR Ceiling Demonstrated
F1 ≈ 0.13 is the hard ceiling for LR on this task. Four systematic approaches — baseline, hyperparameter tuning, feature engineering, and both combined — all produced the same result. The ceiling is architectural: LR's linear decision boundary cannot capture the nonlinear feature interactions that drive fire ignition.

### 5.3 The Information Gap (Identified, Not Yet Addressed)
The CA engine computes a directional `wind_weighted_score` using a 3x3 kernel that weights each neighbor by wind alignment. But `_predict_with_model()` passes only `neighbor_burning_count` (a uniform integer count) to the ML model. The model cannot distinguish between 2 upwind burning neighbors (very dangerous) and 2 downwind burning neighbors (much safer). This information gap likely limits both LR and RF performance.

### 5.4 Circularity Acknowledgment
The ML model trains on data generated by the CA's own rules, then replaces those rules during simulation. This circularity is acknowledged as a limitation. It is mitigated by: (a) the CA is grounded in published fire physics, (b) ML learns emergent aggregate behavior across thousands of cells, not the formula itself, and (c) the methodology is transferable to real fire data when available.

---

## 6. Files Produced

| File | Purpose |
|------|---------|
| `dataFiles/multi_scenario_dataset.csv` | Training data: 21,079 rows, 200 scenarios |
| `models/fire_lr_model.joblib` | LR baseline (9 features, defaults) |
| `models/fire_lr_optuna.joblib` | LR Path A (9 features, Optuna-tuned) |
| `models/fire_lr_engineered_baseline.joblib` | LR Path B-1 (16 features, defaults) |
| `models/fire_lr_engineered_optuna.joblib` | LR Path B-2 (16 features, Optuna-tuned) |
| `models/optuna_study_lr.joblib` | Optuna study object (Path A, 100 trials) |
| `models/optuna_study_lr_engineered.joblib` | Optuna study object (Path B, 100 trials) |
| `modules/feature_engineering.py` | Interaction feature generation |
| `train_lr.py` | Baseline LR training script |
| `train_lr_optuna.py` | Path A training script |
| `train_lr_pathb.py` | Path B training script |
| `generate_multi_scenario.py` | Multi-scenario data generation (200 runs) |

---

## 7. Methodology References (Verify on Google Scholar before citing)

| Citation | What it grounds |
|----------|----------------|
| Alexandridis et al. (2011) | CA fire model, wind/slope modulation of spread probability |
| Gao et al. (2008) | Wind velocity and direction in CA fire models |
| Stone (1974) | Cross-validation for model selection |
| Kohavi (1995) | Stratified k-fold CV for reliable performance estimates |
| Hastie, Tibshirani & Friedman (2009) | Bias-variance tradeoff, model assessment (Ch. 7) |
| Bergstra & Bengio (2012) | Systematic hyperparameter search over manual/grid search |
| Akiba et al. (2019) | Optuna framework, TPE sampler |
| Geman, Bienenstock & Doursat (1992) | Bias-variance dilemma formalization |

---

## 8. Post-Report Update: Information Gap Investigation Completed

After the initial report, we investigated the hypothesis that models performed poorly due to an information gap between what the CA uses for ignition decisions and what the ML model receives as features.

### The Fix
Added `wind_weighted_score` as a 10th feature — the exact directional score the CA uses internally. Modified 3 files (backed up in `backups_option2/`):
- `feature_pipeline.py`, `dataset_generator.py`, `automata_engine.py`
Regenerated the 200-scenario dataset with the new feature.

### Final 8-Variant Comparison

| # | Variant | Features | F1 | Recall | Precision | AUC-ROC |
|---|---------|:--------:|:--:|:------:|:---------:|:-------:|
| 1 | 9feat, defaults | 9 | 0.133 | 0.470 | 0.077 | 0.630 |
| 2 | 9feat, Optuna | 9 | 0.129 | 0.379 | 0.078 | 0.629 |
| 3 | 9 + engineered, defaults | 16 | 0.135 | 0.489 | 0.078 | 0.631 |
| 4 | 9 + engineered, Optuna | 16 | 0.134 | 0.489 | 0.078 | 0.629 |
| 5 | 10feat, defaults | 10 | 0.162 | 0.306 | 0.110 | 0.656 |
| 6 | 10feat, Optuna | 10 | 0.164 | 0.237 | 0.125 | 0.656 |
| 7 | 10 + engineered, defaults | 17 | 0.165 | 0.320 | 0.111 | 0.658 |
| 8 | 10 + engineered, Optuna | 17 | **0.165** | 0.320 | 0.111 | 0.658 |

### Key Finding

**One physics-grounded feature outperformed seven hand-engineered interactions:**
- wind_weighted_score alone: F1 +0.029
- 7 engineered interactions: F1 +0.002
- 7 engineered on top of wind_weighted_score: F1 +0.003

The bottleneck was never LR's architecture — it was the feature pipeline fidelity between simulation physics and ML input.

### Updated Thesis Narrative

"We systematically investigated logistic regression's performance on CA-based fire spread prediction through 8 configurations (4 feature sets x 2 tuning strategies). During this investigation, we identified and corrected an information gap in the feature pipeline where the CA uses a directional wind-weighted neighbor score for ignition decisions but the ML model only received a uniform neighbor count. Correcting this gap improved F1 by 22% (0.133 → 0.165), while systematic hyperparameter tuning and physics-informed feature engineering each produced negligible gains. This demonstrates that for CA-ML hybrid systems, fidelity between simulation physics and ML input features is the primary performance driver."

## 9. What Remains

1. **RF comparison:** Kent needs to retrain RF on updated `multi_scenario_dataset.csv` (21,079 rows, 10 features with wind_weighted_score) for fair LR vs RF comparison. He also needs the updated `automata_engine.py` and `feature_pipeline.py`.
2. **Visualization:** Fire spread maps, coefficient comparison charts, ROC curves
3. **Thesis writeup:** Methodology section (including the information gap discovery), results section, discussion of LR limitations and the pipeline fidelity finding

---

## 9. Summary for Conference Presentation

The progression narrative:

1. **Discovered wind was broken** in the CA engine — fixed it with a directional kernel grounded in Alexandridis (2011) and Gao (2008). Wind coefficients went from zero to non-zero, confirming the fix.

2. **Generated diverse training data** — 200 simulation runs across 5 wind speeds, 8 directions, 5 random seeds each. 21,079 susceptible cell records with 1,095 positive ignition events.

3. **Established LR baseline** — F1 = 0.133. Identified that 10x more data did not improve performance, suggesting an architectural limitation.

4. **Systematically tuned LR** — Bayesian optimization (Optuna) with 5-fold stratified CV across 100 trials. Result: F1 = 0.129. Default hyperparameters were already near-optimal. The ceiling is not a tuning problem.

5. **Engineered physics-informed features** — Added 7 interaction terms grounded in fire literature. Result: F1 = 0.135. The ceiling is not a feature representation problem that 2-way interactions can solve.

6. **Conclusion (updated, May 3 2026):** The per-cell findings suggested LR's linear architecture was insufficient. However, the subsequent information-gap fix and spatial validation substantially revised this conclusion — see Section 10 below.

---

## 10. Spatial Validation — May 3, 2026 Update

After the diagnostic per-cell investigation, we conducted spatial validation against the Sitio Santa Maria, Lapu-Lapu City fire of December 12, 2023, using the partner-supplied ground truth raster (manually digitized from Google satellite imagery; BFP-supplied burned-area polygons unavailable).

### 10.1 Methodology
- Both LR variants run through the full CA + ML simulation
- Ignition point: GPS coords (606748.78, 1141695.77) → grid (892, 2730)
- Wind: 10 km/h NW, NOT matched to actual Dec 12 2023 weather (acknowledged limitation)
- 100 timesteps, early-stop enabled
- Comparison metric: cells in CA states {3, 4, 5} treated as "fire" predictions; compared against binary ground truth using `validate_simulation.py`

### 10.2 Two-Stage Comparison

To isolate the effect of each candidate 10th feature:

| Variant | 10th feature | F1 | Recall | Precision | AUC-ROC |
|---------|--------------|:--:|:------:|:---------:|:-------:|
| CA only (baseline, partner-reported) | — | 0.684 | 0.544 | 0.921 | 0.772 |
| Simulation-RF (partner, broken-wind training data) | material_class | 0.785 | 0.963 | 0.663 | 0.981 |
| **Simulation-LR Stage A** | material_class | **0.602** | **0.915** ✓ | 0.448 | 0.958 |
| **Simulation-LR Stage B** | wind_weighted_score | **0.607** | **0.925** ✓ | 0.452 | 0.963 |

Both LR variants pass the proposal's primary recall target (≥ 0.80). F1 falls below the secondary 0.80 target due to ~2× over-prediction of burn area.

### 10.3 Key Spatial Findings

1. **Recall target met by both stages.** The proposal designates recall as the primary metric; Stage A and Stage B both clear it by a wide margin
2. **Per-cell F1 gain does not translate proportionally to spatial F1.** `wind_weighted_score` improved per-cell F1 by +0.029 but spatial F1 by only +0.005
3. **Stage A ≈ Stage B spatially.** `material_class` and `wind_weighted_score` produce nearly identical spatial outcomes when each is added to the 9 base features
4. **The "architectural ceiling" argument from §3.4 needs softening.** Per-cell F1 ≈ 0.165 looked like a hard limit, but spatial F1 = 0.61 with all features available — the model is functioning well enough to catch >91% of real fire cells. The "ceiling" was largely a per-cell artifact

### 10.4 LR vs RF — Direct Comparison Not Yet Available

The reported partner RF result (F1 = 0.785) was produced on the original synthetic dataset that predates the wind-kernel fix. Both LR stages above use the wind-fixed dataset. Until the partner retrains RF on Stage A or Stage B data, the LR vs RF comparison is **indicative only**.

### 10.5 Visualizations
- `Code/sandbox_kent/output/stage_a_comparison.png`
- `Code/output/stage_b_comparison.png`

### 10.6 Revised Framing for the Thesis

The per-cell investigation chapter remains intact. What changes is the framing of the contribution:
- Before: "We discovered an information gap that broke the LR ceiling"
- After: "We discovered an information gap that **substantially improves per-cell prediction but only marginally improves spatial prediction**, demonstrating that per-cell and spatial metrics measure different things and should be reported jointly"

This dual-scale evaluation methodology is itself a methodological contribution. Many CA-ML papers report only per-cell metrics; the spatial divergence here suggests this overstates operational value.
