# Logistic Regression Implementation Plan

## Context

- Thesis: "Predictive Fire Spread Modelling Using Cellular Automata with Machine Learning"
- My role: Logistic Regression model (partner Kent handles Random Forest)
- Branch: `rf-training-8params-test` (Kent's RF base, converting to LR)
- Deadline: ~1 week for implementation, second week for writeups
- Quality bar: International conference presentation

## Current State

- Kent's RF best result on filtered data: F1=0.35, Recall=0.42
- Our earlier LR attempt on old branch: F1=0.41, Recall=0.64
- Both are far below thesis targets (F1 >= 0.80, Recall >= 0.80)
- Root cause: Single fire scenario (1 wind config, 5 ignition points, 40 timesteps) = insufficient training data

## Feature Set (9 features)

1. slope_risk (static, from raster)
2. proximity_risk (static, from raster)
3. building_presence (static, binary)
4. material_risk (static, from raster) ← NEW in this branch
5. wind_speed (static per simulation)
6. wind_sin (static per simulation)
7. wind_cos (static per simulation)
8. neighbor_burning_count (dynamic, 0-8)
9. composite_flammability (derived: building_presence * material_risk)

Target: Ignited (binary)

## Plan

### Phase 0: Setup

- [x] Clone `rf-training-8params-test` branch
- [x] Explore codebase and understand Kent's changes
- [ ] Create virtual environment and install dependencies
- [ ] Create skills (/validate, /train-lr, /analyze)
- [ ] Copy data files to expected locations

### Phase 1: Data Improvement (HIGHEST PRIORITY)

WHY: Both RF and LR fail with current data. This is the #1 lever.

- [ ] Write multi-scenario dataset generator wrapper

  - Vary wind speed: 5, 10, 15, 20, 25 km/h
  - Vary wind direction: 0, 45, 90, 135, 180, 225, 270, 315 degrees
  - Vary ignition points: different random seeds
  - Increase timesteps: 60-80 (more fire spread = more positive cases)
  - Research grounding: thesis Section 1.4 (varied wind conditions), Section 4.6 (sensitivity analysis), Gao et al. 2008 (wind velocity/direction effects)
- [ ] Run multi-scenario generation (depends on raster files being accessible)
- [ ] Analyze new dataset quality (class balance, feature distributions, wind variation)
- [ ] Document findings

NOTE: This step depends on having the raster TIF files accessible for the CA engine. If we can't run simulations (rasters are LFS/missing), we work with the existing 4,252-row dataset and optimize the model as best we can.

### Phase 2: LR Model Training

- [ ] Add LogisticRegression support to model_trainer.py

  - Use sklearn Pipeline (StandardScaler + LogisticRegression)
  - class_weight='balanced'
  - Threshold optimization (same approach Kent used for RF)
- [ ] Train on the dataset (new multi-scenario if available, else existing 4,252 rows)
- [ ] Evaluate: precision, recall, F1, AUC-ROC, Jaccard
- [ ] Compare with Kent's RF results
- [ ] Document results in findings.md

### Phase 3: Hyperparameter Tuning (if needed)

- [ ] Use Optuna for systematic tuning

  - C (regularization strength)
  - penalty (l1 vs l2)
  - solver (matching penalty choice)
  - threshold optimization
- [ ] Cross-validation to avoid overfitting on small dataset
- [ ] Document best hyperparameters and improvement over baseline

### Phase 4: Integration & Validation

- [ ] Verify LR model works with automata_engine.py (load_model + predict_proba)
- [ ] Run CA simulation with LR model
- [ ] Compare simulation output (fire spread pattern) with RF simulation
- [ ] Validate against thesis equations and methodology
- [ ] Generate visualizations (confusion matrix, ROC curve, feature importance, fire spread maps)

### Phase 5: Deliverables

- [ ] Trained LR model (.joblib)
- [ ] Comparison table: LR vs RF metrics
- [ ] Visualizations for thesis/conference
- [ ] Notes on thesis sections that need updating based on results
- [ ] findings.md with full documentation

## Risks

1. Raster files might not be runnable locally → can't generate more data → limited to 4,252 rows
2. LR may fundamentally underperform RF on this task (LR is linear, fire spread has nonlinear interactions) → this is actually a valid thesis finding if it happens
3. Time pressure → focus on Phase 1-2 first, Phase 3-4 if time permits

## Questions to Resolve

- What dataset did Kent use for his Training 2 results? (the 4,252 row one?)
- Can we run the raster pipeline locally?
- Does Kent plan to regenerate data with multi-scenario approach too?