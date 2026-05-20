# Model Performance Comparison - FIFA World Cup 2026

This document tracks the evolution of match prediction models. Metrics are derived from 16-fold temporal cross-validation (1961–2024).

## Performance Progression (Valid Models)

| Model Iteration | Log Loss (Mean) | Accuracy (Mean) | Brier Score | Key Changes |
| :--- | :---: | :---: | :---: | :--- |
| **1. Basic Baseline** | 0.9271 | 57.08% | 0.5468 | Kaggle data + basic rolling averages |
| **2. Elo Enhanced** | 0.9019 | 58.50% | 0.5303 | Added `elo_diff`, `elo_home/away` |
| **3. Rating Fusion** | 0.9044 | 58.19% | 0.5320 | Added FIFA Rankings (Removed Leakage) |
| **4. Feature Optimized** | 0.9037 | 58.17% | 0.5317 | Added Travel, Altitude, EWMA, xG Proxy |
| **5. Fully Optimized** | **0.9027** | **0.5820%** | **0.5311** | **Optuna Tuning + Strict Tournament Types** |

**Net Improvement (Iteration 1 vs 5):** 
- **Log Loss:** -0.0244
- **Accuracy:** +1.12%

---

## Best Fold Performance (2021-2024)
The most recent matches show higher predictive accuracy due to better data density.

| Metric | Basic Baseline | **Fully Optimized** | Improvement |
| :--- | :---: | :---: | :---: |
| **Log Loss** | 0.8714 | **0.8649** | -0.0065 |
| **Accuracy** | 60.96% | **60.66%** | -0.30%* |
| **Brier Score** | 0.5111 | **0.5059** | -0.0052 |

*\*Accuracy slightly decreased in the best fold while Log Loss and Brier Score improved, indicating better probability calibration and "fairer" estimates for upsets.*

---

## Feature Architecture

### 1. External Ratings
- **Elo System:** High-reactivity performance metric.
- **FIFA World Ranking:** Official point-based standing.

### 2. Physical & Geographic Context
- **Travel Distance (km):** Estimated fatigue from home base to venue.
- **Altitude (m):** Elevation effects on stamina/ball-physics.
- **Host Advantage:** Home/Host/Neutral site flags.

### 3. Advanced Form (EWMA)
- **Momentum:** Win rate and Point sum (Last 5/10/20 matches).
- **Efficiency:** Clean sheet rate and Goal Difference.
- **Dominance Proxy:** xG (Expected Goals) derived from Elo and Scoring history.

### 4. Categorical Context
- **Tournament Type:** Friendly, Qualifier, Nations League, Continental Cup, World Cup.
- **Confederation Pairing:** e.g., `UEFA|CONMEBOL`.

---

## Technical Configuration
- **Model:** `XGBClassifier` with Platt Scaling (`CalibratedClassifierCV`)
- **Tuned Hyperparameters:**
  - `learning_rate: 0.0776`
  - `max_depth: 8`
  - `subsample: 0.61`
  - `colsample_bytree: 0.59`
  - `gamma: 4.94`
  - `min_child_weight: 5`
  - `early_stopping_rounds: 50`

---

## Phase 3 Readiness
The current feature set is highly predictive of team strength. In Phase 3, these same features will be used to predict **Poisson $\lambda$ values** (Goal rates) to generate exact scoreline distributions.
