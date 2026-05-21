# FIFA World Cup 2026 Prediction

This repository provides a toolkit for predicting FIFA World Cup 2026 outcomes using match-level probabilities and Monte Carlo simulations.

## Project Architecture

The codebase is organized into a modular structure to ensure scalability and clarity:

- `core/`: Shared schemas (`MatchContext`, `MatchPrediction`), constants, and global configuration.
- `data/ingestion/`: Fetches raw data from external sources (Elo, FIFA, Kaggle).
- `data/processing/`: Logic for cleaning, team aliasing, and merging disparate datasets.
- `features/`: Engineered features including EWMA rolling form, geographic context, and rating signals.
- `models/`: Predictor implementations (XGBoost, Poisson goal models).
- `simulation/`: Monte Carlo tournament simulation engine.
- `scripts/`: Utility scripts for hyperparameter tuning (Optuna) and importance analysis.

## Current Progress: Phase 5 Completed

### Phase 2 — XGBoost Baseline
A fully optimized XGBoost classifier trained on historical match data with dual ratings, contextual features, and Platt-scaled calibration.

| Metric | Mean (16 folds) | Recent fold (2021–2024) |
|---|---|---|
| Log Loss | 0.9027 | 0.8649 |
| Accuracy | 58.2% | 60.7% |
| Brier Score | 0.5311 | 0.5059 |

### Phase 3 — Diagnostics + Poisson Goal Models + Feature Pruning

**3a — Diagnostics & pruning:** Calibration curves confirmed all three outcome classes are well-calibrated (MAE < 0.05). Feature importance and correlation analysis revealed 62 redundant columns — `points_sum_N` (algebraically identical to `points_last_N`), all `win_*` variants (≥0.95 correlated with `points_*`), and all but one `opponent_elo` window (0.95–0.997 correlated across windows). Pruned from `features/registry.py`, reducing 172 → 110 features. XGBoost baseline holds at 58.2% accuracy (61.35% recent fold).

**3b — Poisson λ predictor (final):** Three successive improvements to the Poisson model, then real xG integration:

| Iteration | Change | Mean Log Loss | Recent Fold LL | Goals MAE |
|---|---|---|---|---|
| Baseline | Linear GLM (PoissonRegressor) | 0.9216 | 0.8859 | 1.019 |
| + DC | Dixon-Coles ρ correction (ρ ≈ −0.05) | 0.9208 | 0.8850 | 1.019 |
| + XGB | XGBRegressor (count:poisson objective) | 0.9001 | 0.8648 | 0.986 |
| + clean | Removed fake xG features (target leakage) | 0.9007 | 0.8649 | 0.986 |
| **+ real xG** | **StatsBomb open data (262 intl. matches)** | **0.8999** | **0.8645** | **0.986** |

The Poisson model **beats the XGBoost classifier** on mean log loss (0.8999 vs 0.9027) while also producing full scoreline distributions for tournament simulation. The Dixon-Coles correction (ρ ≈ −0.05) addresses plain Poisson's tendency to over-predict 0-0 and 1-1 outcomes. Real StatsBomb xG is used for WC 2022/2018, Euro 2024/2020, and Copa América 2024 (262 total matches); all other matches fall back to the score-based proxy.

### Phase 5 — Bookmaker Odds Integration

Scraped pre-match 1X2 average odds from Betexplorer for all accessible international tournaments (WC 2022/2006, Euro 2024/2020/2004, Copa América 2024/2021), covering 1,526 matches (1,413 joined to training data, 2.9% coverage). Key findings:

| Aspect | Finding |
|---|---|
| **Scraper** | `data/ingestion/odds.py` — Betexplorer, 3-pass join (forward / neutral-venue swap / UTC date-shift) |
| **Feature threshold** | Odds features activate when coverage ≥ 15%; below that, sparse imputed values hurt tree models |
| **Poisson impact** | `odds_team_prob` context feature improves recent-fold log loss: **0.8645 → 0.8624** |
| **XGBoost impact** | No change (2.9% < 15% threshold); features ready for when WC 2026 odds are available |
| **Older tournaments** | WC 2018/2014, Euro 2016/2012/2008 odds not available in static HTML (JavaScript-rendered) |

The infrastructure is fully wired: once WC 2026 group-stage odds are published (days before each match), they can be injected via `data fetch --source odds` + `data process --step odds-merge`, at which point they will be included as XGBoost features for inference.

## Getting Started

### 1. Environment Setup
```bash
uv sync
```

### 2. Data Pipeline
```bash
uv run python -m world_cup_2026 data fetch --source elo
uv run python -m world_cup_2026 data fetch --source fifa

uv run python -m world_cup_2026 data process --step elo-history
uv run python -m world_cup_2026 data process --step fifa-parse
uv run python -m world_cup_2026 data process --step merge-ratings
uv run python -m world_cup_2026 data process --step geo

# Optional: fetch real StatsBomb xG (262 international matches) and merge
uv run python -m world_cup_2026 data fetch --source xg
uv run python -m world_cup_2026 data process --step xg-merge

# Optional: fetch bookmaker odds (Betexplorer) and merge
uv run python -m world_cup_2026 data fetch --source odds
uv run python -m world_cup_2026 data process --step odds-merge
```

### 3. Model Training
```bash
# XGBoost win/draw/loss classifier
uv run python -m world_cup_2026 train --include-elo --include-fifa

# Poisson expected-goals predictor
uv run python -m world_cup_2026 train --model poisson
```

### 4. Diagnostics
```bash
uv run python -m world_cup_2026.scripts.diagnostics
```

## Roadmap
- [x] Phase 1: Data Ingestion (Kaggle & Elo)
- [x] Phase 2: Optimized XGBoost Baseline
- [x] Phase 3: Diagnostics + Poisson Goal Models
  - [x] 3a: Calibration + feature importance + correlation pruning → 172 → 110 features
  - [x] 3b: Poisson λ predictor with Dixon-Coles correction + XGBoost regressor → log loss 0.9007, beats XGBoost classifier
- [ ] Phase 4: Monte Carlo Tournament Simulation (requires Phase 3b scoreline distributions)
- [x] Phase 5: Bookmaker Odds Integration — 1,526 matches from Betexplorer; Poisson recent-fold LL 0.8645 → 0.8624
- [x] Phase 6: Real xG Data — StatsBomb open data (262 intl. matches: WC 2022/18, Euro 2024/20, Copa América 2024)
