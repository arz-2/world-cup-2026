# Gemini CLI - World Cup 2026 Prediction

This document provides instructions for Gemini CLI on how to interact with this codebase.

## Project Structure
The codebase follows a modular architecture:
- `core/`: Shared schemas, constants, and global configuration.
- `data/ingestion/`: Modules that talk to the outside world (fetching).
- `data/processing/`: Logic for cleaning, team aliasing, and merging datasets into `matches_final.csv`.
- `features/`: Modular feature engineering. Registry-based entry point.
- `models/`: Sub-packages for different model types (XGBoost, Poisson).
- `simulation/`: Monte Carlo tournament simulation engine.

## Unified CLI Commands
Manage the entire pipeline via `uv run python -m world_cup_2026`:

### 1. Data Ingestion
- **Fetch Elo History**: `uv run python -m world_cup_2026 data fetch --source elo`
- **Fetch FIFA Rankings**: `uv run python -m world_cup_2026 data fetch --source fifa`

### 2. Data Processing
- **Build Elo History**: `uv run python -m world_cup_2026 data process --step elo-history`
- **Parse FIFA Snapshots**: `uv run python -m world_cup_2026 data process --step fifa-parse`
- **Merge Ratings**: `uv run python -m world_cup_2026 data process --step merge-ratings`
- **Add Geo Context**: `uv run python -m world_cup_2026 data process --step geo`

### 3. Training
- **XGBoost Experiment**: `uv run python -m world_cup_2026 train --include-elo --include-fifa`

## Coding Standards
- **Separation of Concerns**: Keep fetching logic in `ingestion/` and merging logic in `processing/`.
- **Schema Usage**: All predictions MUST return a `MatchPrediction` object from `core.schemas`.
- **Feature Registration**: Add new feature groups to `features/registry.py` to ensure they are available to all models.
- **Validation**: Use time-based cross-validation (16 folds) for all experiments to prevent temporal leakage.

## Current Focus: Phase 3
We are now moving into **Phase 3: Poisson Goal Models**. The goal is to predict $\lambda$ values for home and away goals to generate full scoreline distributions.
