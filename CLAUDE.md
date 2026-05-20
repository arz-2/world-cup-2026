# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
uv sync          # install dependencies (including dev extras)
uv run pytest    # run tests
uv run python -m world_cup_2026 <command>  # run CLI
```

No env vars or API keys are required.

## Data Pipeline

Run steps in order — each step's output is the next step's input:

```bash
uv run python -m world_cup_2026 data fetch --source elo
uv run python -m world_cup_2026 data fetch --source fifa   # requires raw/fifa_ranking_dates.csv

uv run python -m world_cup_2026 data process --step elo-history
uv run python -m world_cup_2026 data process --step fifa-parse
uv run python -m world_cup_2026 data process --step merge-ratings
uv run python -m world_cup_2026 data process --step geo

uv run python -m world_cup_2026 train --include-elo --include-fifa
```

Processed CSVs live in `processed/`. The most complete file is `matches_with_geo.csv` (requires all steps). Training auto-selects the richest available file.

## Architectural Invariants

- **All match predictions must return a `MatchPrediction` object** from `core.schemas`. Never return raw dicts or tuples.
- **New feature groups must be registered in `features/registry.py`** to be available across all models.
- **Use time-based cross-validation (16 folds) only.** Never use random train/test splits — temporal leakage will silently inflate metrics.
- **Simulation engine requires ≥ 100,000 iterations** for stable tournament probability estimates.
- Fetching logic belongs in `data/ingestion/`; merging/cleaning logic belongs in `data/processing/`.

## Current Phase

Phase 3: Poisson goal models. Goal is to predict λ values for home and away goals to generate full scoreline distributions, then derive match result probabilities from those.

## Git

Push directly to `main`. No PRs.
