---
name: run-pipeline
description: Run the full data pipeline (fetch → process → train) end-to-end. Use when the user wants to refresh data or retrain from scratch.
disable-model-invocation: false
---

Run the full World Cup 2026 prediction pipeline in this order. Report the output of each step before moving to the next.

1. **Fetch Elo data**
   ```
   uv run python -m world_cup_2026 data fetch --source elo
   ```

2. **Fetch FIFA rankings**
   ```
   uv run python -m world_cup_2026 data fetch --source fifa
   ```
   Note: this requires `raw/fifa_ranking_dates.csv` to exist. If it's missing, skip this step and warn the user.

3. **Build Elo match history**
   ```
   uv run python -m world_cup_2026 data process --step elo-history
   ```

4. **Parse FIFA snapshots**
   ```
   uv run python -m world_cup_2026 data process --step fifa-parse
   ```

5. **Merge ratings**
   ```
   uv run python -m world_cup_2026 data process --step merge-ratings
   ```

6. **Add geo features**
   ```
   uv run python -m world_cup_2026 data process --step geo
   ```

7. **Train XGBoost model**
   ```
   uv run python -m world_cup_2026 train --include-elo --include-fifa
   ```

After training completes, print the summary JSON and highlight log loss and accuracy metrics.
