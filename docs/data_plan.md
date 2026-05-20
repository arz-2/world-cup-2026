# Data Ingestion Plan

## Kaggle Backbone

Use `data/results.csv` as the match backbone and enrich it with:

- `data/shootouts.csv`
- `data/goalscorers.csv`
- `data/former_names.csv`

The parser in `src/world_cup_2026/data/kaggle.py` normalizes:

- match dates
- boolean flags
- optional scores for future fixtures
- optional goalscorer minutes
- historical team name aliases

## Elo Ingestion Plan

Target output:

- `team`
- `rating_date`
- `elo_rating`
- `elo_rank`
- `source_url`
- `retrieved_at`

Recommended approach:

1. Start with current ranking snapshots.
2. Add historical snapshot backfill.
3. Materialize daily or match-date aligned team ratings.

Engineering notes:

- keep the raw HTML payloads for reproducibility
- persist scrape metadata with fetch timestamp and source URL
- map team names into the same canonical naming system used by the Kaggle parser
- validate every rating snapshot against duplicate teams and missing ranks

## FIFA Ranking Ingestion Plan

Target output:

- `team`
- `ranking_date`
- `fifa_rank`
- `fifa_points`
- `rank_change`
- `source_url`
- `retrieved_at`

Recommended approach:

1. Scrape official ranking windows, not ad hoc match-day states.
2. Store the ranking release date exactly as published.
3. Forward-fill rankings to the next official release date when joining to matches.

Engineering notes:

- preserve the official ranking date parameter used in the source page
- capture both rank and points
- version the parser because FIFA page structure can change
- reconcile country naming against the canonical team mapping layer

## Normalized Warehouse Tables

Recommended processed tables:

- `processed/matches.csv`
- `processed/match_events_summary.csv`
- `processed/team_name_mapping.csv`
- `processed/team_elo_snapshots.csv`
- `processed/team_fifa_rankings.csv`

