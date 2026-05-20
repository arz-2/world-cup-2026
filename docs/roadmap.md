# Delivery Roadmap

## 1. Prediction Framing

Do not predict the champion directly. Predict match probabilities first, then estimate tournament outcomes through Monte Carlo simulation.

### Core outputs

- `P(team_a_win)`
- `P(draw)`
- `P(team_b_win)`
- exact scoreline probabilities
- tournament advancement probabilities

## 2. Data Layers

### Historical matches

Minimum fields:

- `match_date`
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`
- `tournament`
- `neutral_site`
- `venue_country`

Optional but high value:

- `home_xg`
- `away_xg`
- `rest_days`
- `travel_distance_km`

### Team ratings

Start with:

- Elo
- FIFA rank

Later:

- custom attack / defense ratings
- bookmaker priors

### Player and squad context

- squad value
- injury absences
- average caps
- club minutes
- roster continuity

### Tournament context

- host advantage
- rest differential
- altitude
- climate
- confederation matchup

## 3. Modeling Order

### Baseline 1 — Elo classifier

Primary features:

- Elo difference
- neutral site
- host flag
- recent form
- match importance

### Baseline 2 — Poisson goals

Model expected goals for both teams, then derive scoreline and result probabilities.

### Baseline 3 — Gradient boosting

Candidate models:

- XGBoost
- LightGBM
- CatBoost

## 4. Evaluation

Use:

- log loss
- Brier score
- calibration plots
- out-of-time validation by international window or tournament cycle

Benchmark against:

- bookmaker closing odds
- public rating systems

## 5. Simulation Engine

Support:

- group stage points and tiebreakers
- knockout brackets
- extra time
- penalties

Run at least `100_000` simulations for stable tournament estimates.

## 6. Production Path

Recommended stack:

- Python
- pandas / polars
- scikit-learn
- numpy
- plotly / streamlit

