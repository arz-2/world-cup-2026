from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


from world_cup_2026.core.schemas import BaselineFeatureArtifacts
from world_cup_2026.core.constants import RESULT_TO_CODE

def build_baseline_training_frame(
    processed_dir: str | Path = "processed",
    include_elo: bool = False,
    include_fifa: bool = False,
) -> BaselineFeatureArtifacts:
    processed_path = Path(processed_dir)
    if include_elo and include_fifa:
        # Use the file with geo features if it exists, fallback to ratings
        if (processed_path / "matches_with_xg.csv").exists():
            matches_file = "matches_with_xg.csv"
        elif (processed_path / "matches_with_geo.csv").exists():
            matches_file = "matches_with_geo.csv"
        else:
            matches_file = "matches_with_ratings.csv"
    elif include_elo:
        matches_file = "matches_with_elo.csv"
    elif include_fifa:
        matches_file = "matches_with_fifa.csv"
    else:
        matches_file = "matches.csv"
        
    matches = pd.read_csv(processed_path / matches_file, parse_dates=["match_date"])

    completed_matches = matches[matches["is_future_fixture"] == False].copy()
    completed_matches = completed_matches.reset_index(drop=True)
    completed_matches["match_id"] = completed_matches.index
    completed_matches["result_code"] = completed_matches["result_label"].map(RESULT_TO_CODE)
    completed_matches["host_home"] = (
        (~completed_matches["neutral"]) & (completed_matches["country"] == completed_matches["home_team"])
    ).astype(int)
    completed_matches["host_away"] = (
        (~completed_matches["neutral"]) & (completed_matches["country"] == completed_matches["away_team"])
    ).astype(int)
    completed_matches["tournament_group"] = completed_matches["tournament"].map(_map_tournament_group)

    if include_fifa:
        # Load team confederations from snapshots
        fifa_snapshots = pd.read_csv(processed_path / "team_fifa_snapshots.csv")
        team_confed = fifa_snapshots.groupby("team")["confederation"].last().to_dict()
        completed_matches["home_confed"] = completed_matches["home_team"].map(team_confed).fillna("other")
        completed_matches["away_confed"] = completed_matches["away_team"].map(team_confed).fillna("other")
        completed_matches["confed_pairing"] = (
            completed_matches.apply(lambda row: "|".join(sorted([str(row["home_confed"]), str(row["away_confed"])])), axis=1)
        )

    # xG: use real StatsBomb xG where available, fall back to proxy
    has_real_xg = (
        "home_xg_real" in completed_matches.columns
        and completed_matches["home_xg_real"].notna().any()
    )
    if has_real_xg:
        if "elo_diff_pre" in completed_matches.columns:
            proxy_home = completed_matches["home_score"] * 0.9 + 0.1 * (1.5 + completed_matches["elo_diff_pre"] / 400.0)
            proxy_away = completed_matches["away_score"] * 0.9 + 0.1 * (1.5 - completed_matches["elo_diff_pre"] / 400.0)
        else:
            proxy_home = completed_matches["home_score"].astype(float)
            proxy_away = completed_matches["away_score"].astype(float)
        completed_matches["home_xg"] = completed_matches["home_xg_real"].fillna(proxy_home)
        completed_matches["away_xg"] = completed_matches["away_xg_real"].fillna(proxy_away)
    elif "elo_diff_pre" in completed_matches.columns:
        completed_matches["home_xg"] = completed_matches["home_score"] * 0.9 + 0.1 * (1.5 + completed_matches["elo_diff_pre"] / 400.0)
        completed_matches["away_xg"] = completed_matches["away_score"] * 0.9 + 0.1 * (1.5 - completed_matches["elo_diff_pre"] / 400.0)
    else:
        completed_matches["home_xg"] = completed_matches["home_score"]
        completed_matches["away_xg"] = completed_matches["away_score"]

    home_team_rows = _build_team_rows(completed_matches, side="home")
    away_team_rows = _build_team_rows(completed_matches, side="away")
    team_rows = pd.concat([home_team_rows, away_team_rows], ignore_index=True)
    team_features = _compute_team_rolling_features(team_rows)

    home_features = _prepare_side_features(team_features, side="home")
    away_features = _prepare_side_features(team_features, side="away")

    feature_frame = completed_matches.merge(home_features, on="match_id", how="left").merge(
        away_features,
        on="match_id",
        how="left",
    )
    feature_frame = feature_frame.merge(_compute_head_to_head_features(completed_matches), on="match_id", how="left")

    feature_frame["days_since_match"] = (
        feature_frame["match_date"] - feature_frame["match_date"].min()
    ).dt.days.astype(float)
    feature_frame["neutral_flag"] = feature_frame["neutral"].astype(int)
    feature_frame["host_advantage_flag"] = (feature_frame["host_home"] - feature_frame["host_away"]).astype(int)

    feature_frame["home_rest_days"] = feature_frame["home_rest_days"].fillna(feature_frame["home_rest_days"].median())
    feature_frame["away_rest_days"] = feature_frame["away_rest_days"].fillna(feature_frame["away_rest_days"].median())
    feature_frame["rest_days_diff"] = feature_frame["home_rest_days"] - feature_frame["away_rest_days"]

    numeric_columns = [
        "days_since_match",
        "neutral_flag",
        "host_advantage_flag",
        "rest_days_diff",
        "h2h_matches",
        "h2h_home_points_avg",
        "h2h_goal_diff_avg",
        "h2h_days_since_last",
    ]
    
    if "match_altitude" in feature_frame.columns:
        numeric_columns += ["match_altitude", "away_travel_km", "travel_diff_km"]

    numeric_columns += _side_feature_names("home")
    numeric_columns += _side_feature_names("away")

    if include_elo:
        numeric_columns += ["elo_home_pre", "elo_away_pre", "elo_diff_pre"]
    
    if include_fifa:
        numeric_columns += [
            "home_fifa_rank", "away_fifa_rank", "fifa_rank_diff",
            "home_fifa_points", "away_fifa_points", "fifa_points_diff"
        ]

    for column in numeric_columns:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce").fillna(0.0)

    categorical_columns = ["tournament_group"]
    if include_fifa:
        categorical_columns += ["confed_pairing"]
        
    feature_frame[categorical_columns] = feature_frame[categorical_columns].fillna("other")

    feature_columns = numeric_columns + categorical_columns
    feature_frame = feature_frame.sort_values("match_date").reset_index(drop=True)

    return BaselineFeatureArtifacts(
        feature_frame=feature_frame,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )


def _build_team_rows(matches: pd.DataFrame, side: str) -> pd.DataFrame:
    is_home = side == "home"
    team_column = "home_team" if is_home else "away_team"
    opponent_column = "away_team" if is_home else "home_team"
    goals_for_column = "home_score" if is_home else "away_score"
    goals_against_column = "away_score" if is_home else "home_score"
    xg_for_column = "home_xg" if is_home else "away_xg"
    xg_against_column = "away_xg" if is_home else "home_xg"
    goalscorers_column = "home_goalscorers" if is_home else "away_goalscorers"
    penalty_goals_column = "home_penalty_goals" if is_home else "away_penalty_goals"
    own_goals_for_column = "home_own_goals_for" if is_home else "away_own_goals_for"

    cols = [
        "match_date",
        team_column,
        opponent_column,
        goals_for_column,
        goals_against_column,
        xg_for_column,
        xg_against_column,
        goalscorers_column,
        penalty_goals_column,
        own_goals_for_column,
        "match_id",
    ]
    if "elo_home_pre" in matches.columns:
        cols.extend(["elo_home_pre", "elo_away_pre"])

    rows = matches[cols].copy()
    rows = rows.rename(
        columns={
            team_column: "team",
            opponent_column: "opponent",
            goals_for_column: "goals_for",
            goals_against_column: "goals_against",
            xg_for_column: "xg_for",
            xg_against_column: "xg_against",
            goalscorers_column: "goalscorers",
            penalty_goals_column: "penalty_goals",
            own_goals_for_column: "own_goals_for",
        }
    )
    if "elo_home_pre" in matches.columns:
        rows["team_elo"] = matches["elo_home_pre"] if is_home else matches["elo_away_pre"]
        rows["opponent_elo"] = matches["elo_away_pre"] if is_home else matches["elo_home_pre"]
    else:
        rows["team_elo"] = 1500.0
        rows["opponent_elo"] = 1500.0

    rows["side"] = side
    rows["goal_diff"] = rows["goals_for"] - rows["goals_against"]
    rows["points"] = rows["goal_diff"].map(lambda value: 3 if value > 0 else (1 if value == 0 else 0))
    rows["win"] = (rows["goal_diff"] > 0).astype(int)
    rows["clean_sheet"] = (rows["goals_against"] == 0).astype(int)
    
    # Adjusted points: points * (opponent_elo / 1500)
    rows["adjusted_points"] = rows["points"] * (rows["opponent_elo"] / 1500.0)
    return rows


def _compute_team_rolling_features(team_rows: pd.DataFrame) -> pd.DataFrame:
    ordered = team_rows.sort_values(["team", "match_date", "match_id"]).copy()
    grouped = ordered.groupby("team", group_keys=False)

    ordered["rest_days"] = grouped["match_date"].diff().dt.days
    ordered["matches_played"] = grouped.cumcount()

    rolling_specs = {
        "points": [5, 10, 20],
        "goal_diff": [5, 10, 20],
        "goals_for": [10],
        "goals_against": [10],
        "win": [5, 10],
        "clean_sheet": [5, 10],
        "adjusted_points": [5, 10, 20],
        "opponent_elo": [5, 10, 20],
        "xg_for": [5, 10],
        "xg_against": [5, 10],
    }

    for column, windows in rolling_specs.items():
        shifted = grouped[column].shift(1)
        # Expanding mean (career avg)
        ordered[f"{column}_career_avg"] = shifted.groupby(ordered["team"]).expanding().mean().reset_index(level=0, drop=True)
        
        for window in windows:
            # Traditional rolling mean
            ordered[f"{column}_last_{window}"] = (
                shifted.groupby(ordered["team"])
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            # Sum for points
            if column == "points":
                ordered[f"{column}_sum_{window}"] = (
                    shifted.groupby(ordered["team"])
                    .rolling(window=window, min_periods=1)
                    .sum()
                    .reset_index(level=0, drop=True)
                )
            # EWMA
            ordered[f"{column}_ewm_{window}"] = (
                shifted.groupby(ordered["team"])
                .apply(lambda x: x.ewm(span=window, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )

    return ordered


def _prepare_side_features(team_features: pd.DataFrame, side: str) -> pd.DataFrame:
    side_features = team_features[team_features["side"] == side].copy()
    
    # Dynamic feature mapping based on side
    feature_map = {"matches_played": f"{side}_matches_played", "rest_days": f"{side}_rest_days"}
    
    rolling_cols = [
        "points", "goal_diff", "goals_for", "goals_against", "win", 
        "adjusted_points", "opponent_elo", "clean_sheet", "xg_for", "xg_against"
    ]
    windows = [5, 10, 20]
    
    for col in rolling_cols:
        feature_map[f"{col}_career_avg"] = f"{side}_{col}_career_avg"
        for window in windows:
            last_key = f"{col}_last_{window}"
            if last_key in side_features.columns:
                feature_map[last_key] = f"{side}_{last_key}"
            
            sum_key = f"{col}_sum_{window}"
            if sum_key in side_features.columns:
                feature_map[sum_key] = f"{side}_{sum_key}"
                
            ewm_key = f"{col}_ewm_{window}"
            if ewm_key in side_features.columns:
                feature_map[ewm_key] = f"{side}_{ewm_key}"
                
    return side_features[["match_id", *feature_map.keys()]].rename(columns=feature_map)


def _compute_head_to_head_features(matches: pd.DataFrame) -> pd.DataFrame:
    ordered = matches.sort_values(["match_date", "match_id"]).copy()
    ordered["pair_key"] = ordered.apply(lambda row: "|".join(sorted((row["home_team"], row["away_team"]))), axis=1)
    ordered["home_points_from_result"] = ordered["result_label"].map({"home_win": 3, "draw": 1, "away_win": 0})
    ordered["home_goal_diff"] = ordered["home_score"] - ordered["away_score"]

    features: list[dict[str, float | int | str]] = []
    history: dict[str, dict[str, float | pd.Timestamp | int]] = {}

    for row in ordered.itertuples(index=False):
        pair_state = history.setdefault(
            row.pair_key,
            {"matches": 0, "home_points_sum": 0.0, "goal_diff_sum": 0.0, "last_match_date": None},
        )

        last_match_date = pair_state["last_match_date"]
        days_since_last = (
            float((row.match_date - last_match_date).days) if last_match_date is not None else float("nan")
        )
        matches_played = int(pair_state["matches"])
        features.append(
            {
                "match_id": row.match_id,
                "h2h_matches": matches_played,
                "h2h_home_points_avg": float(pair_state["home_points_sum"]) / matches_played if matches_played else 0.0,
                "h2h_goal_diff_avg": float(pair_state["goal_diff_sum"]) / matches_played if matches_played else 0.0,
                "h2h_days_since_last": days_since_last,
            }
        )

        pair_state["matches"] = matches_played + 1
        pair_state["home_points_sum"] = float(pair_state["home_points_sum"]) + float(row.home_points_from_result)
        pair_state["goal_diff_sum"] = float(pair_state["goal_diff_sum"]) + float(row.home_goal_diff)
        pair_state["last_match_date"] = row.match_date

    return pd.DataFrame(features)


def _map_tournament_group(tournament: str) -> str:
    value = tournament.lower()
    if "world cup qualification" in value or "qualifier" in value:
        return "Qualifier"
    if "world cup" in value:
        return "World Cup"
    if "friendly" in value:
        return "Friendly"
    if "nations league" in value:
        return "Nations League"
    if "cup" in value or "championship" in value or "copa" in value or "euro" in value:
        return "Continental Cup"
    return "Other"


def _side_feature_names(side: str) -> list[str]:
    names = [f"{side}_matches_played", f"{side}_rest_days"]

    # points: keep last/ewm across all windows; drop sum_N (= last_N × N, identical ranking)
    for window in [5, 10, 20]:
        names += [f"{side}_points_last_{window}", f"{side}_points_ewm_{window}"]
    names.append(f"{side}_points_career_avg")

    # goal_diff: keep all windows
    for window in [5, 10, 20]:
        names += [f"{side}_goal_diff_last_{window}", f"{side}_goal_diff_ewm_{window}"]
    names.append(f"{side}_goal_diff_career_avg")

    # goals_for / goals_against: window 10 only (as computed)
    for col in ["goals_for", "goals_against"]:
        names += [f"{side}_{col}_last_10", f"{side}_{col}_ewm_10", f"{side}_{col}_career_avg"]

    # win: dropped — correlated 0.95-0.99 with points_* (points = 3×win + draw)

    # adjusted_points: keep all windows
    for window in [5, 10, 20]:
        names += [f"{side}_adjusted_points_last_{window}", f"{side}_adjusted_points_ewm_{window}"]
    names.append(f"{side}_adjusted_points_career_avg")

    # opponent_elo: all windows are correlated 0.95-0.997; keep ewm_10 only
    names.append(f"{side}_opponent_elo_ewm_10")

    # clean_sheet: windows 5 and 10
    for window in [5, 10]:
        names += [f"{side}_clean_sheet_last_{window}", f"{side}_clean_sheet_ewm_{window}"]
    names.append(f"{side}_clean_sheet_career_avg")

    # xg_for / xg_against: windows 5 and 10
    for col in ["xg_for", "xg_against"]:
        for window in [5, 10]:
            names += [f"{side}_{col}_last_{window}", f"{side}_{col}_ewm_{window}"]
        names.append(f"{side}_{col}_career_avg")

    return names
