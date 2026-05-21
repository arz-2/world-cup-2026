"""
Merge bookmaker 1X2 odds into the processed match history.

Reads raw/bookmaker_odds.csv, converts decimal odds to normalized implied
probabilities (vigorish removed), then left-joins onto
processed/matches_with_xg.csv on (match_date, home_team, away_team).

The same three-pass join strategy as xg.py is used:
  1. Forward join (betexplorer home == Kaggle home)
  2. Reversed join (neutral-venue home/away swap)
  3. ±1-day date-shifted join (UTC vs local timezone)

Output columns added:
  odds_home_prob  — market implied P(home win), overround removed
  odds_draw_prob  — market implied P(draw),     overround removed
  odds_away_prob  — market implied P(away win),  overround removed
  odds_overround  — raw sum of implied probs (>1 = bookmaker margin)

Rows without odds match keep NaN (imputed to median by the feature pipeline).

Run with:
    uv run python -m world_cup_2026 data process --step odds-merge
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_cup_2026.data.processing.xg import _normalize_text, _resolve

DEFAULT_PROCESSED_DIR = Path("processed")
DEFAULT_RAW_DIR = Path("raw")

# Betexplorer uses its own team naming conventions
_BETEXPLORER_OVERRIDES: dict[str, str] = {
    "usa": "United States",
    "united states": "United States",
    "korea republic": "South Korea",
    "republic of ireland": "Republic of Ireland",
    "czech republic": "Czech Republic",
    "north macedonia": "North Macedonia",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "ir iran": "Iran",
    "china pr": "China",
}


def _resolve_odds(name: str) -> str:
    low = _normalize_text(name)
    if low in _BETEXPLORER_OVERRIDES:
        return _BETEXPLORER_OVERRIDES[low]
    return _resolve(name)


def _add_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Convert decimal odds to overround-corrected implied probabilities."""
    raw_h = 1.0 / df["h_odd"]
    raw_d = 1.0 / df["d_odd"]
    raw_a = 1.0 / df["a_odd"]
    overround = raw_h + raw_d + raw_a
    df = df.copy()
    df["odds_home_prob"] = (raw_h / overround).round(4)
    df["odds_draw_prob"] = (raw_d / overround).round(4)
    df["odds_away_prob"] = (raw_a / overround).round(4)
    df["odds_overround"] = overround.round(4)
    return df


def _make_join_frame(odds_raw: pd.DataFrame, shift_days: int = 0, swap: bool = False) -> pd.DataFrame:
    dates = pd.to_datetime(odds_raw["match_date"]) + pd.Timedelta(days=shift_days)
    if swap:
        return pd.DataFrame({
            "date_key": dates.dt.date.astype(str),
            "home_team": odds_raw["away_team_canon"].values,
            "away_team": odds_raw["home_team_canon"].values,
            "odds_home_prob": odds_raw["odds_away_prob"].values,
            "odds_draw_prob": odds_raw["odds_draw_prob"].values,
            "odds_away_prob": odds_raw["odds_home_prob"].values,
            "odds_overround": odds_raw["odds_overround"].values,
        })
    return pd.DataFrame({
        "date_key": dates.dt.date.astype(str),
        "home_team": odds_raw["home_team_canon"].values,
        "away_team": odds_raw["away_team_canon"].values,
        "odds_home_prob": odds_raw["odds_home_prob"].values,
        "odds_draw_prob": odds_raw["odds_draw_prob"].values,
        "odds_away_prob": odds_raw["odds_away_prob"].values,
        "odds_overround": odds_raw["odds_overround"].values,
    })


def merge_bookmaker_odds(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> Path:
    processed_path = Path(processed_dir)
    raw_path = Path(raw_dir)

    src_file = "matches_with_xg.csv" if (processed_path / "matches_with_xg.csv").exists() else "matches_with_geo.csv"
    matches = pd.read_csv(processed_path / src_file, low_memory=False)
    odds_raw = pd.read_csv(raw_path / "bookmaker_odds.csv")

    # Canonicalise betexplorer team names
    odds_raw["home_team_canon"] = odds_raw["home_team"].map(_resolve_odds)
    odds_raw["away_team_canon"] = odds_raw["away_team"].map(_resolve_odds)

    # Derive normalized probability columns
    odds_raw = _add_probs(odds_raw)

    matches["date_key"] = pd.to_datetime(matches["match_date"]).dt.date.astype(str)

    ODDS_COLS = ["odds_home_prob", "odds_draw_prob", "odds_away_prob", "odds_overround"]

    for col in ODDS_COLS:
        matches[col] = float("nan")

    def _patch(mask: pd.Series, join_df: pd.DataFrame) -> None:
        if not mask.any():
            return
        sub = matches.loc[mask, ["date_key", "home_team", "away_team"]].merge(
            join_df, on=["date_key", "home_team", "away_team"], how="left"
        )
        for col in ODDS_COLS:
            if col in sub.columns:
                matches.loc[mask, col] = sub[col].values

    # Pass 1: forward (same home/away)
    _patch(matches["odds_home_prob"].isna(), _make_join_frame(odds_raw))
    # Pass 2: reversed (neutral venue home/away swap)
    _patch(matches["odds_home_prob"].isna(), _make_join_frame(odds_raw, swap=True))
    # Pass 3a: date −1, forward (UTC ahead of local)
    _patch(matches["odds_home_prob"].isna(), _make_join_frame(odds_raw, shift_days=-1))
    # Pass 3b: date −1, reversed
    _patch(matches["odds_home_prob"].isna(), _make_join_frame(odds_raw, shift_days=-1, swap=True))

    matches.drop(columns=["date_key"], inplace=True)

    n_matched = matches["odds_home_prob"].notna().sum()
    print(f"Matched {n_matched} / {len(odds_raw)} odds rows into the match history.")

    out_path = processed_path / "matches_with_odds.csv"
    matches.to_csv(out_path, index=False)
    print(f"Saved → {out_path}  ({len(matches)} rows)")
    return out_path
