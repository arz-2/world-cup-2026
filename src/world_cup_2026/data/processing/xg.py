"""
Join StatsBomb real xG to the processed match history.

Reads raw/statsbomb_xg.csv (produced by data/ingestion/statsbomb.py) and
left-joins it onto processed/matches_with_geo.csv on (match_date, home_team,
away_team).  StatsBomb team names are normalised via the shared alias table
plus a small StatsBomb-specific override map.  Outputs
processed/matches_with_xg.csv; unmatched rows keep NaN for xg columns
(the feature pipeline imputes these with a median fallback).

Run with:
    uv run python -m world_cup_2026 data process --step xg-merge
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_cup_2026.data.processing.aliases import DEFAULT_SOURCE_ALIASES

def _normalize_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())

DEFAULT_PROCESSED_DIR = Path("processed")
DEFAULT_RAW_DIR = Path("raw")

# Build a flat lookup: lower-stripped source_name → canonical_name
_ALIAS_LOOKUP: dict[str, str] = {
    _normalize_text(a.source_name): a.canonical_name for a in DEFAULT_SOURCE_ALIASES
}

# StatsBomb uses its own naming for some nations — map to Kaggle canonical names
_STATSBOMB_OVERRIDES: dict[str, str] = {
    "united states": "United States",
    "korea republic": "South Korea",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "north macedonia": "North Macedonia",
    "czech republic": "Czech Republic",
    "curacao": "Curaçao",
    # Euro / Copa squad naming differences
    "ir iran": "Iran",
    "china pr": "China",
    "guinea-bissau": "Guinea-Bissau",
}


def _resolve(name: str) -> str:
    low = _normalize_text(name)
    if low in _STATSBOMB_OVERRIDES:
        return _STATSBOMB_OVERRIDES[low]
    if low in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[low]
    return name


def merge_statsbomb_xg(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> Path:
    processed_path = Path(processed_dir)
    raw_path = Path(raw_dir)

    matches = pd.read_csv(processed_path / "matches_with_geo.csv", low_memory=False)
    xg_raw = pd.read_csv(raw_path / "statsbomb_xg.csv")

    # Canonicalise StatsBomb team names
    xg_raw["home_team_canon"] = xg_raw["home_team"].map(_resolve)
    xg_raw["away_team_canon"] = xg_raw["away_team"].map(_resolve)
    xg_raw["date_key"] = pd.to_datetime(xg_raw["match_date"]).dt.date.astype(str)

    matches["date_key"] = pd.to_datetime(matches["match_date"]).dt.date.astype(str)

    # Forward join: StatsBomb home == Kaggle home
    xg_fwd = xg_raw[["date_key", "home_team_canon", "away_team_canon", "home_xg_real", "away_xg_real"]].rename(
        columns={"home_team_canon": "home_team", "away_team_canon": "away_team"}
    )
    merged = matches.merge(xg_fwd, on=["date_key", "home_team", "away_team"], how="left")

    # Reverse join: StatsBomb swapped home/away for neutral-venue games
    unmatched_mask = merged["home_xg_real"].isna()
    if unmatched_mask.any():
        xg_rev = xg_raw[["date_key", "home_team_canon", "away_team_canon", "home_xg_real", "away_xg_real"]].rename(
            columns={
                "home_team_canon": "away_team",
                "away_team_canon": "home_team",
                "home_xg_real": "away_xg_real_rev",
                "away_xg_real": "home_xg_real_rev",
            }
        )
        patched = merged.loc[unmatched_mask].drop(columns=["home_xg_real", "away_xg_real"]).merge(
            xg_rev, on=["date_key", "home_team", "away_team"], how="left"
        )
        patched = patched.rename(columns={"home_xg_real_rev": "home_xg_real", "away_xg_real_rev": "away_xg_real"})
        merged.loc[unmatched_mask, "home_xg_real"] = patched["home_xg_real"].values
        merged.loc[unmatched_mask, "away_xg_real"] = patched["away_xg_real"].values

    # Date-shifted fallback: StatsBomb Copa América uses UTC dates (1 day ahead of US local)
    unmatched_mask = merged["home_xg_real"].isna()
    if unmatched_mask.any():
        xg_shifted = xg_raw.copy()
        xg_shifted["date_key"] = (
            pd.to_datetime(xg_raw["match_date"]) - pd.Timedelta(days=1)
        ).dt.date.astype(str)
        for try_fwd, h_col, a_col in [
            (True, "home_team_canon", "away_team_canon"),
            (False, "away_team_canon", "home_team_canon"),
        ]:
            still_mask = merged["home_xg_real"].isna()
            if not still_mask.any():
                break
            xg_try = xg_shifted[["date_key", h_col, a_col, "home_xg_real", "away_xg_real"]].rename(
                columns={h_col: "home_team", a_col: "away_team"}
            )
            if not try_fwd:
                xg_try = xg_try.rename(columns={"home_xg_real": "away_xg_real", "away_xg_real": "home_xg_real"})
            sub = merged.loc[still_mask].drop(columns=["home_xg_real", "away_xg_real"]).merge(
                xg_try, on=["date_key", "home_team", "away_team"], how="left"
            )
            merged.loc[still_mask, "home_xg_real"] = sub["home_xg_real"].values
            merged.loc[still_mask, "away_xg_real"] = sub["away_xg_real"].values

    merged.drop(columns=["date_key"], inplace=True)

    n_matched = merged["home_xg_real"].notna().sum()
    print(f"Matched {n_matched} / {len(xg_raw)} StatsBomb xG rows into the match history.")

    out_path = processed_path / "matches_with_xg.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved → {out_path}  ({len(merged)} rows)")
    return out_path
