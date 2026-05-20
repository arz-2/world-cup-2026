"""
Fetch match-level xG from StatsBomb open data via statsbombpy.

Iterates over target competitions, loads shot events per match, sums
statsbomb_xg for home and away teams, and writes raw/statsbomb_xg.csv.

Run with:
    uv run python -m world_cup_2026 data fetch --source xg
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")  # suppress statsbombpy NoAuthWarning

DEFAULT_RAW_DIR = Path("raw")

# StatsBomb open-data international competitions with xG coverage
TARGET_COMPETITIONS = [
    (43, 106, "FIFA World Cup 2022"),
    (43, 3, "FIFA World Cup 2018"),
    (55, 282, "UEFA Euro 2024"),
    (55, 43, "UEFA Euro 2020"),
    (223, 282, "Copa America 2024"),
]


def fetch_statsbomb_xg(raw_dir: str | Path = DEFAULT_RAW_DIR) -> Path:
    from statsbombpy import sb

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    out_path = raw_path / "statsbomb_xg.csv"

    rows: list[dict] = []

    for comp_id, season_id, label in TARGET_COMPETITIONS:
        print(f"  Fetching {label}...")
        matches = sb.matches(competition_id=comp_id, season_id=season_id)

        for _, m in matches.iterrows():
            try:
                events = sb.events(match_id=m.match_id)
                shots = events[events["type"] == "Shot"]

                home_xg = float(
                    shots[shots["team"] == m.home_team]["shot_statsbomb_xg"]
                    .fillna(0)
                    .sum()
                )
                away_xg = float(
                    shots[shots["team"] == m.away_team]["shot_statsbomb_xg"]
                    .fillna(0)
                    .sum()
                )

                rows.append({
                    "match_date": str(m.match_date),
                    "home_team": m.home_team,
                    "away_team": m.away_team,
                    "home_xg_real": round(home_xg, 4),
                    "away_xg_real": round(away_xg, 4),
                    "competition": label,
                    "match_id_statsbomb": int(m.match_id),
                })
            except Exception as exc:
                print(f"    Skipped match {m.match_id} ({m.home_team} vs {m.away_team}): {exc}")

        print(f"    {len(matches)} matches processed.")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} matches → {out_path}")
    return out_path
