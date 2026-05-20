from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from world_cup_2026.data.processing.aliases import load_source_aliases, resolve_team_name

DEFAULT_PROCESSED_PATH = Path("processed/team_fifa_snapshots.csv")
DEFAULT_ALIAS_PATH = Path("config/team_source_aliases.csv")
DEFAULT_RAW_DIR = Path("raw/fifa")

@dataclass(frozen=True)
class FifaSnapshot:
    team: str
    rating_date: date
    fifa_rank: int
    fifa_points: float
    previous_points: float
    confederation: str
    source_team_name: str

def parse_fifa_snapshot_file(
    file_path: str | Path, 
    rating_date: date, 
    alias_path: str | Path = DEFAULT_ALIAS_PATH
) -> list[FifaSnapshot]:
    file_path = Path(file_path)
    if not file_path.exists():
        return []
    
    data = json.loads(file_path.read_text(encoding="utf-8"))
    aliases = load_source_aliases(alias_path) if Path(alias_path).exists() else []
    
    snapshots = []
    for entry in data.get("rankings", []):
        item = entry.get("rankingItem", {})
        source_name = item.get("name", "")
        if not source_name: continue
            
        canonical_name = resolve_team_name(source_name, "fifa", aliases)
        snapshots.append(FifaSnapshot(
            team=canonical_name,
            rating_date=rating_date,
            fifa_rank=item.get("rank", 0),
            fifa_points=item.get("totalPoints", 0.0),
            previous_points=entry.get("previousPoints", 0.0),
            confederation=entry.get("tag", {}).get("text", ""),
            source_team_name=source_name
        ))
    return snapshots

def build_fifa_snapshots_csv(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    dates_csv: str | Path = "raw/fifa_ranking_dates.csv",
    output_path: str | Path = DEFAULT_PROCESSED_PATH,
    alias_path: str | Path = DEFAULT_ALIAS_PATH
) -> Path:
    raw_path = Path(raw_dir)
    dates_path = Path(dates_csv)
    all_snapshots: list[FifaSnapshot] = []
    
    with dates_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_id = row["id"]
            rating_date = date.fromisoformat(row["date"])
            raw_file = raw_path / f"{date_id}.json"
            if raw_file.exists():
                all_snapshots.extend(parse_fifa_snapshot_file(raw_file, rating_date, alias_path))
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "team", "rating_date", "fifa_rank", "fifa_points", "previous_points", "confederation", "source_team_name"
        ])
        writer.writeheader()
        for s in all_snapshots:
            writer.writerow(asdict(s))
    return output_path

def join_fifa_to_matches(
    matches_path: str | Path = "processed/matches.csv",
    fifa_snapshots_path: str | Path = DEFAULT_PROCESSED_PATH,
    output_path: str | Path = "processed/matches_with_fifa.csv"
) -> Path:
    matches = pd.read_csv(matches_path, parse_dates=["match_date"])
    fifa = pd.read_csv(fifa_snapshots_path, parse_dates=["rating_date"])
    
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    fifa["rating_date"] = pd.to_datetime(fifa["rating_date"])
    
    def get_fifa_for_side(df, team_col, prefix):
        side_fifa = fifa[["team", "rating_date", "fifa_rank", "fifa_points"]].rename(columns={
            "team": team_col,
            "rating_date": f"{prefix}_fifa_date",
            "fifa_rank": f"{prefix}_fifa_rank",
            "fifa_points": f"{prefix}_fifa_points"
        })
        df = df.sort_values("match_date").reset_index(drop=True)
        side_fifa = side_fifa.sort_values(f"{prefix}_fifa_date").reset_index(drop=True)
        return pd.merge_asof(
            df, side_fifa, left_on="match_date", right_on=f"{prefix}_fifa_date", 
            by=team_col, direction="backward"
        )

    result = get_fifa_for_side(matches, "home_team", "home")
    result = get_fifa_for_side(result, "away_team", "away")
    result["fifa_rank_diff"] = result["home_fifa_rank"] - result["away_fifa_rank"]
    result["fifa_points_diff"] = result["home_fifa_points"] - result["away_fifa_points"]
    
    result = result.sort_values("match_date").reset_index(drop=True)
    result.to_csv(output_path, index=False)
    return Path(output_path)

def merge_ratings_to_matches(
    matches_with_elo_path: str | Path = "processed/matches_with_elo.csv",
    fifa_snapshots_path: str | Path = DEFAULT_PROCESSED_PATH,
    output_path: str | Path = "processed/matches_with_ratings.csv"
) -> Path:
    return join_fifa_to_matches(
        matches_path=matches_with_elo_path,
        fifa_snapshots_path=fifa_snapshots_path,
        output_path=output_path
    )
