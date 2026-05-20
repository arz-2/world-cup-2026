from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from world_cup_2026.data.ingestion.kaggle import HistoricalMatchRecord, KaggleDataset, load_kaggle_dataset


@dataclass(frozen=True)
class ProcessedKaggleArtifacts:
    matches_csv: Path
    mapping_csv: Path
    events_csv: Path


def build_processed_kaggle_artifacts(
    data_dir: str | Path,
    output_dir: str | Path,
) -> ProcessedKaggleArtifacts:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = load_kaggle_dataset(data_dir)

    matches_file = output_path / "matches.csv"
    mapping_file = output_path / "team_name_mapping.csv"
    events_file = output_path / "match_events_summary.csv"

    _write_matches_csv(dataset.matches, matches_file)
    _write_mapping_csv(dataset, mapping_file)
    _write_events_summary_csv(dataset.matches, events_file)

    return ProcessedKaggleArtifacts(
        matches_csv=matches_file,
        mapping_csv=mapping_file,
        events_csv=events_file,
    )


def _write_matches_csv(matches: Iterable[HistoricalMatchRecord], target: Path) -> None:
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "match_date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "winner",
                "result_label",
                "tournament",
                "city",
                "country",
                "neutral",
                "went_to_shootout",
                "shootout_winner",
                "first_shooter",
                "goalscorer_count",
                "home_goalscorers",
                "away_goalscorers",
                "home_own_goals_for",
                "away_own_goals_for",
                "home_penalty_goals",
                "away_penalty_goals",
                "is_future_fixture",
            ],
        )
        writer.writeheader()
        for match in sorted(matches, key=lambda m: m.match_date):
            is_future = match.home_score is None or match.away_score is None
            
            home_goals = [g for g in match.goalscorers if g.team == match.home_team]
            away_goals = [g for g in match.goalscorers if g.team == match.away_team]
            
            writer.writerow(
                {
                    "match_date": match.match_date.isoformat(),
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "home_score": _stringify_optional(match.home_score),
                    "away_score": _stringify_optional(match.away_score),
                    "winner": match.winner or "",
                    "result_label": match.result_label or "",
                    "tournament": match.tournament,
                    "city": match.city,
                    "country": match.country,
                    "neutral": match.neutral,
                    "went_to_shootout": match.shootout_winner is not None,
                    "shootout_winner": match.shootout_winner or "",
                    "first_shooter": match.first_shooter or "",
                    "goalscorer_count": len(match.goalscorers),
                    "home_goalscorers": len(home_goals),
                    "away_goalscorers": len(away_goals),
                    "home_own_goals_for": len([g for g in home_goals if g.own_goal]),
                    "away_own_goals_for": len([g for g in away_goals if g.own_goal]),
                    "home_penalty_goals": len([g for g in home_goals if g.penalty]),
                    "away_penalty_goals": len([g for g in away_goals if g.penalty]),
                    "is_future_fixture": is_future,
                }
            )


def _write_mapping_csv(dataset: KaggleDataset, target: Path) -> None:
    teams = set()
    for match in dataset.matches:
        teams.add(match.home_team)
        teams.add(match.away_team)

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["team_name"])
        writer.writeheader()
        for team in sorted(teams):
            writer.writerow({"team_name": team})


def _write_events_summary_csv(matches: Iterable[HistoricalMatchRecord], target: Path) -> None:
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "match_date",
                "home_team",
                "away_team",
                "total_goals",
                "had_shootout",
                "had_penalty_goal",
                "had_own_goal",
            ],
        )
        writer.writeheader()
        for match in matches:
            if match.home_score is None: continue
            
            home_penalty_goals = len([g for g in match.goalscorers if g.team == match.home_team and g.penalty])
            away_penalty_goals = len([g for g in match.goalscorers if g.team == match.away_team and g.penalty])
            home_own_goals_for = len([g for g in match.goalscorers if g.team == match.home_team and g.own_goal])
            away_own_goals_for = len([g for g in match.goalscorers if g.team == match.away_team and g.own_goal])

            writer.writerow({
                "match_date": match.match_date.isoformat(),
                "home_team": match.home_team,
                "away_team": match.away_team,
                "total_goals": int(match.home_score) + int(match.away_score),
                "had_shootout": match.shootout_winner is not None,
                "had_penalty_goal": home_penalty_goals > 0 or away_penalty_goals > 0,
                "had_own_goal": home_own_goals_for > 0 or away_own_goals_for > 0,
            })


def _stringify_optional(value: int | None) -> str:
    return "" if value is None else str(value)
