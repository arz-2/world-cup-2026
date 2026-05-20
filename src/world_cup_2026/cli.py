from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_cup_2026.data.ingestion.elo import (
    fetch_current_elo_snapshot,
    fetch_all_team_history_pages,
)
from world_cup_2026.data.ingestion.fifa import fetch_fifa_ranking
from world_cup_2026.data.processing.elo import (
    parse_all_elo_payloads,
    build_historical_elo_match_history,
    join_historical_elo_to_matches,
)
from world_cup_2026.data.processing.fifa import build_fifa_snapshots_csv, merge_ratings_to_matches
from world_cup_2026.data.geo import build_geo_features
from world_cup_2026.models.xgboost.trainer import run_baseline_experiment
from world_cup_2026.models.poisson_lambda import run_poisson_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="FIFA World Cup 2026 Prediction Toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Data Command
    data_parser = subparsers.add_parser("data")
    data_subparsers = data_parser.add_subparsers(dest="subcommand", required=True)

    # Data Fetch
    fetch_parser = data_subparsers.add_parser("fetch")
    fetch_parser.add_argument("--source", choices=["elo", "fifa"], required=True)
    fetch_parser.add_argument("--limit", type=int, default=None)

    # Data Process
    process_parser = data_subparsers.add_parser("process")
    process_parser.add_argument(
        "--step", choices=["elo-history", "fifa-parse", "merge-ratings", "geo"], required=True
    )

    # Train Command
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", choices=["xgboost", "poisson"], default="xgboost")
    train_parser.add_argument("--include-elo", action="store_true")
    train_parser.add_argument("--include-fifa", action="store_true")

    args = parser.parse_args()

    if args.command == "data":
        if args.subcommand == "fetch":
            if args.source == "elo":
                fetch_current_elo_snapshot()
                fetch_all_team_history_pages()
            elif args.source == "fifa":
                import csv

                dates_path = Path("raw/fifa_ranking_dates.csv")
                with dates_path.open("r") as f:
                    dates = list(csv.DictReader(f))
                if args.limit:
                    dates = dates[: args.limit]
                for row in dates:
                    fetch_fifa_ranking(row["id"])

        elif args.subcommand == "process":
            if args.step == "elo-history":
                build_historical_elo_match_history()
                join_historical_elo_to_matches()
            elif args.step == "fifa-parse":
                build_fifa_snapshots_csv()
            elif args.step == "merge-ratings":
                merge_ratings_to_matches()
            elif args.step == "geo":
                build_geo_features(
                    "processed/matches_with_ratings.csv", "processed/matches_with_geo.csv"
                )

    elif args.command == "train":
        if args.model == "xgboost":
            summary = run_baseline_experiment(
                include_elo=args.include_elo, include_fifa=args.include_fifa
            )
            print(json.dumps(summary, indent=2))
        elif args.model == "poisson":
            summary = run_poisson_experiment()
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
