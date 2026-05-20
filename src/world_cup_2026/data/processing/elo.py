from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from world_cup_2026.data.ingestion.kaggle import build_name_resolver, load_former_names_csv
from world_cup_2026.data.processing.aliases import TeamAlias, resolve_team_name, load_source_aliases


DEFAULT_RAW_DIR = Path("raw/elo")
DEFAULT_PROCESSED_PATH = Path("processed/team_elo_snapshots.csv")
DEFAULT_ALIAS_PATH = Path("config/team_source_aliases.csv")
DEFAULT_HISTORICAL_MATCH_PATH = Path("processed/team_elo_match_history.csv")
DEFAULT_MATCH_JOIN_PATH = Path("processed/matches_with_elo.csv")


@dataclass(frozen=True)
class EloSnapshot:
    team: str
    rating_date: date
    elo_rating: float
    elo_rank: int
    source_url: str
    retrieved_at: str
    source_team_name: str


@dataclass(frozen=True)
class EloMatchHistoryRecord:
    match_date: date
    team1: str
    team2: str
    score1: int
    score2: int
    tournament_code: str
    venue_code: str | None
    elo_change_team1: float
    elo_team1_pre: float
    elo_team1_post: float
    elo_team2_pre: float
    elo_team2_post: float
    rank_team1_post: int | None
    rank_team2_post: int | None
    source_page: str


def parse_elo_snapshot_file(
    html_path: str | Path,
    alias_path: str | Path = DEFAULT_ALIAS_PATH,
    rating_date: date | None = None,
) -> list[EloSnapshot]:
    html_file = Path(html_path)
    metadata = _load_metadata_for_html(html_file)
    rows = _load_snapshot_rows(html_file)
    aliases = load_source_aliases(alias_path) if Path(alias_path).exists() else []
    snapshot_date = rating_date or _infer_rating_date_from_path(html_file) or datetime.fromisoformat(
        metadata["retrieved_at"]
    ).date()

    snapshots: list[EloSnapshot] = []
    for row in rows:
        source_team_name = row["team"]
        team = resolve_team_name(source_team_name, "elo", aliases)
        snapshots.append(
            EloSnapshot(
                team=team,
                rating_date=snapshot_date,
                elo_rating=float(row["elo_rating"]),
                elo_rank=int(row["elo_rank"]),
                source_url=metadata["source_url"],
                retrieved_at=metadata["retrieved_at"],
                source_team_name=source_team_name,
            )
        )

    _validate_snapshots(snapshots)
    return snapshots


def parse_all_elo_payloads(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output_path: str | Path = DEFAULT_PROCESSED_PATH,
    alias_path: str | Path = DEFAULT_ALIAS_PATH,
) -> Path:
    snapshots: list[EloSnapshot] = []
    for html_path in sorted(Path(raw_dir).glob("*.html")):
        try:
            snapshots.extend(parse_elo_snapshot_file(html_path=html_path, alias_path=alias_path))
        except ValueError:
            continue
    return _write_elo_snapshots_csv(snapshots=snapshots, output_path=output_path)


def build_historical_elo_match_history(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    alias_path: str | Path = DEFAULT_ALIAS_PATH,
    former_names_path: str | Path = "data/former_names.csv",
    output_path: str | Path = DEFAULT_HISTORICAL_MATCH_PATH,
) -> Path:
    raw_path = Path(raw_dir)
    history_dir = raw_path / "history"
    team_dictionary_path = _latest_team_dictionary_path(raw_path)
    team_dictionary = _load_team_dictionary(team_dictionary_path)
    aliases = load_source_aliases(alias_path) if Path(alias_path).exists() else []
    former_name_resolver = build_name_resolver(load_former_names_csv(former_names_path))

    records: dict[tuple[date, str, str, int, int, str], EloMatchHistoryRecord] = {}
    for tsv_path in sorted(history_dir.glob("*.tsv")):
        page_name = tsv_path.stem
        for record in _parse_team_history_page(
            tsv_path=tsv_path,
            source_page=page_name,
            team_dictionary=team_dictionary,
            aliases=aliases,
            former_name_resolver=former_name_resolver,
        ):
            key = (
                record.match_date,
                record.team1,
                record.team2,
                record.score1,
                record.score2,
                record.tournament_code,
            )
            records.setdefault(key, record)

    return _write_historical_match_history(records.values(), output_path=output_path)


def join_historical_elo_to_matches(
    matches_path: str | Path = "processed/matches.csv",
    elo_history_path: str | Path = DEFAULT_HISTORICAL_MATCH_PATH,
    output_path: str | Path = DEFAULT_MATCH_JOIN_PATH,
) -> Path:
    matches = pd.read_csv(matches_path, parse_dates=["match_date"])
    elo_history = pd.read_csv(elo_history_path, parse_dates=["match_date"])

    completed_matches = matches[matches["is_future_fixture"] == False].copy()
    completed_matches["match_id"] = range(len(completed_matches))
    completed_matches["home_score_int"] = completed_matches["home_score"].astype(int)
    completed_matches["away_score_int"] = completed_matches["away_score"].astype(int)

    direct = completed_matches.merge(
        elo_history,
        left_on=["match_date", "home_team", "away_team", "home_score_int", "away_score_int"],
        right_on=["match_date", "team1", "team2", "score1", "score2"],
        how="left",
    )
    direct["join_orientation"] = "direct"

    swapped_history = elo_history.rename(
        columns={
            "team1": "away_team",
            "team2": "home_team",
            "score1": "away_score_int",
            "score2": "home_score_int",
            "elo_team1_pre": "elo_away_pre",
            "elo_team1_post": "elo_away_post",
            "elo_team2_pre": "elo_home_pre",
            "elo_team2_post": "elo_home_post",
            "rank_team1_post": "elo_rank_away_post",
            "rank_team2_post": "elo_rank_home_post",
        }
    )
    direct["elo_home_pre"] = direct["elo_team1_pre"]
    direct["elo_away_pre"] = direct["elo_team2_pre"]
    direct["elo_home_post"] = direct["elo_team1_post"]
    direct["elo_away_post"] = direct["elo_team2_post"]
    direct["elo_rank_home_post"] = direct["rank_team1_post"]
    direct["elo_rank_away_post"] = direct["rank_team2_post"]

    unmatched = direct[direct["elo_home_pre"].isna()][completed_matches.columns]
    swapped = unmatched.merge(
        swapped_history,
        on=["match_date", "home_team", "away_team", "home_score_int", "away_score_int"],
        how="left",
    )
    swapped["join_orientation"] = "swapped"

    combined = direct[~direct["elo_home_pre"].isna()].copy()
    if not swapped.empty:
        merged_swapped = completed_matches.merge(
            swapped[
                [
                    "match_id",
                    "elo_home_pre",
                    "elo_away_pre",
                    "elo_home_post",
                    "elo_away_post",
                    "elo_rank_home_post",
                    "elo_rank_away_post",
                    "source_page",
                    "join_orientation",
                ]
            ],
            on="match_id",
            how="right",
        )
        combined = pd.concat([combined, merged_swapped], ignore_index=True, sort=False)

    combined["elo_diff_pre"] = combined["elo_home_pre"] - combined["elo_away_pre"]
    combined["elo_diff_post"] = combined["elo_home_post"] - combined["elo_away_post"]
    
    # Final join back to full matches (including futures)
    combined = matches.merge(
        combined[
            [
                "match_date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "elo_home_pre",
                "elo_away_pre",
                "elo_home_post",
                "elo_away_post",
                "elo_rank_home_post",
                "elo_rank_away_post",
                "elo_diff_pre",
                "elo_diff_post",
                "join_orientation",
                "source_page",
            ]
        ],
        on=["match_date", "home_team", "away_team", "home_score", "away_score"],
        how="left",
    )
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    return output


def _write_elo_snapshots_csv(
    snapshots: Iterable[EloSnapshot],
    output_path: str | Path = DEFAULT_PROCESSED_PATH,
) -> Path:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "team",
                "rating_date",
                "elo_rating",
                "elo_rank",
                "source_url",
                "retrieved_at",
                "source_team_name",
            ],
        )
        writer.writeheader()
        for snapshot in sorted(snapshots, key=lambda item: (item.rating_date, item.elo_rank, item.team)):
            row = asdict(snapshot)
            row["rating_date"] = snapshot.rating_date.isoformat()
            writer.writerow(row)
    return target_path


def _load_metadata_for_html(html_path: Path) -> dict[str, str]:
    import json
    metadata_path = html_path.with_suffix(".json")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_snapshot_rows(html_file: Path) -> list[dict[str, str]]:
    tsv_path = html_file.with_suffix(".tsv")
    if tsv_path.exists():
        return _parse_world_tsv_snapshot(tsv_path=tsv_path, team_dictionary_path=html_file.with_suffix(".teams.tsv"))
    # Fallback for old HTML format if needed, but we prefer TSV
    raise ValueError(f"Missing TSV payload for {html_file}")


def _parse_world_tsv_snapshot(tsv_path: Path, team_dictionary_path: Path) -> list[dict[str, str]]:
    team_dictionary = _load_team_dictionary(team_dictionary_path)
    rows: list[dict[str, str]] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            fields = line.strip().split("\t")
            if len(fields) < 4: continue
            rank = fields[0].strip().replace(".", "")
            team_code = fields[2].strip()
            rating = fields[3].strip()
            if team_code in team_dictionary:
                rows.append({"elo_rank": rank, "team": team_dictionary[team_code], "elo_rating": rating})
    return rows


def _load_team_dictionary(team_dictionary_path: Path) -> dict[str, str]:
    dictionary: dict[str, str] = {}
    with team_dictionary_path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            fields = line.strip().split("\t")
            if len(fields) >= 2:
                dictionary[fields[0].strip()] = fields[1].strip()
    return dictionary


def _infer_rating_date_from_path(html_path: Path) -> date | None:
    match = re.search(r"(\d{8})T\d{6}Z", html_path.stem)
    if match:
        token = match.group(1)
        return date(int(token[0:4]), int(token[4:6]), int(token[6:8]))
    return None


def _validate_snapshots(snapshots: list[EloSnapshot]) -> None:
    if not snapshots: raise ValueError("No snapshots parsed")


def _latest_team_dictionary_path(raw_path: Path) -> Path:
    candidates = sorted(raw_path.glob("elo_*.teams.tsv"))
    return candidates[-1] if candidates else raw_path / "en.teams.tsv"


def _team_page_names_from_dictionary(team_dictionary_path: Path) -> dict[str, str]:
    page_names: dict[str, str] = {}
    with team_dictionary_path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            fields = line.strip().split("\t")
            if len(fields) >= 2:
                page_names[fields[0].strip()] = _page_name(fields[1].strip())
    return page_names


def _page_name(text: str) -> str:
    return text.replace(" ", "_").replace("à", "a").replace("á", "a").replace("ç", "c") # Simplified


def _parse_team_history_page(
    tsv_path: Path,
    source_page: str,
    team_dictionary: dict[str, str],
    aliases: list[TeamAlias],
    former_name_resolver,
) -> list[EloMatchHistoryRecord]:
    records: list[EloMatchHistoryRecord] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            fields = line.strip().split("\t")
            if len(fields) < 16: continue
            try:
                match_date = date(int(fields[0]), int(fields[1]), int(fields[2]))
                team1 = _normalize_historical_name(team_dictionary.get(fields[3], fields[3]), match_date, aliases, former_name_resolver)
                team2 = _normalize_historical_name(team_dictionary.get(fields[4], fields[4]), match_date, aliases, former_name_resolver)
                change = float(fields[9].replace("−", "-"))
                t1_post = float(fields[10].replace("−", "-"))
                t2_post = float(fields[11].replace("−", "-"))
                records.append(EloMatchHistoryRecord(
                    match_date=match_date, team1=team1, team2=team2, 
                    score1=int(fields[5]), score2=int(fields[6]),
                    tournament_code=fields[7], venue_code=fields[8],
                    elo_change_team1=change, elo_team1_pre=t1_post - change, elo_team1_post=t1_post,
                    elo_team2_pre=t2_post + change, elo_team2_post=t2_post,
                    rank_team1_post=int(fields[14]) if fields[14].isdigit() else None,
                    rank_team2_post=int(fields[15]) if fields[15].isdigit() else None,
                    source_page=source_page
                ))
            except (ValueError, IndexError): continue
    return records


def _normalize_historical_name(team_name: str, match_date: date, aliases: list[TeamAlias], former_name_resolver) -> str:
    alias_resolved = resolve_team_name(team_name, "elo", aliases)
    return former_name_resolver(alias_resolved, match_date)


def _write_historical_match_history(records: Iterable[EloMatchHistoryRecord], output_path: str | Path) -> Path:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "match_date", "team1", "team2", "score1", "score2", "tournament_code", "venue_code",
            "elo_change_team1", "elo_team1_pre", "elo_team1_post", "elo_team2_pre", "elo_team2_post",
            "rank_team1_post", "rank_team2_post", "source_page"
        ])
        writer.writeheader()
        for record in sorted(records, key=lambda x: (x.match_date, x.team1, x.team2)):
            row = asdict(record)
            row["match_date"] = record.match_date.isoformat()
            writer.writerow(row)
    return target_path
