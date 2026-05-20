from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Iterable


MatchKey = tuple[date, str, str]


@dataclass(frozen=True)
class FormerNamePeriod:
    current_name: str
    former_name: str
    start_date: date
    end_date: date

    def applies_to(self, team_name: str, match_date: date) -> bool:
        return self.former_name == team_name and self.start_date <= match_date <= self.end_date


@dataclass(frozen=True)
class ShootoutRecord:
    match_date: date
    home_team: str
    away_team: str
    winner: str
    first_shooter: str | None

    @property
    def match_key(self) -> MatchKey:
        return (self.match_date, self.home_team, self.away_team)


@dataclass(frozen=True)
class GoalscorerRecord:
    match_date: date
    home_team: str
    away_team: str
    team: str
    scorer: str
    minute: int | None
    own_goal: bool
    penalty: bool

    @property
    def match_key(self) -> MatchKey:
        return (self.match_date, self.home_team, self.away_team)


@dataclass(frozen=True)
class HistoricalMatchRecord:
    match_date: date
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    tournament: str
    city: str
    country: str
    neutral: bool
    shootout_winner: str | None = None
    first_shooter: str | None = None
    goalscorers: tuple[GoalscorerRecord, ...] = ()

    @property
    def match_key(self) -> MatchKey:
        return (self.match_date, self.home_team, self.away_team)

    @property
    def winner(self) -> str | None:
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return self.home_team
        if self.away_score > self.home_score:
            return self.away_team
        return None

    @property
    def result_label(self) -> str | None:
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return "home_win"
        if self.away_score > self.home_score:
            return "away_win"
        return "draw"


@dataclass(frozen=True)
class KaggleDataset:
    matches: tuple[HistoricalMatchRecord, ...]
    shootouts: tuple[ShootoutRecord, ...]
    goalscorers: tuple[GoalscorerRecord, ...]
    former_names: tuple[FormerNamePeriod, ...]


def load_kaggle_dataset(data_dir: str | Path, canonicalize_names: bool = True) -> KaggleDataset:
    base_dir = Path(data_dir)
    former_names = tuple(load_former_names_csv(base_dir / "former_names.csv"))
    name_resolver = build_name_resolver(former_names) if canonicalize_names else None

    shootouts = tuple(load_shootouts_csv(base_dir / "shootouts.csv", name_resolver=name_resolver))
    goalscorers = tuple(load_goalscorers_csv(base_dir / "goalscorers.csv", name_resolver=name_resolver))
    matches = tuple(
        load_results_csv(
            base_dir / "results.csv",
            shootouts=shootouts,
            goalscorers=goalscorers,
            name_resolver=name_resolver,
        )
    )
    return KaggleDataset(
        matches=matches,
        shootouts=shootouts,
        goalscorers=goalscorers,
        former_names=former_names,
    )


def load_results_csv(
    csv_path: str | Path,
    shootouts: Iterable[ShootoutRecord] = (),
    goalscorers: Iterable[GoalscorerRecord] = (),
    name_resolver: Callable[[str, date], str] | None = None,
) -> list[HistoricalMatchRecord]:
    shootout_lookup = {record.match_key: record for record in shootouts}
    goalscorer_lookup = _group_goalscorers_by_match(goalscorers)

    matches: list[HistoricalMatchRecord] = []
    for row in _read_csv_dicts(csv_path):
        match_date = _parse_date(row["date"])
        home_team = _normalize_team_name(row["home_team"], match_date, name_resolver)
        away_team = _normalize_team_name(row["away_team"], match_date, name_resolver)
        match_key = (match_date, home_team, away_team)
        shootout = shootout_lookup.get(match_key)
        match_goalscorers = tuple(goalscorer_lookup.get(match_key, ()))

        matches.append(
            HistoricalMatchRecord(
                match_date=match_date,
                home_team=home_team,
                away_team=away_team,
                home_score=_parse_optional_int(row["home_score"]),
                away_score=_parse_optional_int(row["away_score"]),
                tournament=row["tournament"],
                city=row["city"],
                country=row["country"],
                neutral=_parse_bool(row["neutral"]),
                shootout_winner=shootout.winner if shootout else None,
                first_shooter=shootout.first_shooter if shootout else None,
                goalscorers=match_goalscorers,
            )
        )

    return matches


def load_shootouts_csv(
    csv_path: str | Path,
    name_resolver: Callable[[str, date], str] | None = None,
) -> list[ShootoutRecord]:
    records: list[ShootoutRecord] = []
    for row in _read_csv_dicts(csv_path):
        match_date = _parse_date(row["date"])
        records.append(
            ShootoutRecord(
                match_date=match_date,
                home_team=_normalize_team_name(row["home_team"], match_date, name_resolver),
                away_team=_normalize_team_name(row["away_team"], match_date, name_resolver),
                winner=_normalize_team_name(row["winner"], match_date, name_resolver),
                first_shooter=_normalize_team_name(row["first_shooter"], match_date, name_resolver)
                if row["first_shooter"]
                else None,
            )
        )
    return records


def load_goalscorers_csv(
    csv_path: str | Path,
    name_resolver: Callable[[str, date], str] | None = None,
) -> list[GoalscorerRecord]:
    records: list[GoalscorerRecord] = []
    for row in _read_csv_dicts(csv_path):
        match_date = _parse_date(row["date"])
        records.append(
            GoalscorerRecord(
                match_date=match_date,
                home_team=_normalize_team_name(row["home_team"], match_date, name_resolver),
                away_team=_normalize_team_name(row["away_team"], match_date, name_resolver),
                team=_normalize_team_name(row["team"], match_date, name_resolver),
                scorer=row["scorer"],
                minute=_parse_optional_int(row["minute"]),
                own_goal=_parse_bool(row["own_goal"]),
                penalty=_parse_bool(row["penalty"]),
            )
        )
    return records


def load_former_names_csv(csv_path: str | Path) -> list[FormerNamePeriod]:
    return [
        FormerNamePeriod(
            current_name=row["current"],
            former_name=row["former"],
            start_date=_parse_date(row["start_date"]),
            end_date=_parse_date(row["end_date"]),
        )
        for row in _read_csv_dicts(csv_path)
    ]


def build_name_resolver(former_names: Iterable[FormerNamePeriod]) -> Callable[[str, date], str]:
    periods = tuple(former_names)

    def resolve(team_name: str, match_date: date) -> str:
        for period in periods:
            if period.applies_to(team_name, match_date):
                return period.current_name
        return team_name

    return resolve


def _group_goalscorers_by_match(goalscorers: Iterable[GoalscorerRecord]) -> dict[MatchKey, list[GoalscorerRecord]]:
    grouped: dict[MatchKey, list[GoalscorerRecord]] = {}
    for record in goalscorers:
        grouped.setdefault(record.match_key, []).append(record)
    return grouped


def _normalize_team_name(
    team_name: str,
    match_date: date,
    name_resolver: Callable[[str, date], str] | None,
) -> str:
    if not team_name or name_resolver is None:
        return team_name
    return name_resolver(team_name, match_date)


def _read_csv_dicts(csv_path: str | Path) -> Iterable[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_bool(value: str) -> bool:
    return value.strip().upper() == "TRUE"


def _parse_optional_int(value: str) -> int | None:
    stripped = value.strip()
    if not stripped or stripped.upper() == "NA":
        return None
    return int(stripped)
