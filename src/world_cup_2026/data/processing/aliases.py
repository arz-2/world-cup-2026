from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import csv


@dataclass(frozen=True)
class TeamAlias:
    canonical_name: str
    source_name: str
    source_system: str


DEFAULT_SOURCE_ALIASES: tuple[TeamAlias, ...] = (
    TeamAlias(canonical_name="United States", source_name="USA", source_system="elo"),
    TeamAlias(canonical_name="United States", source_name="United States", source_system="elo"),
    TeamAlias(canonical_name="South Korea", source_name="Korea Republic", source_system="fifa"),
    TeamAlias(canonical_name="South Korea", source_name="Korea Rep", source_system="elo"),
    TeamAlias(canonical_name="Ivory Coast", source_name="Côte d'Ivoire", source_system="fifa"),
    TeamAlias(canonical_name="Ivory Coast", source_name="Cote d'Ivoire", source_system="elo"),
    TeamAlias(canonical_name="North Macedonia", source_name="Macedonia", source_system="elo"),
    TeamAlias(canonical_name="Curaçao", source_name="Curacao", source_system="elo"),
    TeamAlias(canonical_name="Czech Republic", source_name="Czechia", source_system="fifa"),
)


def load_source_aliases(csv_path: str | Path) -> list[TeamAlias]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        return [
            TeamAlias(
                canonical_name=row["canonical_name"],
                source_name=row["source_name"],
                source_system=row["source_system"],
            )
            for row in csv.DictReader(handle)
        ]


def resolve_team_name(source_name: str, source_system: str, alias_rows: list[TeamAlias]) -> str:
    normalized_source = _normalize_text(source_name)
    normalized_system = source_system.strip().lower()

    for alias in alias_rows:
        if alias.source_system.strip().lower() != normalized_system:
            continue
        if _normalize_text(alias.source_name) == normalized_source:
            return alias.canonical_name

    return source_name


def write_default_aliases(csv_path: str | Path) -> Path:
    target_path = Path(csv_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["canonical_name", "source_name", "source_system"])
        writer.writeheader()
        for alias in DEFAULT_SOURCE_ALIASES:
            writer.writerow(
                {
                    "canonical_name": alias.canonical_name,
                    "source_name": alias.source_name,
                    "source_system": alias.source_system,
                }
            )
    return target_path


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())

