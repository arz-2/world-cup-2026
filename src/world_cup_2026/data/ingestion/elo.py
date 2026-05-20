from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_ELO_URL = "https://eloratings.net/"
DEFAULT_ELO_TABLE_URL = "https://eloratings.net/World.tsv"
DEFAULT_ELO_TEAM_DICTIONARY_URL = "https://eloratings.net/en.teams.tsv"
DEFAULT_RAW_DIR = Path("raw/elo")

@dataclass(frozen=True)
class RawEloPayload:
    html_path: Path
    metadata_path: Path
    table_path: Path | None = None
    team_dictionary_path: Path | None = None

def fetch_current_elo_snapshot(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    source_url: str = DEFAULT_ELO_URL,
    user_agent: str = "world-cup-2026/0.1",
) -> RawEloPayload:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    request = Request(source_url, headers={"User-Agent": user_agent})
    with urlopen(request) as response:
        html_bytes = response.read()

    table_request = Request(DEFAULT_ELO_TABLE_URL, headers={"User-Agent": user_agent})
    with urlopen(table_request) as response:
        table_bytes = response.read()

    team_request = Request(DEFAULT_ELO_TEAM_DICTIONARY_URL, headers={"User-Agent": user_agent})
    with urlopen(team_request) as response:
        team_bytes = response.read()

    retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    html_path = raw_path / f"elo_{timestamp}.html"
    metadata_path = raw_path / f"elo_{timestamp}.json"
    table_path = raw_path / f"elo_{timestamp}.tsv"
    team_dictionary_path = raw_path / f"elo_{timestamp}.teams.tsv"

    html_path.write_bytes(html_bytes)
    table_path.write_bytes(table_bytes)
    team_dictionary_path.write_bytes(team_bytes)
    metadata_path.write_text(
        json.dumps(
            {
                "source_url": source_url,
                "table_url": DEFAULT_ELO_TABLE_URL,
                "team_dictionary_url": DEFAULT_ELO_TEAM_DICTIONARY_URL,
                "retrieved_at": retrieved_at,
                "content_hash": hashlib.sha256(html_bytes).hexdigest(),
                "table_hash": hashlib.sha256(table_bytes).hexdigest(),
                "team_dictionary_hash": hashlib.sha256(team_bytes).hexdigest(),
                "parser_version": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return RawEloPayload(
        html_path=html_path,
        metadata_path=metadata_path,
        table_path=table_path,
        team_dictionary_path=team_dictionary_path,
    )

def fetch_all_team_history_pages(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    team_dictionary_path: str | Path | None = None,
    user_agent: str = "world-cup-2026/0.1",
) -> list[Path]:
    from world_cup_2026.data.processing.elo import _latest_team_dictionary_path, _team_page_names_from_dictionary
    
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    dictionary_path = Path(team_dictionary_path) if team_dictionary_path is not None else _latest_team_dictionary_path(raw_path)
    page_names = sorted(set(_team_page_names_from_dictionary(dictionary_path).values()))

    fetched_paths: list[Path] = []
    for page_name in page_names:
        url = f"https://eloratings.net/{page_name}.tsv"
        request = Request(url, headers={"User-Agent": user_agent})
        try:
            with urlopen(request) as response:
                payload = response.read()
        except Exception:
            continue
        target_path = raw_path / "history" / f"{page_name}.tsv"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(payload)
        fetched_paths.append(target_path)
    return fetched_paths
