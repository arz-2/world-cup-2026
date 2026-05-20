from datetime import date
from pathlib import Path

from world_cup_2026.data.processing.elo import parse_elo_snapshot_file
from world_cup_2026.data.processing.aliases import resolve_team_name, write_default_aliases


def test_resolve_team_name_from_default_aliases(tmp_path: Path) -> None:
    alias_path = write_default_aliases(tmp_path / "team_source_aliases.csv")
    from world_cup_2026.data.processing.aliases import load_source_aliases

    aliases = load_source_aliases(alias_path)
    assert resolve_team_name("USA", "elo", aliases) == "United States"
    assert resolve_team_name("Korea Republic", "fifa", aliases) == "South Korea"


def test_parse_elo_snapshot_file_from_tsv(tmp_path: Path) -> None:
    html_path = tmp_path / "elo_20260516T120000Z.html"
    metadata_path = tmp_path / "elo_20260516T120000Z.json"
    tsv_path = tmp_path / "elo_20260516T120000Z.tsv"
    teams_path = tmp_path / "elo_20260516T120000Z.teams.tsv"
    alias_path = write_default_aliases(tmp_path / "team_source_aliases.csv")

    html_path.write_text("<html></html>", encoding="utf-8")
    metadata_path.write_text(
        """
        {
          "source_url": "https://eloratings.net/",
          "retrieved_at": "2026-05-16T12:00:00+00:00",
          "content_hash": "dummy",
          "parser_version": 1
        }
        """,
        encoding="utf-8",
    )
    tsv_path.write_text("1\t1\tAR\t2143\n2\t2\tES\t2101\n12\t12\tUS\t1844\n", encoding="utf-8")
    teams_path.write_text("AR\tArgentina\nES\tSpain\nUS\tUSA\n", encoding="utf-8")

    snapshots = parse_elo_snapshot_file(html_path=html_path, alias_path=alias_path)
    assert len(snapshots) == 3
    assert snapshots[0].rating_date == date(2026, 5, 16)
    assert snapshots[2].team == "United States"
    assert snapshots[2].elo_rank == 12


def test_parse_elo_snapshot_file_prefers_tsv_payload(tmp_path: Path) -> None:
    html_path = tmp_path / "elo_20260516T120000Z.html"
    metadata_path = tmp_path / "elo_20260516T120000Z.json"
    tsv_path = tmp_path / "elo_20260516T120000Z.tsv"
    teams_path = tmp_path / "elo_20260516T120000Z.teams.tsv"
    alias_path = write_default_aliases(tmp_path / "team_source_aliases.csv")

    html_path.write_text("<html></html>", encoding="utf-8")
    metadata_path.write_text(
        """
        {
          "source_url": "https://eloratings.net/",
          "retrieved_at": "2026-05-16T12:00:00+00:00",
          "content_hash": "dummy",
          "parser_version": 1
        }
        """,
        encoding="utf-8",
    )
    tsv_path.write_text("1\t1\tAR\t2143\n41\t41\tUS\t1721\n", encoding="utf-8")
    teams_path.write_text("AR\tArgentina\nUS\tUSA\n", encoding="utf-8")

    snapshots = parse_elo_snapshot_file(html_path=html_path, alias_path=alias_path)
    assert len(snapshots) == 2
    assert snapshots[1].team == "United States"


def test_page_name_style_matches_elo_site() -> None:
    from world_cup_2026.data.processing.elo import _page_name

    assert _page_name("United States") == "United_States"
    assert _page_name("Curaçao") == "Curacao"
