import csv
from pathlib import Path

from world_cup_2026.data.processing.etl import build_processed_kaggle_artifacts


def test_build_processed_kaggle_artifacts(tmp_path: Path) -> None:
    artifacts = build_processed_kaggle_artifacts(data_dir="data", output_dir=tmp_path)

    assert artifacts.matches_csv.exists()
    assert artifacts.events_csv.exists()
    assert artifacts.mapping_csv.exists()

    with artifacts.matches_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) > 40_000
    assert rows[0]["home_team"] == "Scotland"
    assert rows[0]["result_label"] == "draw"

    future_fixture = next(row for row in rows if row["match_date"] == "2026-06-11" and row["home_team"] == "Mexico")
    assert future_fixture["is_future_fixture"] in ("true", "True")
    assert future_fixture["home_score"] == ""
