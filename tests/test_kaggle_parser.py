from datetime import date

from world_cup_2026.data.ingestion.kaggle import build_name_resolver, load_kaggle_dataset


def test_load_kaggle_dataset_from_repo_data() -> None:
    dataset = load_kaggle_dataset("data")

    assert len(dataset.matches) > 40_000
    assert len(dataset.shootouts) > 100
    assert len(dataset.goalscorers) > 1_000
    assert dataset.matches[0].match_date == date(1872, 11, 30)
    assert dataset.matches[0].home_team == "Scotland"


def test_former_name_resolver_maps_historical_names() -> None:
    dataset = load_kaggle_dataset("data", canonicalize_names=False)
    resolver = build_name_resolver(dataset.former_names)

    assert resolver("Dahomey", date(1965, 1, 1)) == "Benin"
    assert resolver("Dahomey", date(1980, 1, 1)) == "Dahomey"
