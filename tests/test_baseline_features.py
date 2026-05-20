from world_cup_2026.features.registry import build_baseline_training_frame


def test_build_baseline_training_frame() -> None:
    artifacts = build_baseline_training_frame("processed")

    assert len(artifacts.feature_frame) > 40_000
    assert "result_code" in artifacts.feature_frame.columns
    assert "home_points_last_5" in artifacts.feature_frame.columns
    assert "away_points_last_10" in artifacts.feature_frame.columns
    assert "tournament_group" in artifacts.feature_frame.columns
