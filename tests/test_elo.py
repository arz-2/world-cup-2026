from datetime import date

from world_cup_2026.models.elo import EloMatchPredictor
from world_cup_2026.core.schemas import MatchContext


def test_elo_prediction_probabilities_sum_to_one() -> None:
    predictor = EloMatchPredictor()
    context = MatchContext(
        match_date=date(2026, 6, 11),
        home_team="Brazil",
        away_team="Serbia",
        tournament="World Cup",
        city="Los Angeles",
        country="United States",
        neutral=True,
    )

    prediction = predictor.predict(context)
    total = prediction.team_a_win + prediction.draw + prediction.team_b_win

    assert round(total, 6) == 1.0

