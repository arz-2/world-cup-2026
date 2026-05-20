from world_cup_2026.models.poisson import PoissonMatchModel


def test_poisson_result_probabilities_sum_to_one() -> None:
    model = PoissonMatchModel()

    prediction = model.predict_result(lambda_a=1.8, lambda_b=1.1)
    total = prediction.team_a_win + prediction.draw + prediction.team_b_win

    assert round(total, 6) == 1.0
