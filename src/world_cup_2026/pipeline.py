from __future__ import annotations

from world_cup_2026.models.elo import EloMatchPredictor
from world_cup_2026.models.poisson import PoissonMatchModel
from world_cup_2026.core.schemas import MatchContext


def build_baseline_prediction(context: MatchContext, lambda_a: float | None = None, lambda_b: float | None = None) -> dict[str, dict[str, float]]:
    elo_prediction = EloMatchPredictor().predict(context).as_dict()

    result = {"elo": elo_prediction}
    if lambda_a is not None and lambda_b is not None:
        poisson_prediction = PoissonMatchModel().predict_result(lambda_a=lambda_a, lambda_b=lambda_b).as_dict()
        result["poisson"] = poisson_prediction

    return result
