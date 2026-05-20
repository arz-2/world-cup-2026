from __future__ import annotations

from dataclasses import dataclass
from math import exp

from world_cup_2026.core.schemas import MatchContext, MatchPrediction


@dataclass
class EloMatchPredictor:
    draw_bias: float = 0.24
    home_field_elo: float = 55.0
    host_bonus_elo: float = 35.0
    form_weight: float = 25.0

    def predict(self, context: MatchContext) -> MatchPrediction:
        # Note: MatchContext currently lacks some fields needed for this legacy logic
        # We will need to update MatchContext or this logic in Phase 3
        return MatchPrediction(team_a_win=0.33, draw=0.34, team_b_win=0.33)

    def _logistic(self, value: float) -> float:
        return 1.0 / (1.0 + exp(-value))
