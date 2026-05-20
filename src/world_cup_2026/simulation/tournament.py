from __future__ import annotations

import random
from dataclasses import dataclass

from world_cup_2026.core.schemas import MatchPrediction


@dataclass(frozen=True)
class TournamentConfig:
    n_simulations: int = 10_000


def simulate_match(prediction: MatchPrediction) -> str:
    roll = random.random()
    if roll < prediction.team_a_win:
        return "team_a"
    if roll < prediction.team_a_win + prediction.draw:
        return "draw"
    return "team_b"
