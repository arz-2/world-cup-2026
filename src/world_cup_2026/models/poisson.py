from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, factorial

import numpy as np

from world_cup_2026.core.schemas import MatchPrediction


@dataclass
class PoissonMatchModel:
    # Dixon-Coles correction parameter. 0.0 = no correction (plain Poisson).
    # Fit via fit_rho() in poisson_lambda.py; typically a small negative value.
    rho: float = 0.0

    def predict_result(self, lambda_a: float, lambda_b: float, max_goals: int = 10) -> MatchPrediction:
        prob_a_win = 0.0
        prob_draw = 0.0
        prob_b_win = 0.0

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p_i = self._poisson_prob(lambda_a, i)
                p_j = self._poisson_prob(lambda_b, j)
                p_score = p_i * p_j * self._dc_tau(i, j, lambda_a, lambda_b)

                if i > j:
                    prob_a_win += p_score
                elif i < j:
                    prob_b_win += p_score
                else:
                    prob_draw += p_score

        # Normalize to ensure sum is 1.0 (max_goals truncation + DC correction)
        total = prob_a_win + prob_draw + prob_b_win
        return MatchPrediction(
            team_a_win=prob_a_win / total,
            draw=prob_draw / total,
            team_b_win=prob_b_win / total,
        )

    def _dc_tau(self, i: int, j: int, lambda_a: float, lambda_b: float) -> float:
        """Dixon-Coles correction factor for low-scoring scorelines."""
        if i == 0 and j == 0:
            return max(1 - lambda_a * lambda_b * self.rho, 1e-10)
        if i == 1 and j == 0:
            return max(1 + lambda_b * self.rho, 1e-10)
        if i == 0 and j == 1:
            return max(1 + lambda_a * self.rho, 1e-10)
        if i == 1 and j == 1:
            return max(1 - self.rho, 1e-10)
        return 1.0

    @staticmethod
    def _poisson_prob(lmbda: float, k: int) -> float:
        return (exp(-lmbda) * (lmbda**k)) / factorial(k)


def fit_rho(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    n_candidates: int = 500,
) -> float:
    """Find the rho that maximises the DC log-likelihood on training scorelines.

    Only the four low-scoring cells (0-0, 1-0, 0-1, 1-1) contribute non-trivially;
    all other scorelines have tau=1 so their log(tau)=0 and we can skip them.
    """
    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_11 = (home_goals == 1) & (away_goals == 1)

    best_rho, best_ll = 0.0, -np.inf
    for rho in np.linspace(-0.99, 0.0, n_candidates):
        log_tau = np.zeros(len(home_goals))
        log_tau[mask_00] = np.log(np.maximum(1 - lambda_home[mask_00] * lambda_away[mask_00] * rho, 1e-10))
        log_tau[mask_10] = np.log(np.maximum(1 + lambda_away[mask_10] * rho, 1e-10))
        log_tau[mask_01] = np.log(np.maximum(1 + lambda_home[mask_01] * rho, 1e-10))
        log_tau[mask_11] = np.log(np.maximum(1 - rho, 1e-10))
        ll = float(log_tau.sum())
        if ll > best_ll:
            best_ll, best_rho = ll, rho

    return best_rho
