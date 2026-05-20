from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class MatchContext:
    match_date: date
    home_team: str
    away_team: str
    tournament: str
    city: str
    country: str
    neutral: bool


@dataclass(frozen=True)
class MatchPrediction:
    team_a_win: float
    draw: float
    team_b_win: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineFeatureArtifacts:
    feature_frame: pd.DataFrame
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
