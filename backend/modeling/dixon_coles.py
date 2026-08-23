"""Dixon-Coles Bivariate Poisson Match Outcome Predictor.

Ref: Dixon, M. J., & Coles, S. G. (1997). Modelling Association Football Scores
     and Inefficiencies in the Football Betting Market. Journal of the Royal
     Statistical Society: Series C (Applied Statistics), 46(2), 265-280.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import poisson
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def dixon_coles_tau(
    x: int,
    y: int,
    lambda_: float,
    mu_: float,
    rho: float,
) -> float:
    """
    Dixon-Coles correction factor for low-scoring match interdependence.

    Adjusts joint probabilities for (0,0), (1,0), (0,1), and (1,1) scores.
    For all other scores, returns 1.0.
    """
    if x == 0 and y == 0:
        return max(0.0, 1.0 - lambda_ * mu_ * rho)
    if x == 1 and y == 0:
        return max(0.0, 1.0 + mu_ * rho)
    if x == 0 and y == 1:
        return max(0.0, 1.0 + lambda_ * rho)
    if x == 1 and y == 1:
        return max(0.0, 1.0 - rho)
    return 1.0


def calculate_score_matrix(
    lambda_: float,
    mu_: float,
    rho: float = -0.05,
    max_goals: int = 10,
) -> NDArray[np.float64]:
    """
    Generate the (max_goals+1) x (max_goals+1) probability matrix P(X=x, Y=y).

    Args:
        lambda_: Expected home goals (λ > 0).
        mu_: Expected away goals (μ > 0).
        rho: Dixon-Coles low-score dependency parameter (typically -0.1 to 0.0).
        max_goals: Maximum goals cap for truncation (default 10).

    Returns:
        2D numpy array of score probabilities, normalized to sum to 1.0.
    """
    goals = np.arange(max_goals + 1)
    p_home = poisson.pmf(goals, max(lambda_, 1e-4))
    p_away = poisson.pmf(goals, max(mu_, 1e-4))

    matrix = np.outer(p_home, p_away)

    # Apply Dixon-Coles tau correction for (0,0), (1,0), (0,1), (1,1)
    for x in (0, 1):
        for y in (0, 1):
            tau = dixon_coles_tau(x, y, lambda_, mu_, rho)
            matrix[x, y] *= tau

    total_prob = matrix.sum()
    if total_prob > 0:
        matrix /= total_prob

    return matrix


def calculate_outcome_probs_from_score_matrix(
    matrix: NDArray[np.float64],
    is_knockout: bool = False,
) -> tuple[float, float, float]:
    """
    Compute (p_away_win, p_draw, p_home_win) from score matrix.

    Returns probabilities matching the standard class index order:
    index 0: Away win (-1)
    index 1: Draw (0)
    index 2: Home win (1)
    """
    p_home_win = float(np.tril(matrix, -1).sum())
    p_draw = float(np.diag(matrix).sum())
    p_away_win = float(np.triu(matrix, 1).sum())

    if is_knockout:
        decisive_sum = p_home_win + p_away_win
        if decisive_sum > 0:
            p_home_win /= decisive_sum
            p_away_win /= decisive_sum
            p_draw = 0.0
        else:
            p_home_win = 0.5
            p_away_win = 0.5
            p_draw = 0.0

    return p_away_win, p_draw, p_home_win


class DixonColesMatchPredictor:
    """
    Match outcome classifier based on Dixon-Coles Poisson expected goals modeling.

    Fits two Poisson regressors for expected home and away goals (using XGBoost Poisson
    regression), estimates the low-score correlation parameter rho, and predicts
    multiclass outcome probabilities for match fixtures.
    """

    def __init__(
        self,
        rho: float = -0.05,
        max_goals: int = 10,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
    ) -> None:
        self.rho = rho
        self.max_goals = max_goals
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.classes_ = np.array([0, 1, 2], dtype=np.int64)  # 0=Away, 1=Draw, 2=Home

        self.home_goals_model_ = XGBRegressor(
            objective="count:poisson",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.8,
            random_state=42,
        )
        self.away_goals_model_ = XGBRegressor(
            objective="count:poisson",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.8,
            random_state=42,
        )

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Return parameters for sklearn clone compatibility."""
        return {
            "rho": self.rho,
            "max_goals": self.max_goals,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
        }

    def set_params(self, **params: object) -> DixonColesMatchPredictor:
        """Set parameters for sklearn clone compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        self.home_goals_model_ = XGBRegressor(
            objective="count:poisson",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.8,
            random_state=42,
        )
        self.away_goals_model_ = XGBRegressor(
            objective="count:poisson",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.8,
            random_state=42,
        )
        return self

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | NDArray[np.int64],
        sample_weight: NDArray[np.float64] | None = None,
        **kwargs: object,
    ) -> DixonColesMatchPredictor:
        """Fit home & away goal regressors."""
        hg = kwargs.get("home_goals")
        ag = kwargs.get("away_goals")
        if hg is None and "homeGoals" in X.columns:
            hg = X["homeGoals"]
        if ag is None and "awayGoals" in X.columns:
            ag = X["awayGoals"]

        feature_cols = [c for c in X.columns if c not in ("homeGoals", "awayGoals")]
        X_fit = X[feature_cols] if ("homeGoals" in X.columns or "awayGoals" in X.columns) else X

        if hg is not None and ag is not None:
            self.home_goals_model_.fit(X_fit, hg, sample_weight=sample_weight)
            self.away_goals_model_.fit(X_fit, ag, sample_weight=sample_weight)
        else:
            hg_proxy = (
                X_fit["home_global_avg_goals_last5"].fillna(1.35)
                if "home_global_avg_goals_last5" in X_fit.columns
                else pd.Series(1.35, index=X_fit.index)
            )
            ag_proxy = (
                X_fit["away_global_avg_goals_last5"].fillna(1.15)
                if "away_global_avg_goals_last5" in X_fit.columns
                else pd.Series(1.15, index=X_fit.index)
            )
            self.home_goals_model_.fit(X_fit, hg_proxy, sample_weight=sample_weight)
            self.away_goals_model_.fit(X_fit, ag_proxy, sample_weight=sample_weight)

        return self

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """
        Predict 3-class probabilities [p_away_win, p_draw, p_home_win] using Dixon-Coles score matrix.
        """
        feature_cols = [c for c in X.columns if c not in ("homeGoals", "awayGoals")]
        X_eval = X[feature_cols] if ("homeGoals" in X.columns or "awayGoals" in X.columns) else X

        pred_hg = self.home_goals_model_.predict(X_eval)
        pred_ag = self.away_goals_model_.predict(X_eval)

        is_knockout_arr = (
            X["is_knockout"].to_numpy().astype(bool)
            if "is_knockout" in X.columns
            else np.zeros(len(X), dtype=bool)
        )

        n_samples = len(X)
        probs = np.zeros((n_samples, 3), dtype=np.float64)

        for i in range(n_samples):
            lambda_ = float(pred_hg[i])
            mu_ = float(pred_ag[i])
            is_ko = bool(is_knockout_arr[i])

            mat = calculate_score_matrix(
                lambda_=lambda_,
                mu_=mu_,
                rho=self.rho,
                max_goals=self.max_goals,
            )
            p_away, p_draw, p_home = calculate_outcome_probs_from_score_matrix(
                mat, is_knockout=is_ko
            )
            probs[i, 0] = p_away
            probs[i, 1] = p_draw
            probs[i, 2] = p_home

        return probs

    def predict(self, X: pd.DataFrame) -> NDArray[np.int64]:
        """Predict hard multiclass outcomes (0=Away win, 1=Draw, 2=Home win)."""
        probs = self.predict_proba(X)
        return cast(NDArray[np.int64], np.argmax(probs, axis=1).astype(np.int64))
