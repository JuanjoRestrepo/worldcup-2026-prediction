"""Unit tests for Dixon-Coles expected goals match outcome model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.modeling.dixon_coles import (
    DixonColesMatchPredictor,
    calculate_outcome_probs_from_score_matrix,
    calculate_score_matrix,
    dixon_coles_tau,
)


def test_dixon_coles_tau_values() -> None:
    """Verify tau correction values for low-score combinations."""
    rho = -0.05
    lambda_ = 1.5
    mu_ = 1.1

    tau_00 = dixon_coles_tau(0, 0, lambda_, mu_, rho)
    tau_10 = dixon_coles_tau(1, 0, lambda_, mu_, rho)
    tau_01 = dixon_coles_tau(0, 1, lambda_, mu_, rho)
    tau_11 = dixon_coles_tau(1, 1, lambda_, mu_, rho)
    tau_22 = dixon_coles_tau(2, 2, lambda_, mu_, rho)

    assert tau_00 > 1.0  # rho < 0 increases 0-0 probability
    assert tau_10 < 1.0  # rho < 0 decreases 1-0 probability
    assert tau_01 < 1.0
    assert tau_11 > 1.0
    assert tau_22 == 1.0  # No correction for 2+ goals


def test_calculate_score_matrix_normalized() -> None:
    """Verify score matrix dimensions and normalization."""
    matrix = calculate_score_matrix(1.5, 1.2, rho=-0.05, max_goals=5)
    assert matrix.shape == (6, 6)
    assert pytest.approx(float(matrix.sum()), abs=1e-5) == 1.0


def test_calculate_outcome_probs_knockout_suppression() -> None:
    """Verify draw probability is 0 in knockout mode."""
    matrix = calculate_score_matrix(1.5, 1.2, rho=-0.05, max_goals=5)

    p_away_reg, p_draw_reg, p_home_reg = calculate_outcome_probs_from_score_matrix(
        matrix, is_knockout=False
    )
    assert p_draw_reg > 0.1
    assert pytest.approx(p_away_reg + p_draw_reg + p_home_reg, abs=1e-5) == 1.0

    p_away_ko, p_draw_ko, p_home_ko = calculate_outcome_probs_from_score_matrix(
        matrix, is_knockout=True
    )
    assert p_draw_ko == 0.0
    assert pytest.approx(p_away_ko + p_home_ko, abs=1e-5) == 1.0
    assert p_home_ko > p_away_ko


def test_dixon_coles_match_predictor_fit_predict() -> None:
    """Verify DixonColesMatchPredictor estimator interface."""
    X = pd.DataFrame(
        {
            "home_global_avg_goals_last5": [1.8, 1.2, 0.9, 2.1],
            "away_global_avg_goals_last5": [0.8, 1.5, 1.1, 0.7],
            "elo_diff": [200.0, -150.0, -50.0, 300.0],
            "is_knockout": [0, 0, 1, 1],
        }
    )
    y = pd.Series([2, 0, 0, 2])  # 2=Home win, 0=Away win
    hg = pd.Series([2, 0, 0, 3])
    ag = pd.Series([0, 2, 1, 0])

    model = DixonColesMatchPredictor(n_estimators=10)
    model.fit(X, y, home_goals=hg, away_goals=ag)

    probs = model.predict_proba(X)
    assert probs.shape == (4, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Knockout rows (indices 2 & 3) must have 0 draw prob
    assert probs[2, 1] == 0.0
    assert probs[3, 1] == 0.0

    preds = model.predict(X)
    assert len(preds) == 4
