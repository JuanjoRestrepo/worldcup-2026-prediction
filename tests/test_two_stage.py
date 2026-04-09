"""Unit tests for the two-stage draw-aware classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling.two_stage import TwoStageDrawClassifier


def _scaled_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def test_two_stage_classifier_produces_multiclass_probabilities():
    X = pd.DataFrame(
        {
            "elo_diff": [-120, -80, -10, 0, 10, 70, 130, 5, -5, 95, -95, 25],
            "is_qualifier": [0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        }
    )
    y = pd.Series([0, 0, 1, 1, 1, 2, 2, 1, 1, 2, 0, 2], dtype="int64")
    sample_weight = np.ones(len(X), dtype=np.float64)
    sample_weight[y.to_numpy() == 1] = 2.0

    model = TwoStageDrawClassifier(
        stage1_estimator=_scaled_logistic(),
        stage2_estimator=_scaled_logistic(),
        draw_probability_scale=1.05,
    )
    model.fit(X, y, sample_weight=sample_weight)

    probabilities = model.predict_proba(X.iloc[:3])
    predictions = model.predict(X.iloc[:3])

    assert probabilities.shape == (3, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(predictions.tolist()).issubset({0, 1, 2})
    assert model.classes_.tolist() == [0, 1, 2]
