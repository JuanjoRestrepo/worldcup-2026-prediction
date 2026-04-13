"""Two-stage draw-aware classifier for football match outcome prediction."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class TwoStageDrawClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    """
    Two-stage multiclass classifier:

    1. Predict draw vs non-draw.
    2. For non-draw matches, predict away win vs home win.

    Final multiclass probabilities are composed as:
    - away_win = (1 - p_draw) * p_away_given_non_draw
    - draw = p_draw
    - home_win = (1 - p_draw) * p_home_given_non_draw
    """

    def __init__(
        self,
        stage1_estimator: object,
        stage2_estimator: object,
        *,
        draw_probability_scale: float = 1.0,
    ) -> None:
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator
        self.draw_probability_scale = draw_probability_scale

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> TwoStageDrawClassifier:
        y_series = pd.Series(y, index=X.index, dtype="int64")
        is_draw = (y_series == 1).astype("int64")
        non_draw_mask = y_series != 1

        if not non_draw_mask.any():
            raise ValueError(
                "TwoStageDrawClassifier requires at least one non-draw sample."
            )

        self.stage1_model_ = clone(self.stage1_estimator)
        if sample_weight is None:
            self.stage1_model_.fit(X, is_draw)
        else:
            self.stage1_model_.fit(X, is_draw, model__sample_weight=sample_weight)

        stage2_y = y_series.loc[non_draw_mask].map({0: 0, 2: 1}).astype("int64")
        self.stage2_model_ = clone(self.stage2_estimator)
        if sample_weight is None:
            self.stage2_model_.fit(X.loc[non_draw_mask], stage2_y)
        else:
            self.stage2_model_.fit(
                X.loc[non_draw_mask],
                stage2_y,
                model__sample_weight=sample_weight[non_draw_mask.to_numpy()],
            )

        self.classes_ = np.array([0, 1, 2], dtype=np.int64)
        return self

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        stage1_probabilities = cast(
            NDArray[np.float64], self.stage1_model_.predict_proba(X)
        )
        stage1_classes = cast(
            NDArray[np.int64], np.asarray(self.stage1_model_.classes_, dtype=np.int64)
        )
        draw_column = int(np.where(stage1_classes == 1)[0][0])
        draw_probability = stage1_probabilities[:, draw_column]
        draw_probability = np.clip(
            draw_probability * self.draw_probability_scale, 0.0, 0.999
        )

        stage2_probabilities = cast(
            NDArray[np.float64], self.stage2_model_.predict_proba(X)
        )
        stage2_classes = cast(
            NDArray[np.int64], np.asarray(self.stage2_model_.classes_, dtype=np.int64)
        )
        away_column = int(np.where(stage2_classes == 0)[0][0])
        home_column = int(np.where(stage2_classes == 1)[0][0])

        non_draw_probability = 1.0 - draw_probability
        away_probability = non_draw_probability * stage2_probabilities[:, away_column]
        home_probability = non_draw_probability * stage2_probabilities[:, home_column]

        probabilities = np.column_stack(
            [away_probability, draw_probability, home_probability]
        ).astype(np.float64)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities / row_sums
        return cast(NDArray[np.float64], probabilities)

    def predict(self, X: pd.DataFrame) -> NDArray[np.int64]:
        probabilities = self.predict_proba(X)
        predictions = self.classes_[probabilities.argmax(axis=1)]
        return cast(NDArray[np.int64], predictions)
