"""Unit tests for model reporting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling.reporting import build_prediction_frame, build_segment_analysis


def test_build_prediction_frame_adds_segments_and_probabilities():
    test_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "homeTeam": ["Brazil", "United States"],
            "awayTeam": ["Argentina", "Japan"],
            "tournament": ["FIFA World Cup", "Friendly"],
            "is_friendly": [0, 1],
            "is_world_cup": [1, 0],
            "is_qualifier": [0, 0],
            "is_continental": [0, 0],
            "target_multiclass": [1, 0],
        }
    )
    probabilities = np.array([[0.2, 0.2, 0.6], [0.3, 0.5, 0.2]], dtype=np.float64)
    y_pred_encoded = np.array([2, 1], dtype=np.int64)

    frame = build_prediction_frame(
        test_df=test_df,
        probabilities=probabilities,
        y_pred_encoded=y_pred_encoded,
    )

    assert "competition_segment" in frame.columns
    assert "time_window" in frame.columns
    assert "home_confederation" in frame.columns
    assert frame.loc[0, "competition_segment"] == "World Cup"
    assert frame.loc[1, "competition_segment"] == "Friendly"
    assert frame.loc[0, "home_confederation"] == "CONMEBOL"


def test_build_segment_analysis_filters_small_groups():
    prediction_frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=6, freq="D"),
            "competition_segment": ["Qualifier", "Qualifier", "Qualifier", "Friendly", "Friendly", "Other"],
            "actual_outcome": [1, 0, -1, 1, 0, -1],
            "predicted_outcome": [1, 0, -1, 1, 1, -1],
            "away_win_probability": [0.1, 0.2, 0.7, 0.1, 0.2, 0.8],
            "draw_probability": [0.2, 0.6, 0.1, 0.2, 0.3, 0.1],
            "home_win_probability": [0.7, 0.2, 0.2, 0.7, 0.5, 0.1],
        }
    )

    segments = build_segment_analysis(
        prediction_frame,
        group_column="competition_segment",
        min_rows=2,
    )

    segment_names = {segment["segment_value"] for segment in segments}
    assert "Qualifier" in segment_names
    assert "Friendly" in segment_names
    assert "Other" not in segment_names
