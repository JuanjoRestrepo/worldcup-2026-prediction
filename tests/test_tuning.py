"""Unit tests for the semantic threshold tuning logic using out-of-fold predictions."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.hybrid_ensemble_segment_aware import SegmentConfig
from src.modeling.tuning import auto_tune_segment_thresholds


class MockEstimator:
    def __init__(self, probabilities: np.ndarray, classes: np.ndarray):
        self.probabilities = probabilities
        self.classes_ = classes
        
    def get_params(self, deep=True):
        return {"probabilities": self.probabilities, "classes": self.classes_}
        
    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        idx = X["dummy"].astype(int).to_numpy()
        return self.probabilities[idx]


def test_auto_tune_segment_thresholds_fallback():
    # Setup data where the generalist is excellent and specialist is terrible
    n_samples = 100
    X = pd.DataFrame({"dummy": np.arange(n_samples)})
    y_encoded = pd.Series(np.zeros(n_samples, dtype=int))  # All 0
    
    metadata_df = pd.DataFrame({"tournament": ["FIFA World Cup"] * n_samples})

    def segment_detector_fn(row):
        return "worldcup"

    # Generalist predicts class 0 perfectly
    gen_probs = np.zeros((n_samples, 3))
    gen_probs[:, 0] = 1.0

    # Specialist incorrectly predicts class 1 (Draw) perfectly
    spec_probs = np.zeros((n_samples, 3))
    spec_probs[:, 1] = 1.0

    gen_estimator = MockEstimator(gen_probs, np.array([0, 1, 2]))
    spec_estimator = MockEstimator(spec_probs, np.array([0, 1, 2]))

    def weight_fn(y):
        return np.ones(len(y), dtype=np.float64)

    tuned_configs = auto_tune_segment_thresholds(
        X=X,
        y_encoded=y_encoded,
        metadata_df=metadata_df,
        segment_detector_fn=segment_detector_fn,
        generalist_pipeline=gen_estimator,
        generalist_sample_weight_fn=weight_fn,
        specialist_pipeline=spec_estimator,
        specialist_sample_weight_fn=weight_fn,
        n_splits=3,
        max_log_loss_degradation=0.005,
    )

    # Since the specialist only predicts wrong things, it should fallback to 0.01 / 0.99
    assert "worldcup" in tuned_configs
    config = tuned_configs["worldcup"]
    assert config.uncertainty_threshold == 0.01
    assert config.draw_conviction_threshold == 0.99


def test_auto_tune_segment_thresholds_improves():
    # Setup data where there's a mix, but overriding uncertain generalist predictions with
    # specialist draw predictions explicitly increases F1 Draw without destroying log loss.
    n_samples = 30
    X = pd.DataFrame({"dummy": np.arange(n_samples)})
    
    # 20 samples class 0, 10 samples class 1 (Draw)
    y_arr = np.zeros(n_samples, dtype=int)
    y_arr[20:] = 1 
    y_encoded = pd.Series(y_arr)
    
    metadata_df = pd.DataFrame({"tournament": ["Friendly"] * n_samples})

    def segment_detector_fn(row):
        return "friendlies"

    gen_probs = np.zeros((n_samples, 3))
    spec_probs = np.zeros((n_samples, 3))
    
    # First 20: Generalist perfectly confident & correct
    gen_probs[:20, 0] = 0.9
    gen_probs[:20, 1] = 0.05
    gen_probs[:20, 2] = 0.05
    
    # Last 10: Generalist is UNCERTAIN (0.45) but predicts class 0
    gen_probs[20:, 0] = 0.45
    gen_probs[20:, 1] = 0.35  # Misses the draw
    gen_probs[20:, 2] = 0.20
    
    # Specialist correctly detects draw for the uncertain cases with conviction > 0.50
    spec_probs[20:, 0] = 0.1
    spec_probs[20:, 1] = 0.60
    spec_probs[20:, 2] = 0.3
    
    # Make sure specialist is ignored for the confident cases just in case
    spec_probs[:20] = gen_probs[:20]

    gen_estimator = MockEstimator(gen_probs, np.array([0, 1, 2]))
    spec_estimator = MockEstimator(spec_probs, np.array([0, 1, 2]))

    def weight_fn(y):
        return np.ones(len(y), dtype=np.float64)

    tuned_configs = auto_tune_segment_thresholds(
        X=X,
        y_encoded=y_encoded,
        metadata_df=metadata_df,
        segment_detector_fn=segment_detector_fn,
        generalist_pipeline=gen_estimator,
        generalist_sample_weight_fn=weight_fn,
        specialist_pipeline=spec_estimator,
        specialist_sample_weight_fn=weight_fn,
        n_splits=3,
        max_log_loss_degradation=0.5, # Allow large degradation for the sake of the test
    )

    assert "friendlies" in tuned_configs
    config = tuned_configs["friendlies"]
    # The grid checks unc from 0.30 to 0.64. At unc = 0.46 (since gen_probs are 0.45),
    # the override condition gen_confidence (< 0.46) would trigger.
    # Conviction is 0.60, so threshold <= 0.60 works.
    assert config.uncertainty_threshold > 0.45
    assert config.draw_conviction_threshold <= 0.60
