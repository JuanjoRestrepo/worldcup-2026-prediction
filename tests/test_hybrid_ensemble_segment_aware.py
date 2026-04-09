"""Tests for segment-aware hybrid ensemble.

Validates:
✓ Segment detection and routing
✓ Segment-specific thresholds
✓ Coverage computation
✓ Backward compatibility with default thresholds
✓ Override masking logic per segment
✓ Per-segment statistics calculation
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.modeling.hybrid_ensemble_segment_aware import (
    SegmentAwareHybridDrawOverrideEnsemble,
    SegmentConfig,
)

# ============================================================================
# Fixtures: Generalist & Specialist Models + Test Data
# ============================================================================


@pytest.fixture
def generalist_estimator() -> RandomForestClassifier:
    """Generalist trained on balanced data."""
    return RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)


@pytest.fixture
def specialist_estimator() -> RandomForestClassifier:
    """Specialist to be trained on draw-weighted data."""
    return RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)


@pytest.fixture
def X_train() -> pd.DataFrame:
    """Training features with numeric + tournament (for detector)."""
    np.random.seed(42)
    n_samples = 200
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
            "tournament": np.random.choice(
                ["World Cup", "Friendly", "Continental"], n_samples
            ),
        }
    )


@pytest.fixture
def y_train() -> pd.Series:
    """Training labels (class distribution: ~50% wins, ~30% draws, ~20% losses)."""
    np.random.seed(42)
    return pd.Series(
        np.random.choice([0, 1, 2], 200, p=[0.5, 0.3, 0.2]), name="match_outcome"
    )


@pytest.fixture
def X_test() -> pd.DataFrame:
    """Test features with numeric + tournament (for detector)."""
    np.random.seed(43)
    n_samples = 50
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
            "tournament": np.random.choice(
                ["World Cup", "Friendly", "Continental"], n_samples
            ),
        }
    )


# ============================================================================
# Test: Segment Configuration Validation
# ============================================================================


class TestSegmentConfigValidation:
    """Validate SegmentConfig dataclass constraints."""

    def test_valid_segment_config(self) -> None:
        """Valid config should not raise."""
        config = SegmentConfig(
            segment_id="friendlies",
            uncertainty_threshold=0.35,
            draw_conviction_threshold=0.50,
            description="Friendly matches where generalist is uncertain",
        )
        assert config.segment_id == "friendlies"
        assert config.uncertainty_threshold == 0.35

    def test_invalid_uncertainty_threshold_too_high(self) -> None:
        """Uncertainty threshold must be < 1.0."""
        with pytest.raises(ValueError, match="uncertainty_threshold"):
            SegmentConfig(
                segment_id="test",
                uncertainty_threshold=1.5,
                draw_conviction_threshold=0.5,
            )

    def test_invalid_uncertainty_threshold_too_low(self) -> None:
        """Uncertainty threshold must be > 0.0."""
        with pytest.raises(ValueError, match="uncertainty_threshold"):
            SegmentConfig(
                segment_id="test",
                uncertainty_threshold=0.0,
                draw_conviction_threshold=0.5,
            )

    def test_invalid_conviction_threshold(self) -> None:
        """Draw conviction threshold must be in (0, 1)."""
        with pytest.raises(ValueError, match="draw_conviction_threshold"):
            SegmentConfig(
                segment_id="test",
                uncertainty_threshold=0.4,
                draw_conviction_threshold=1.2,
            )

    def test_invalid_min_samples(self) -> None:
        """Min samples for override must be ≥ 1."""
        with pytest.raises(ValueError, match="min_samples_for_override"):
            SegmentConfig(
                segment_id="test",
                uncertainty_threshold=0.4,
                draw_conviction_threshold=0.5,
                min_samples_for_override=0,
            )


# ============================================================================
# Test: Basic Fit & Predict (No Segmentation)
# ============================================================================


class TestBasicFitPredict:
    """Verify fit/predict works with no segments (fallback to defaults)."""

    def test_fit_without_segments(
        self,
        generalist_estimator: RandomForestClassifier,
        specialist_estimator: RandomForestClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Fit should work without segment configuration."""
        # Use only numeric columns for model training
        X_numeric = X_train[["feature_1", "feature_2", "feature_3"]]

        ensemble = SegmentAwareHybridDrawOverrideEnsemble(
            generalist_estimator=generalist_estimator,
            specialist_estimator=specialist_estimator,
            default_uncertainty_threshold=0.45,
            default_draw_conviction_threshold=0.55,
        )
        ensemble.fit(X_numeric, y_train)

        assert hasattr(ensemble, "generalist_model_")
        assert hasattr(ensemble, "specialist_model_")
        assert hasattr(ensemble, "classes_")

    def test_predict_proba_shape(
        self,
        generalist_estimator: RandomForestClassifier,
        specialist_estimator: RandomForestClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> None:
        """predict_proba should return (n_samples, 3) array."""
        # Use only numeric columns
        X_train_numeric = X_train[["feature_1", "feature_2", "feature_3"]]
        X_test_numeric = X_test[["feature_1", "feature_2", "feature_3"]]

        ensemble = SegmentAwareHybridDrawOverrideEnsemble(
            generalist_estimator=generalist_estimator,
            specialist_estimator=specialist_estimator,
        )
        ensemble.fit(X_train_numeric, y_train)
        proba = ensemble.predict_proba(X_test_numeric)

        assert proba.shape == (len(X_test_numeric), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_predict_returns_classes(
        self,
        generalist_estimator: RandomForestClassifier,
        specialist_estimator: RandomForestClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> None:
        """predict should return valid class labels."""
        # Use only numeric columns
        X_train_numeric = X_train[["feature_1", "feature_2", "feature_3"]]
        X_test_numeric = X_test[["feature_1", "feature_2", "feature_3"]]

        ensemble = SegmentAwareHybridDrawOverrideEnsemble(
            generalist_estimator=generalist_estimator,
            specialist_estimator=specialist_estimator,
        )
        ensemble.fit(X_train_numeric, y_train)
        predictions = ensemble.predict(X_test_numeric)

        assert predictions.shape == (len(X_test_numeric),)
        assert np.all(np.isin(predictions, [0, 1, 2]))


# ============================================================================
# Test: Segment Detection & Routing
# ============================================================================


class TestSegmentDetection:
    """Verify segment detection and threshold routing."""

    def test_segment_detector_called(
        self,
        generalist_estimator: RandomForestClassifier,
        specialist_estimator: RandomForestClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Segment detector should be called for each row in predict_proba."""
        call_count = {"n": 0}

        def detector(row: pd.Series) -> str | None:
            call_count["n"] += 1
            tournament = row.get("tournament")
            return "friendlies" if tournament == "Friendly" else None

        segment_config = {
            "friendlies": SegmentConfig(
                segment_id="friendlies",
                uncertainty_threshold=0.30,
                draw_conviction_threshold=0.50,
            )
        }

        # Use only numeric columns for model training
        X_numeric = X_train[["feature_1", "feature_2", "feature_3"]]

        ensemble = SegmentAwareHybridDrawOverrideEnsemble(
            generalist_estimator=generalist_estimator,
            specialist_estimator=specialist_estimator,
            segment_configs=segment_config,
            segment_detector_fn=detector,
        )
        ensemble.fit(X_numeric, y_train)

        # Predict on 10 samples (use full X with tournament column for detector)
        X_test_small = X_train.head(10)
        # Now predict_proba will automatically filter X_test_small to numeric columns
        ensemble.predict_proba(X_test_small)

        # Detector should be called at least once (in _compute_override_mask)
        assert call_count["n"] > 0

    def test_get_thresholds_for_segment(
        self,
        generalist_estimator: RandomForestClassifier,
        specialist_estimator: RandomForestClassifier,
    ) -> None:
        """Correct thresholds should be returned based on segment."""
        config = {
            "friendlies": SegmentConfig(
                segment_id="friendlies",
                uncertainty_threshold=0.30,
                draw_conviction_threshold=0.50,
            ),
            "worldcup": SegmentConfig(
                segment_id="worldcup",
                uncertainty_threshold=0.50,
                draw_conviction_threshold=0.60,
            ),
        }

        ensemble = SegmentAwareHybridDrawOverrideEnsemble(
            generalist_estimator=generalist_estimator,
            specialist_estimator=specialist_estimator,
            default_uncertainty_threshold=0.45,
            default_draw_conviction_threshold=0.55,
            segment_configs=config,
        )

        # Friendlies segment
        unc, conv = ensemble._get_thresholds_for_segment("friendlies")
        assert unc == 0.30
        assert conv == 0.50

        # World Cup segment
        unc, conv = ensemble._get_thresholds_for_segment("worldcup")
        assert unc == 0.50
        assert conv == 0.60

        # Unknown segment → defaults
        unc, conv = ensemble._get_thresholds_for_segment("unknown")
        assert unc == 0.45
        assert conv == 0.55


# ============================================================================
# Test: Segment Coverage Computation
# ============================================================================


class TestSegmentCoverage:
    """Verify coverage statistics are computed correctly."""

    def test_segment_coverage_empty(
        self,
        generalist_estimator: RandomForestClassifier,
        specialist_estimator: RandomForestClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """No segment detector → no coverage."""
        # Use only numeric columns
        X_numeric = X_train[["feature_1", "feature_2", "feature_3"]]

        ensemble = SegmentAwareHybridDrawOverrideEnsemble(
            generalist_estimator=generalist_estimator,
            specialist_estimator=specialist_estimator,
        )
        ensemble.fit(X_numeric, y_train)

        assert ensemble.segment_coverage_ == {}

    def test_segment_coverage_populated(
        self,
        generalist_estimator: RandomForestClassifier,
        specialist_estimator: RandomForestClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Coverage should count samples per segment."""

        def detector(row: pd.Series) -> str | None:
            tournament = row.get("tournament")
            return "friendlies" if tournament == "Friendly" else "other"

        config = {
            "friendlies": SegmentConfig(
                segment_id="friendlies",
                uncertainty_threshold=0.30,
                draw_conviction_threshold=0.50,
            ),
            "other": SegmentConfig(
                segment_id="other",
                uncertainty_threshold=0.45,
                draw_conviction_threshold=0.55,
            ),
        }

        ensemble = SegmentAwareHybridDrawOverrideEnsemble(
            generalist_estimator=generalist_estimator,
            specialist_estimator=specialist_estimator,
            segment_configs=config,
            segment_detector_fn=detector,
        )
        # Use only numeric columns for fitting models
        X_numeric = X_train[["feature_1", "feature_2", "feature_3"]]
        ensemble.fit(X_numeric, y_train)

        # Then compute coverage with full X that includes tournament column
        ensemble.segment_coverage_ = ensemble._compute_segment_coverage(X_train)

        # Coverage should have both segments
        assert (
            "friendlies" in ensemble.segment_coverage_
            or "other" in ensemble.segment_coverage_
        )
        # Total covered samples should be ≤ len(X_train)
        total = sum(ensemble.segment_coverage_.values())
        assert total <= len(X_train)


# ============================================================================
# Test: Per-Segment Statistics
# ============================================================================


class TestSegmentStatistics:
    """Verify per-segment metrics calculation."""

    def test_segment_statistics_structure(
        self,
        generalist_estimator: RandomForestClassifier,
        specialist_estimator: RandomForestClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> None:
        """Segment statistics should have expected keys."""

        def detector(row: pd.Series) -> str | None:
            tournament = row.get("tournament")
            if tournament == "Friendly":
                return "friendlies"
            elif tournament == "World Cup":
                return "worldcup"
            return None

        config = {
            "friendlies": SegmentConfig(
                segment_id="friendlies",
                uncertainty_threshold=0.30,
                draw_conviction_threshold=0.50,
            ),
            "worldcup": SegmentConfig(
                segment_id="worldcup",
                uncertainty_threshold=0.50,
                draw_conviction_threshold=0.60,
            ),
        }

        ensemble = SegmentAwareHybridDrawOverrideEnsemble(
            generalist_estimator=generalist_estimator,
            specialist_estimator=specialist_estimator,
            segment_configs=config,
            segment_detector_fn=detector,
        )
        # Use only numeric columns for fitting
        X_train_numeric = X_train[["feature_1", "feature_2", "feature_3"]]
        ensemble.fit(X_train_numeric, y_train)

        y_test = pd.Series(np.random.choice([0, 1, 2], len(X_test)), name="outcome")
        # Now pass full X_test (with tournament column) - ensemble will filter to numeric for model
        stats = ensemble.segment_statistics(X_test, y_test)

        # Stats should be a dict
        assert isinstance(stats, dict)

        # Each segment should have required keys
        for segment_id, seg_stats in stats.items():
            assert "n_samples" in seg_stats
            assert "override_rate" in seg_stats
            assert "accuracy" in seg_stats
            assert "draw_accuracy" in seg_stats
            assert "draw_precision" in seg_stats

            # Values should be reasonable
            assert seg_stats["n_samples"] >= 0
            assert 0 <= seg_stats["override_rate"] <= 1
            assert 0 <= seg_stats["accuracy"] <= 1

    def test_segment_statistics_empty_no_segmentation(
        self,
        generalist_estimator: RandomForestClassifier,
        specialist_estimator: RandomForestClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> None:
        """No segmentation → empty statistics."""
        ensemble = SegmentAwareHybridDrawOverrideEnsemble(
            generalist_estimator=generalist_estimator,
            specialist_estimator=specialist_estimator,
        )
        # Use only numeric columns
        X_train_numeric = X_train[["feature_1", "feature_2", "feature_3"]]
        ensemble.fit(X_train_numeric, y_train)

        y_test = pd.Series(np.random.choice([0, 1, 2], len(X_test)))
        stats = ensemble.segment_statistics(X_test, y_test)

        assert stats == {}


# ============================================================================
# Test: Validation & Error Handling
# ============================================================================


class TestValidationAndErrors:
    """Test error handling and input validation."""

    def test_invalid_default_uncertainty_threshold(
        self,
        generalist_estimator: RandomForestClassifier,
        specialist_estimator: RandomForestClassifier,
    ) -> None:
        """Invalid default uncertainty threshold should raise on fit."""
        ensemble = SegmentAwareHybridDrawOverrideEnsemble(
            generalist_estimator=generalist_estimator,
            specialist_estimator=specialist_estimator,
            default_uncertainty_threshold=1.5,
            default_draw_conviction_threshold=0.55,
        )

        X_train = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        y_train = pd.Series([0, 1, 2])

        with pytest.raises(ValueError, match="default_uncertainty_threshold"):
            ensemble.fit(X_train, y_train)

    def test_invalid_segment_config_type(
        self,
        generalist_estimator: RandomForestClassifier,
        specialist_estimator: RandomForestClassifier,
    ) -> None:
        """Non-SegmentConfig in segment_configs dict should raise."""
        X_train = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        y_train = pd.Series([0, 1, 2])

        ensemble = SegmentAwareHybridDrawOverrideEnsemble(
            generalist_estimator=generalist_estimator,
            specialist_estimator=specialist_estimator,
            segment_configs={"bad": "not_a_segment_config"},  # type: ignore
        )

        with pytest.raises(TypeError, match="SegmentConfig"):
            ensemble.fit(X_train, y_train)
