"""Integration tests for segment-aware hybrid ensemble in the training pipeline.

Validates:
✓ Tournament segment detector maps correctly
✓ Segment config builder produces valid configs
✓ Segment-aware candidates are present in candidate specs
✓ Metadata column pass-through works in backtesting
✓ End-to-end: segment-aware candidate runs through temporal evaluation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.modeling.evaluation import (
    CandidateSpec,
    evaluate_candidates_with_backtesting,
)
from src.modeling.hybrid_ensemble_segment_aware import (
    SegmentAwareHybridDrawOverrideEnsemble,
    SegmentConfig,
)
from src.modeling.segment_routing import (
    SEGMENT_METADATA_COLUMNS,
    tournament_segment_detector,
)
from src.modeling.train import (
    _build_candidate_specs,
    _build_segment_configs,
    _make_sample_weight_builder,
)

# ============================================================================
# Test: Tournament Segment Detector
# ============================================================================


class TestTournamentSegmentDetector:
    """Validate tournament_segment_detector routing logic."""

    @pytest.mark.parametrize(
        "tournament,expected_segment",
        [
            ("FIFA World Cup", "worldcup"),
            ("2026 FIFA World Cup", "worldcup"),
            ("World Cup 2022", "worldcup"),
            ("Friendly", "friendlies"),
            ("International Friendly", "friendlies"),
            ("FIFA World Cup qualification", "qualifiers"),
            ("UEFA Euro qualification", "qualifiers"),
            ("African Cup of Nations qualification", "qualifiers"),
            ("Copa América", "continental"),
            ("UEFA Euro", "continental"),
            ("Africa Cup of Nations", "continental"),
            ("Asian Cup", "continental"),
            ("Gold Cup", "continental"),
            ("UEFA Nations League", "continental"),
            ("CONCACAF Nations League", "continental"),
            ("COSAFA Cup", None),  # Unmapped → falls through to None
            ("Island Games", None),
        ],
    )
    def test_segment_mapping(
        self, tournament: str, expected_segment: str | None
    ) -> None:
        """Detector should route tournament names to correct segments."""
        row = pd.Series({"tournament": tournament})
        assert tournament_segment_detector(row) == expected_segment

    def test_none_tournament(self) -> None:
        """None or missing tournament should return None."""
        assert tournament_segment_detector(pd.Series({"tournament": None})) is None
        assert tournament_segment_detector(pd.Series({})) is None

    def test_non_string_tournament(self) -> None:
        """Non-string tournament values should return None."""
        assert tournament_segment_detector(pd.Series({"tournament": 42})) is None


# ============================================================================
# Test: Segment Config Builder
# ============================================================================


class TestBuildSegmentConfigs:
    """Validate _build_segment_configs factory function."""

    def test_produces_four_segments(self) -> None:
        """Builder should produce configs for all four segments."""
        configs = _build_segment_configs(
            friendlies_unc=0.35,
            friendlies_conv=0.55,
            worldcup_unc=0.50,
            worldcup_conv=0.60,
            continental_unc=0.42,
            continental_conv=0.57,
            qualifiers_unc=0.46,
            qualifiers_conv=0.58,
        )
        assert set(configs.keys()) == {
            "friendlies",
            "worldcup",
            "continental",
            "qualifiers",
        }
        for seg_id, config in configs.items():
            assert isinstance(config, SegmentConfig)
            assert config.segment_id == seg_id

    def test_thresholds_assigned_correctly(self) -> None:
        """Each segment should have the specified thresholds."""
        configs = _build_segment_configs(
            friendlies_unc=0.32,
            friendlies_conv=0.52,
            worldcup_unc=0.55,
            worldcup_conv=0.65,
            continental_unc=0.44,
            continental_conv=0.58,
            qualifiers_unc=0.48,
            qualifiers_conv=0.60,
        )
        assert configs["friendlies"].uncertainty_threshold == 0.32
        assert configs["friendlies"].draw_conviction_threshold == 0.52
        assert configs["worldcup"].uncertainty_threshold == 0.55
        assert configs["worldcup"].draw_conviction_threshold == 0.65


# ============================================================================
# Test: Segment-Aware Candidates in _build_candidate_specs
# ============================================================================


class TestCandidateSpecsContainSegmentAware:
    """Verify segment-aware candidates are registered in the search space."""

    def test_segment_aware_candidates_exist(self) -> None:
        """Candidate specs should contain all 4 segment-aware variants."""
        specs = _build_candidate_specs()
        expected_names = {
            "seg_hybrid_conservative",
            "seg_hybrid_friendlies_focus",
            "seg_hybrid_balanced",
            "seg_hybrid_narrow_band",
        }
        actual_seg_names = {
            name for name in specs if name.startswith("seg_hybrid_")
        }
        assert actual_seg_names == expected_names

    def test_segment_aware_family_label(self) -> None:
        """All segment-aware candidates should have correct family."""
        specs = _build_candidate_specs()
        for name, spec in specs.items():
            if name.startswith("seg_hybrid_"):
                assert spec.family == "segment_aware_hybrid"

    def test_segment_aware_hyperparameters_contain_thresholds(self) -> None:
        """Hyperparameters should include per-segment threshold values."""
        specs = _build_candidate_specs()
        for name, spec in specs.items():
            if name.startswith("seg_hybrid_"):
                hp = spec.hyperparameters
                assert "tag" in hp
                assert "default_unc" in hp
                assert "default_conv" in hp
                assert "friendlies_unc" in hp
                assert "worldcup_unc" in hp
                assert "continental_unc" in hp
                assert "qualifiers_unc" in hp

    def test_total_candidate_count_increases(self) -> None:
        """Adding 4 segment-aware variants should increase total count."""
        specs = _build_candidate_specs()
        # Original: 1 dummy + 5 logistic + 3 RF + 4 XGB + 3 two-stage + 6 hybrid = 22
        # New: + 4 segment-aware = 26
        assert len(specs) >= 26


# ============================================================================
# Test: Metadata Column Pass-Through in Backtesting
# ============================================================================


class TestMetadataPassThrough:
    """Verify metadata_columns parameter works in evaluate_candidates_with_backtesting."""

    @pytest.fixture
    def synthetic_train_df(self) -> pd.DataFrame:
        """Synthetic training DataFrame with features + tournament + target."""
        np.random.seed(42)
        n_rows = 600
        dates = pd.date_range("2010-01-01", periods=n_rows, freq="7D")
        return pd.DataFrame(
            {
                "date": dates,
                "feature_1": np.random.randn(n_rows),
                "feature_2": np.random.randn(n_rows),
                "tournament": np.random.choice(
                    [
                        "Friendly",
                        "FIFA World Cup",
                        "Copa América",
                        "FIFA World Cup qualification",
                    ],
                    n_rows,
                ),
                "target_multiclass": np.random.choice(
                    [-1, 0, 1], n_rows, p=[0.28, 0.24, 0.48]
                ),
            }
        )

    def test_backtesting_with_metadata_columns(
        self, synthetic_train_df: pd.DataFrame
    ) -> None:
        """Segment-aware candidate should work with metadata pass-through."""
        segment_configs = _build_segment_configs(
            friendlies_unc=0.36,
            friendlies_conv=0.55,
            worldcup_unc=0.50,
            worldcup_conv=0.60,
            continental_unc=0.42,
            continental_conv=0.56,
            qualifiers_unc=0.46,
            qualifiers_conv=0.58,
        )

        seg_ensemble = SegmentAwareHybridDrawOverrideEnsemble(
            generalist_estimator=RandomForestClassifier(
                n_estimators=10, random_state=42, max_depth=3
            ),
            specialist_estimator=RandomForestClassifier(
                n_estimators=10, random_state=42, max_depth=3
            ),
            segment_configs=segment_configs,
            segment_detector_fn=tournament_segment_detector,
            default_uncertainty_threshold=0.44,
            default_draw_conviction_threshold=0.55,
        )

        candidate_specs = {
            "seg_test_candidate": CandidateSpec(
                name="seg_test_candidate",
                pipeline=seg_ensemble,
                sample_weight_builder=_make_sample_weight_builder(1.0),
                family="segment_aware_hybrid",
                hyperparameters={"tag": "test"},
                notes="Integration test candidate",
            ),
        }

        feature_columns = ["feature_1", "feature_2"]
        outcome_to_encoded = {-1: 0, 0: 1, 1: 2}

        summaries, best_name = evaluate_candidates_with_backtesting(
            candidate_specs=candidate_specs,
            train_df=synthetic_train_df,
            feature_columns=feature_columns,
            target_column="target_multiclass",
            outcome_to_encoded=outcome_to_encoded,
            n_splits=3,
            metadata_columns=["tournament"],
        )

        assert len(summaries) == 1
        assert best_name == "seg_test_candidate"
        summary = summaries[0]
        assert "mean_metrics" in summary
        assert "fold_results" in summary
        mean_metrics = summary["mean_metrics"]
        assert isinstance(mean_metrics, dict)
        assert "macro_f1" in mean_metrics
        assert "draw_f1" in mean_metrics

    def test_backtesting_without_metadata_columns_unchanged(
        self, synthetic_train_df: pd.DataFrame
    ) -> None:
        """Non-segment candidates should work unchanged with metadata_columns=None."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=2500, random_state=42)),
            ]
        )

        candidate_specs = {
            "logistic_baseline": CandidateSpec(
                name="logistic_baseline",
                pipeline=pipeline,
                sample_weight_builder=_make_sample_weight_builder(1.0),
                family="logistic_regression",
                hyperparameters={"C": 1.0},
                notes="Backward-compat test",
            ),
        }

        summaries, best_name = evaluate_candidates_with_backtesting(
            candidate_specs=candidate_specs,
            train_df=synthetic_train_df,
            feature_columns=["feature_1", "feature_2"],
            target_column="target_multiclass",
            outcome_to_encoded={-1: 0, 0: 1, 1: 2},
            n_splits=3,
            metadata_columns=None,
        )

        assert len(summaries) == 1
        assert best_name == "logistic_baseline"


# ============================================================================
# Test: SEGMENT_METADATA_COLUMNS constant
# ============================================================================


class TestSegmentMetadataColumns:
    """Validate SEGMENT_METADATA_COLUMNS constant."""

    def test_contains_tournament(self) -> None:
        """Metadata columns should include 'tournament'."""
        assert "tournament" in SEGMENT_METADATA_COLUMNS

    def test_is_list_of_strings(self) -> None:
        """Metadata columns should be a list of strings."""
        assert isinstance(SEGMENT_METADATA_COLUMNS, list)
        assert all(isinstance(col, str) for col in SEGMENT_METADATA_COLUMNS)
