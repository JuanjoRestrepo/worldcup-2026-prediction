"""Tests for reporting_comparison.py — model comparison report generator.

Tests cover:
- PROMOTE_V2 decision when v2 strictly dominates both primary gate metrics.
- KEEP_V1 decision when v2 Log-Loss regresses beyond tolerance.
- EQUIVALENT → PROMOTE_V2 decision when deltas are within tolerance.
- Required Markdown sections are present in the output file.
- FileNotFoundError on missing report paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.modeling.reporting_comparison import (
    EQUIVALENCE_TOLERANCE,
    generate_comparison_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_report(
    *,
    accuracy: float = 0.56,
    macro_f1: float = 0.48,
    log_loss: float = 0.97,
    ece: float = 0.04,
    draw_precision: float = 0.29,
    draw_recall: float = 0.21,
    draw_f1: float = 0.24,
    draw_support: int = 1500,
    model_name: str = "logistic_c1_draw1",
    deployed_variant: str = "uncalibrated",
    competition_segments: list[dict] | None = None,
    confederation_segments: list[dict] | None = None,
) -> dict:
    """Build a minimal evaluation report JSON structure for testing."""
    if competition_segments is None:
        competition_segments = [
            {
                "segment_value": "World Cup",
                "rows": 400,
                "metrics": {"macro_f1": 0.52, "log_loss": 0.85, "draw_recall": 0.20},
            },
            {
                "segment_value": "Friendly",
                "rows": 350,
                "metrics": {"macro_f1": 0.46, "log_loss": 0.99, "draw_recall": 0.18},
            },
        ]
    if confederation_segments is None:
        confederation_segments = [
            {
                "segment_value": "UEFA",
                "rows": 500,
                "metrics": {"macro_f1": 0.51, "log_loss": 0.92, "draw_recall": 0.22},
            }
        ]

    return {
        "overall_metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "log_loss": log_loss,
            "weighted_f1": accuracy - 0.01,
            "balanced_accuracy": accuracy - 0.03,
            "matthews_corrcoef": 0.31,
            "cohen_kappa": 0.25,
            "multiclass_brier_score": 0.22,
            "expected_calibration_error": ece,
            "draw_f1": draw_f1,
            "draw_recall": draw_recall,
        },
        "selected_model_name": model_name,
        "selected_model_class": "LogisticRegression",
        "deployed_model_variant": deployed_variant,
        "calibration_method": "none",
        "draw_diagnostics": {
            "support": draw_support,
            "precision": draw_precision,
            "recall": draw_recall,
            "f1_score": draw_f1,
        },
        "segment_analysis": {
            "competition_segments": competition_segments,
            "home_confederation_segments": confederation_segments,
        },
    }


def _write_reports(
    tmp_path: Path,
    v1_data: dict,
    v2_data: dict,
) -> tuple[Path, Path, Path]:
    """Write two report JSONs and return (v1_path, v2_path, output_path)."""
    v1 = tmp_path / "report_v1.json"
    v2 = tmp_path / "report_v2.json"
    out = tmp_path / "comparison.md"
    v1.write_text(json.dumps(v1_data), encoding="utf-8")
    v2.write_text(json.dumps(v2_data), encoding="utf-8")
    return v1, v2, out


# ---------------------------------------------------------------------------
# Tests: promotion gate decisions
# ---------------------------------------------------------------------------


class TestPromotionDecision:
    """Verify the three possible promotion gate outcomes."""

    def test_promote_v2_when_both_metrics_improve(self, tmp_path: Path) -> None:
        """PROMOTE_V2 is returned when v2 has lower Log-Loss AND higher Macro F1."""
        v1 = _build_report(log_loss=0.97, macro_f1=0.480)
        v2 = _build_report(log_loss=0.96, macro_f1=0.485)
        v1_path, v2_path, out = _write_reports(tmp_path, v1, v2)

        result = generate_comparison_report(v1_path, v2_path, out)

        assert "PROMOTE_V2" in result["decision"]
        assert out.exists()

    def test_keep_v1_when_log_loss_regresses(self, tmp_path: Path) -> None:
        """KEEP_V1 is returned when v2 Log-Loss is worse beyond tolerance."""
        delta = EQUIVALENCE_TOLERANCE + 0.005
        v1 = _build_report(log_loss=0.97, macro_f1=0.480)
        v2 = _build_report(log_loss=0.97 + delta, macro_f1=0.490)
        v1_path, v2_path, out = _write_reports(tmp_path, v1, v2)

        result = generate_comparison_report(v1_path, v2_path, out)

        assert result["decision"] == "KEEP_V1"

    def test_keep_v1_when_macro_f1_regresses(self, tmp_path: Path) -> None:
        """KEEP_V1 is returned when v2 Macro F1 is worse beyond tolerance."""
        delta = EQUIVALENCE_TOLERANCE + 0.005
        v1 = _build_report(log_loss=0.97, macro_f1=0.480)
        v2 = _build_report(log_loss=0.96, macro_f1=0.480 - delta)
        v1_path, v2_path, out = _write_reports(tmp_path, v1, v2)

        result = generate_comparison_report(v1_path, v2_path, out)

        assert result["decision"] == "KEEP_V1"

    def test_equivalent_promotes_v2_for_data_recency(self, tmp_path: Path) -> None:
        """When delta is within tolerance on both metrics EQUIVALENT→PROMOTE_V2."""
        tiny = EQUIVALENCE_TOLERANCE / 2
        v1 = _build_report(log_loss=0.97, macro_f1=0.480)
        v2 = _build_report(log_loss=0.97 + tiny, macro_f1=0.480 + tiny)
        v1_path, v2_path, out = _write_reports(tmp_path, v1, v2)

        result = generate_comparison_report(v1_path, v2_path, out)

        assert "EQUIVALENT" in result["decision"]
        assert "PROMOTE_V2" in result["decision"]


# ---------------------------------------------------------------------------
# Tests: output file structure
# ---------------------------------------------------------------------------


class TestMarkdownStructure:
    """The generated Markdown contains all required sections."""

    _REQUIRED_SECTIONS = [
        "# Model Comparison Report",
        "## Promotion Decision",
        "## Overall Metrics",
        "## Draw Diagnostics",
        "## Competition Segment Breakdown",
        "## Confederation Segment Breakdown",
        "## Artifacts",
    ]

    def test_all_required_sections_present(self, tmp_path: Path) -> None:
        """Every required Markdown section heading appears in the output."""
        v1 = _build_report(log_loss=0.97, macro_f1=0.480)
        v2 = _build_report(log_loss=0.96, macro_f1=0.485)
        v1_path, v2_path, out = _write_reports(tmp_path, v1, v2)

        generate_comparison_report(v1_path, v2_path, out)

        content = out.read_text(encoding="utf-8")
        for section in self._REQUIRED_SECTIONS:
            assert section in content, f"Missing section: {section}"

    def test_return_dict_has_required_keys(self, tmp_path: Path) -> None:
        """The return value contains decision, rationale, output_path, and metrics."""
        v1 = _build_report(log_loss=0.97, macro_f1=0.480)
        v2 = _build_report(log_loss=0.96, macro_f1=0.485)
        v1_path, v2_path, out = _write_reports(tmp_path, v1, v2)

        result = generate_comparison_report(v1_path, v2_path, out)

        for key in ("decision", "rationale", "output_path", "v1_metrics", "v2_metrics"):
            assert key in result, f"Missing key in return dict: {key}"

    def test_metric_values_in_return_dict(self, tmp_path: Path) -> None:
        """Metric values in the return dict match the source report values."""
        v1 = _build_report(log_loss=0.9700, macro_f1=0.4800)
        v2 = _build_report(log_loss=0.9600, macro_f1=0.4850)
        v1_path, v2_path, out = _write_reports(tmp_path, v1, v2)

        result = generate_comparison_report(v1_path, v2_path, out)

        assert abs(result["v1_metrics"]["log_loss"] - 0.97) < 1e-6  # type: ignore[index]
        assert abs(result["v2_metrics"]["log_loss"] - 0.96) < 1e-6  # type: ignore[index]


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """FileNotFoundError and ValueError are raised for invalid inputs."""

    def test_raises_on_missing_v1_report(self, tmp_path: Path) -> None:
        """FileNotFoundError if the v1 report path does not exist."""
        v2 = tmp_path / "report_v2.json"
        v2.write_text(json.dumps(_build_report()), encoding="utf-8")

        with pytest.raises(FileNotFoundError):
            generate_comparison_report(
                tmp_path / "does_not_exist.json",
                v2,
                tmp_path / "out.md",
            )

    def test_raises_on_missing_v2_report(self, tmp_path: Path) -> None:
        """FileNotFoundError if the v2 report path does not exist."""
        v1 = tmp_path / "report_v1.json"
        v1.write_text(json.dumps(_build_report()), encoding="utf-8")

        with pytest.raises(FileNotFoundError):
            generate_comparison_report(
                v1,
                tmp_path / "does_not_exist.json",
                tmp_path / "out.md",
            )

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        """ValueError if a report file contains malformed JSON."""
        bad = tmp_path / "bad.json"
        bad.write_text("NOT_JSON{{{", encoding="utf-8")
        good = tmp_path / "good.json"
        good.write_text(json.dumps(_build_report()), encoding="utf-8")

        with pytest.raises(ValueError, match="Could not parse JSON"):
            generate_comparison_report(bad, good, tmp_path / "out.md")
