"""Reporting utilities for model evaluation, calibration, and segment analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from src.modeling.evaluation import evaluate_multiclass_predictions
from src.modeling.features import OUTCOME_LABELS, TARGET_COLUMN
from src.modeling.team_confederations import get_team_confederation
from src.modeling.types import TrainingSummary

OUTCOME_TO_ENCODED = {-1: 0, 0: 1, 1: 2}
ENCODED_TO_OUTCOME = {value: key for key, value in OUTCOME_TO_ENCODED.items()}
ENCODED_CLASS_ORDER = np.array([0, 1, 2], dtype=np.int64)
OUTCOME_ORDER = [-1, 0, 1]
COMPETITION_SEGMENT_ORDER = [
    "World Cup",
    "Qualifier",
    "Continental",
    "Friendly",
    "Other",
]


def _competition_segment(row: pd.Series) -> str:
    if int(row.get("is_world_cup", 0)) == 1:
        return "World Cup"
    if int(row.get("is_qualifier", 0)) == 1:
        return "Qualifier"
    if int(row.get("is_continental", 0)) == 1:
        return "Continental"
    if int(row.get("is_friendly", 0)) == 1:
        return "Friendly"
    return "Other"


def _time_window(timestamp: pd.Timestamp) -> str:
    year = int(timestamp.year)
    start_year = year if year % 2 == 1 else year - 1
    end_year = start_year + 1
    return f"{start_year}-{end_year}"


def build_prediction_frame(
    *,
    test_df: pd.DataFrame,
    probabilities: NDArray[np.float64],
    y_pred_encoded: NDArray[np.int64],
) -> pd.DataFrame:
    """Build an analysis frame with predictions, probabilities, and segment columns."""
    prediction_frame = test_df[
        [
            "date",
            "homeTeam",
            "awayTeam",
            "tournament",
            "is_friendly",
            "is_world_cup",
            "is_qualifier",
            "is_continental",
            TARGET_COLUMN,
        ]
    ].copy()
    prediction_frame["actual_outcome"] = prediction_frame[TARGET_COLUMN].astype(int)
    prediction_frame["predicted_outcome"] = np.vectorize(ENCODED_TO_OUTCOME.get)(
        y_pred_encoded.astype(np.int64)
    )
    prediction_frame["actual_label"] = prediction_frame["actual_outcome"].map(
        OUTCOME_LABELS
    )
    prediction_frame["predicted_label"] = prediction_frame["predicted_outcome"].map(
        OUTCOME_LABELS
    )
    prediction_frame["away_win_probability"] = probabilities[:, 0]
    prediction_frame["draw_probability"] = probabilities[:, 1]
    prediction_frame["home_win_probability"] = probabilities[:, 2]
    prediction_frame["max_probability"] = probabilities.max(axis=1)
    prediction_frame["competition_segment"] = prediction_frame.apply(
        _competition_segment,
        axis=1,
    )
    prediction_frame["time_window"] = prediction_frame["date"].map(_time_window)
    prediction_frame["home_confederation"] = prediction_frame["homeTeam"].map(
        get_team_confederation
    )
    prediction_frame["away_confederation"] = prediction_frame["awayTeam"].map(
        get_team_confederation
    )
    prediction_frame["confederation_segment"] = np.where(
        prediction_frame["home_confederation"]
        == prediction_frame["away_confederation"],
        prediction_frame["home_confederation"],
        np.where(
            (
                prediction_frame["home_confederation"].isin(
                    ["Unknown", "Regional/NonFIFA"]
                )
                | prediction_frame["away_confederation"].isin(
                    ["Unknown", "Regional/NonFIFA"]
                )
            ),
            "Unknown/Regional Mix",
            "Inter-confederation",
        ),
    )
    return prediction_frame


def _evaluate_segment(segment_df: pd.DataFrame) -> dict[str, object]:
    y_true = segment_df["actual_outcome"].astype("int64")
    y_true_encoded = y_true.map(OUTCOME_TO_ENCODED).astype("int64")
    y_pred_encoded = (
        segment_df["predicted_outcome"]
        .map(OUTCOME_TO_ENCODED)
        .to_numpy(
            dtype=np.int64,
            copy=False,
        )
    )
    probabilities = segment_df[
        ["away_win_probability", "draw_probability", "home_win_probability"]
    ].to_numpy(dtype=np.float64, copy=False)
    metrics = evaluate_multiclass_predictions(
        y_true=y_true,
        y_true_encoded=y_true_encoded,
        y_pred_encoded=y_pred_encoded,
        probabilities=probabilities,
    )
    return {
        "rows": int(len(segment_df)),
        "date_start": segment_df["date"].min().date().isoformat(),
        "date_end": segment_df["date"].max().date().isoformat(),
        "metrics": metrics,
    }


def build_segment_analysis(
    prediction_frame: pd.DataFrame,
    *,
    group_column: str,
    min_rows: int = 80,
) -> list[dict[str, object]]:
    """Compute segment metrics for slices with enough support."""
    segment_rows: list[dict[str, object]] = []
    for segment_value, segment_df in prediction_frame.groupby(group_column):
        if len(segment_df) < min_rows:
            continue
        evaluated = _evaluate_segment(segment_df)
        segment_rows.append(
            {
                "segment_type": group_column,
                "segment_value": str(segment_value),
                **evaluated,
            }
        )

    segment_rows.sort(
        key=lambda row: (
            -cast(int, row["rows"]),
            -cast(float, cast(dict[str, object], row["metrics"])["macro_f1"]),
        )
    )
    return segment_rows


def _plot_confusion_matrix(
    prediction_frame: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    actual = prediction_frame["actual_outcome"].to_numpy(dtype=np.int64, copy=False)
    predicted = prediction_frame["predicted_outcome"].to_numpy(
        dtype=np.int64, copy=False
    )
    matrix = confusion_matrix(actual, predicted, labels=OUTCOME_ORDER, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(OUTCOME_ORDER)))
    ax.set_xticklabels([OUTCOME_LABELS[label] for label in OUTCOME_ORDER], rotation=20)
    ax.set_yticks(range(len(OUTCOME_ORDER)))
    ax.set_yticklabels([OUTCOME_LABELS[label] for label in OUTCOME_ORDER])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Normalized Confusion Matrix")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration_curves(
    prediction_frame: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    probabilities = prediction_frame[
        ["away_win_probability", "draw_probability", "home_win_probability"]
    ].to_numpy(dtype=np.float64, copy=False)
    actual = prediction_frame["actual_outcome"].to_numpy(dtype=np.int64, copy=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, outcome, encoded_index in zip(
        axes, OUTCOME_ORDER, ENCODED_CLASS_ORDER, strict=False
    ):
        binary_target = (actual == outcome).astype(np.int64)
        prob_true, prob_pred = calibration_curve(
            binary_target,
            probabilities[:, encoded_index],
            n_bins=10,
            strategy="quantile",
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axis.plot(prob_pred, prob_true, marker="o", linewidth=2, color="#1565c0")
        axis.set_title(OUTCOME_LABELS[outcome])
        axis.set_xlabel("Predicted probability")
        axis.set_ylabel("Observed frequency")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)

    fig.suptitle("One-vs-Rest Calibration Curves", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _top_confident_errors(
    prediction_frame: pd.DataFrame, limit: int = 15
) -> list[dict[str, object]]:
    errors = prediction_frame.loc[
        prediction_frame["actual_outcome"] != prediction_frame["predicted_outcome"]
    ].copy()
    if errors.empty:
        return []
    top_errors = errors.sort_values("max_probability", ascending=False).head(limit)
    return [
        {
            "date": row["date"].date().isoformat(),
            "home_team": row["homeTeam"],
            "away_team": row["awayTeam"],
            "tournament": row["tournament"],
            "actual": row["actual_label"],
            "predicted": row["predicted_label"],
            "confidence": round(float(row["max_probability"]), 4),
            "competition_segment": row["competition_segment"],
            "time_window": row["time_window"],
        }
        for _, row in top_errors.iterrows()
    ]


def _serialize_candidate_table(
    candidate_backtests: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for candidate in candidate_backtests:
        mean_metrics = cast(dict[str, float], candidate["mean_metrics"])
        summary_rows.append(
            {
                "model_name": candidate["model_name"],
                "model_family": candidate.get("model_family"),
                "selection_rank": candidate.get("selection_rank"),
                "selection_score": candidate.get("selection_score"),
                "macro_f1": mean_metrics["macro_f1"],
                "draw_f1": mean_metrics["draw_f1"],
                "draw_recall": mean_metrics["draw_recall"],
                "balanced_accuracy": mean_metrics["balanced_accuracy"],
                "log_loss": mean_metrics["log_loss"],
                "expected_calibration_error": mean_metrics[
                    "expected_calibration_error"
                ],
                "hyperparameters": candidate.get("hyperparameters", {}),
            }
        )
    summary_rows.sort(key=lambda row: cast(int, row["selection_rank"]))
    return summary_rows


def generate_evaluation_report(
    *,
    training_summary: TrainingSummary,
    test_df: pd.DataFrame,
    probabilities: NDArray[np.float64],
    y_pred_encoded: NDArray[np.int64],
    artifact_path: Path,
) -> dict[str, object]:
    """Generate JSON/Markdown/PNG evaluation artifacts for portfolio and review."""
    reports_dir = artifact_path.parent
    stem = artifact_path.stem
    prediction_frame = build_prediction_frame(
        test_df=test_df,
        probabilities=probabilities,
        y_pred_encoded=y_pred_encoded,
    )

    competition_segments = build_segment_analysis(
        prediction_frame,
        group_column="competition_segment",
        min_rows=60,
    )
    time_window_segments = build_segment_analysis(
        prediction_frame,
        group_column="time_window",
        min_rows=60,
    )
    confederation_segments = build_segment_analysis(
        prediction_frame.loc[
            ~prediction_frame["home_confederation"].isin(
                ["Unknown", "Regional/NonFIFA"]
            )
        ],
        group_column="home_confederation",
        min_rows=80,
    )
    matchup_segments = build_segment_analysis(
        prediction_frame,
        group_column="confederation_segment",
        min_rows=80,
    )

    confusion_matrix_path = reports_dir / f"{stem}_confusion_matrix.png"
    calibration_plot_path = reports_dir / f"{stem}_calibration_curves.png"
    report_json_path = reports_dir / f"{stem}_evaluation_report.json"
    report_md_path = reports_dir / f"{stem}_evaluation_report.md"

    _plot_confusion_matrix(prediction_frame, output_path=confusion_matrix_path)
    _plot_calibration_curves(prediction_frame, output_path=calibration_plot_path)

    evaluation_artifacts = training_summary["evaluation_artifacts"]
    candidate_backtests = cast(
        list[dict[str, object]],
        evaluation_artifacts["candidate_backtests"],
    )
    classification_report = training_summary["metrics"]["classification_report"]
    draw_report = cast(dict[str, object], classification_report[OUTCOME_LABELS[0]])
    report_payload: dict[str, object] = {
        "overall_metrics": training_summary["metrics"],
        "selected_model_name": training_summary["selected_model_name"],
        "selected_model_class": training_summary["selected_model_class"],
        "deployed_model_variant": training_summary["deployed_model_variant"],
        "calibration_method": training_summary["calibration_method"],
        "candidate_search_summary": _serialize_candidate_table(candidate_backtests),
        "segment_analysis": {
            "competition_segments": competition_segments,
            "time_window_segments": time_window_segments,
            "home_confederation_segments": confederation_segments,
            "confederation_matchup_segments": matchup_segments,
        },
        "confederation_coverage": {
            "known_home_confederation_ratio": float(
                (~prediction_frame["home_confederation"].isin(["Unknown"])).mean()
            ),
            "regional_or_unknown_ratio": float(
                prediction_frame["home_confederation"]
                .isin(["Unknown", "Regional/NonFIFA"])
                .mean()
            ),
        },
        "draw_diagnostics": {
            "support": int(cast(float, draw_report["support"])),
            "precision": float(cast(float, draw_report["precision"])),
            "recall": float(cast(float, draw_report["recall"])),
            "f1_score": float(cast(float, draw_report["f1-score"])),
        },
        "top_confident_errors": _top_confident_errors(prediction_frame),
        "artifacts": {
            "confusion_matrix_png": str(confusion_matrix_path),
            "calibration_curves_png": str(calibration_plot_path),
            "report_json": str(report_json_path),
            "report_markdown": str(report_md_path),
        },
    }
    report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    top_candidates = cast(
        list[dict[str, object]], report_payload["candidate_search_summary"]
    )[:5]
    competition_lines = [
        f"- `{segment['segment_value']}`: rows={segment['rows']}, macro_f1={cast(dict[str, object], segment['metrics'])['macro_f1']:.4f}, log_loss={cast(dict[str, object], segment['metrics'])['log_loss']:.4f}"
        for segment in competition_segments[:5]
    ]
    confederation_lines = [
        f"- `{segment['segment_value']}`: rows={segment['rows']}, macro_f1={cast(dict[str, object], segment['metrics'])['macro_f1']:.4f}, draw_recall={cast(dict[str, object], segment['metrics'])['draw_recall']:.4f}"
        for segment in confederation_segments[:5]
    ]
    candidate_lines = [
        f"- `{row['model_name']}` ({row['model_family']}): rank={row['selection_rank']}, macro_f1={cast(float, row['macro_f1']):.4f}, draw_f1={cast(float, row['draw_f1']):.4f}, log_loss={cast(float, row['log_loss']):.4f}"
        for row in top_candidates
    ]
    error_lines = [
        f"- {row['date']} | {row['home_team']} vs {row['away_team']} | actual={row['actual']} predicted={row['predicted']} conf={cast(float, row['confidence']):.2f}"
        for row in cast(
            list[dict[str, object]], report_payload["top_confident_errors"]
        )[:10]
    ]
    draw_diagnostics = cast(dict[str, float], report_payload["draw_diagnostics"])

    markdown = "\n".join(
        [
            "# Model Evaluation Report",
            "",
            "## Overall",
            f"- Selected model: `{training_summary['selected_model_name']}` ({training_summary['selected_model_class']})",
            f"- Deployed variant: `{training_summary['deployed_model_variant']}`",
            f"- Accuracy: `{training_summary['metrics']['accuracy']:.4f}`",
            f"- Macro F1: `{training_summary['metrics']['macro_f1']:.4f}`",
            f"- Weighted F1: `{training_summary['metrics']['weighted_f1']:.4f}`",
            f"- Balanced accuracy: `{training_summary['metrics']['balanced_accuracy']:.4f}`",
            f"- MCC: `{training_summary['metrics']['matthews_corrcoef']:.4f}`",
            f"- Log loss: `{training_summary['metrics']['log_loss']:.4f}`",
            f"- ECE: `{training_summary['metrics']['expected_calibration_error']:.4f}`",
            "",
            "## Draw Diagnostics",
            f"- Precision: `{draw_diagnostics['precision']:.4f}`",
            f"- Recall: `{draw_diagnostics['recall']:.4f}`",
            f"- F1: `{draw_diagnostics['f1_score']:.4f}`",
            "",
            "## Top Candidate Search Results",
            *candidate_lines,
            "",
            "## Competition Segments",
            *competition_lines,
            "",
            "## Confederation Segments",
            *confederation_lines,
            "",
            "## Highest-Confidence Errors",
            *error_lines,
            "",
            "## Artifact Files",
            f"- Confusion matrix: `{confusion_matrix_path}`",
            f"- Calibration curves: `{calibration_plot_path}`",
            f"- JSON report: `{report_json_path}`",
        ]
    )
    report_md_path.write_text(markdown, encoding="utf-8")
    return report_payload
