"""Model comparison report generator.

Produces a side-by-side Markdown report comparing evaluation metrics between
two trained model versions, and emits a data-driven promotion decision.

Promotion gate:
    PROMOTE_V2  → ``log_loss_v2 <= log_loss_v1  AND  macro_f1_v2 >= macro_f1_v1``
    KEEP_V1     → either condition fails
    EQUIVALENT  → both models within ±0.002 on both metrics (promote v2 for data recency)

Intended use (one-off, after a retrain cycle):

    from pathlib import Path
    from src.modeling.reporting_comparison import generate_comparison_report

    result = generate_comparison_report(
        v1_report_path=Path("models/match_predictor_evaluation_report_v1.json"),
        v2_report_path=Path("models/match_predictor_evaluation_report.json"),
        output_path=Path("models/model_comparison_report.md"),
    )
    print(result["decision"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Threshold below which the absolute delta between two metric values is
# considered statistically negligible for the purposes of the promotion gate.
EQUIVALENCE_TOLERANCE: float = 0.002

# Direction arrows used in the Markdown table.
_ARROW_UP = "↑"
_ARROW_DOWN = "↓"
_ARROW_FLAT = "—"

# Competition segment order for display consistency.
_COMPETITION_SEGMENT_ORDER = [
    "World Cup",
    "Qualifier",
    "Continental",
    "Friendly",
    "Other",
]

# Confederation segment order for display consistency.
_CONFEDERATION_ORDER = ["UEFA", "AFC", "CAF", "CONCACAF", "CONMEBOL"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_report(path: Path) -> dict[str, object]:
    """Load and return a JSON evaluation report.

    Args:
        path: Path to the JSON report file.

    Returns:
        Deserialized report payload.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file cannot be parsed as JSON.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: {path}. "
            "Ensure training has been completed before generating a comparison."
        )
    try:
        return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON report at {path}: {exc}") from exc


def _fmt(value: float | None, decimals: int = 4) -> str:
    """Format a float for display, returning 'N/A' for None."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def _delta_arrow(delta: float, higher_is_better: bool = True) -> str:
    """Return a directional arrow for a metric delta.

    Args:
        delta: v2 - v1 (positive means v2 is numerically higher).
        higher_is_better: True for metrics like F1/Accuracy; False for Log-Loss/ECE.

    Returns:
        Arrow string: ↑ (improved), ↓ (regressed), — (negligible).
    """
    if abs(delta) < EQUIVALENCE_TOLERANCE:
        return _ARROW_FLAT
    improved = delta > 0 if higher_is_better else delta < 0
    return _ARROW_UP if improved else _ARROW_DOWN


def _get_overall(report: dict[str, object]) -> dict[str, float]:
    """Extract the overall metrics dict from a report payload."""
    return cast(dict[str, float], report["overall_metrics"])


def _get_draw(report: dict[str, object]) -> dict[str, float]:
    """Extract draw diagnostics from a report payload."""
    return cast(dict[str, float], report["draw_diagnostics"])


def _get_segments(
    report: dict[str, object], segment_type: str
) -> dict[str, dict[str, object]]:
    """Return a name→{rows, metrics} mapping for a segment type.

    Args:
        report: Deserialized report payload.
        segment_type: One of 'competition_segments', 'home_confederation_segments'.

    Returns:
        Mapping of segment value → segment data dict.
    """
    segment_analysis = cast(dict[str, object], report.get("segment_analysis", {}))
    segments_list = cast(
        list[dict[str, object]], segment_analysis.get(segment_type, [])
    )
    return {cast(str, seg["segment_value"]): seg for seg in segments_list}


# ---------------------------------------------------------------------------
# Markdown rendering helpers
# ---------------------------------------------------------------------------


def _metric_table_row(
    label: str,
    v1_val: float | None,
    v2_val: float | None,
    higher_is_better: bool = True,
    decimals: int = 4,
) -> str:
    """Render a single Markdown table row for a metric comparison."""
    delta = (v2_val - v1_val) if (v1_val is not None and v2_val is not None) else None
    arrow = _delta_arrow(delta, higher_is_better) if delta is not None else _ARROW_FLAT
    delta_str = f"{delta:+.{decimals}f}" if delta is not None else "N/A"
    return (
        f"| {label} | {_fmt(v1_val, decimals)} | {_fmt(v2_val, decimals)} "
        f"| {delta_str} | {arrow} |"
    )


def _segment_table(
    v1_segs: dict[str, dict[str, object]],
    v2_segs: dict[str, dict[str, object]],
    order: list[str],
) -> list[str]:
    """Render a Markdown table comparing segment metrics across two models.

    Args:
        v1_segs: Segment name → data dict for v1.
        v2_segs: Segment name → data dict for v2.
        order: Preferred display order for segment names.

    Returns:
        List of Markdown table line strings.
    """
    all_segments = sorted(
        set(v1_segs) | set(v2_segs),
        key=lambda s: order.index(s) if s in order else len(order),
    )
    lines = [
        "| Segment | v1 rows | v1 F1 | v1 LL | v2 rows | v2 F1 | v2 LL | Δ F1 | Δ LL |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for seg in all_segments:
        v1_data = v1_segs.get(seg)
        v2_data = v2_segs.get(seg)

        def _mval(data: dict[str, object] | None, key: str) -> float | None:
            if data is None:
                return None
            metrics = cast(dict[str, float], data.get("metrics", {}))
            val = metrics.get(key)
            return float(val) if val is not None else None

        v1_f1 = _mval(v1_data, "macro_f1")
        v2_f1 = _mval(v2_data, "macro_f1")
        v1_ll = _mval(v1_data, "log_loss")
        v2_ll = _mval(v2_data, "log_loss")
        v1_rows = int(cast(int, v1_data["rows"])) if v1_data else 0
        v2_rows = int(cast(int, v2_data["rows"])) if v2_data else 0

        delta_f1 = (
            f"{(v2_f1 - v1_f1):+.4f}"
            if (v1_f1 is not None and v2_f1 is not None)
            else "N/A"
        )
        delta_ll = (
            f"{(v2_ll - v1_ll):+.4f}"
            if (v1_ll is not None and v2_ll is not None)
            else "N/A"
        )
        lines.append(
            f"| {seg} | {v1_rows} | {_fmt(v1_f1)} | {_fmt(v1_ll)} "
            f"| {v2_rows} | {_fmt(v2_f1)} | {_fmt(v2_ll)} "
            f"| {delta_f1} | {delta_ll} |"
        )
    return lines


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


def _evaluate_promotion(
    v1_overall: dict[str, float],
    v2_overall: dict[str, float],
) -> tuple[str, str]:
    """Apply the promotion gate and return (decision, rationale).

    Gate rules (applied in order):
    1. If both ``|Δ log_loss|`` and ``|Δ macro_f1|`` are within
       EQUIVALENCE_TOLERANCE → EQUIVALENT (promote v2 for data recency).
    2. PROMOTE_V2 if ``log_loss_v2 ≤ log_loss_v1 AND macro_f1_v2 ≥ macro_f1_v1``.
    3. Otherwise KEEP_V1.

    Args:
        v1_overall: Overall metric dict for the v1 model.
        v2_overall: Overall metric dict for the v2 model.

    Returns:
        Tuple of (decision_label, rationale_string).
    """
    v1_ll = v1_overall["log_loss"]
    v2_ll = v2_overall["log_loss"]
    v1_f1 = v1_overall["macro_f1"]
    v2_f1 = v2_overall["macro_f1"]

    delta_ll = v2_ll - v1_ll
    delta_f1 = v2_f1 - v1_f1

    if (
        abs(delta_ll) <= EQUIVALENCE_TOLERANCE
        and abs(delta_f1) <= EQUIVALENCE_TOLERANCE
    ):
        decision = "EQUIVALENT — PROMOTE_V2"
        rationale = (
            f"Both models are within the equivalence tolerance "
            f"(±{EQUIVALENCE_TOLERANCE}) on Log-Loss (Δ={delta_ll:+.4f}) and "
            f"Macro F1 (Δ={delta_f1:+.4f}). **Promoting v2** for data recency: "
            "the updated ELO ratings reflecting March 2026 matches are more accurate "
            "for tournament prediction even when aggregate classification metrics are stable."
        )
    elif v2_ll <= v1_ll and v2_f1 >= v1_f1:
        decision = "PROMOTE_V2"
        rationale = (
            f"v2 strictly improves on both primary metrics: "
            f"Log-Loss {_fmt(v1_ll)} → {_fmt(v2_ll)} (Δ={delta_ll:+.4f} ↓ better), "
            f"Macro F1 {_fmt(v1_f1)} → {_fmt(v2_f1)} (Δ={delta_f1:+.4f} ↑ better). "
            "Promote v2 to production."
        )
    else:
        failing = []
        if v2_ll > v1_ll + EQUIVALENCE_TOLERANCE:
            failing.append(
                f"Log-Loss regressed: {_fmt(v1_ll)} → {_fmt(v2_ll)} (Δ={delta_ll:+.4f})"
            )
        if v2_f1 < v1_f1 - EQUIVALENCE_TOLERANCE:
            failing.append(
                f"Macro F1 regressed: {_fmt(v1_f1)} → {_fmt(v2_f1)} (Δ={delta_f1:+.4f})"
            )
        decision = "KEEP_V1"
        rationale = (
            "v2 does not satisfy the promotion gate. Failing condition(s): "
            + "; ".join(failing)
            + ". Investigate potential distribution shift from the new data before promoting. "
            "v2 artifact is archived as `match_predictor_v2_apr2026.joblib` for reference."
        )

    return decision, rationale


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_comparison_report(
    v1_report_path: Path,
    v2_report_path: Path,
    output_path: Path,
    *,
    v1_label: str = "v1 (Jan 2026)",
    v2_label: str = "v2 (Apr 2026)",
) -> dict[str, object]:
    """Generate a side-by-side model comparison Markdown report.

    Loads both evaluation JSON reports, computes metric deltas, applies the
    promotion gate, and writes a structured Markdown file to *output_path*.

    Args:
        v1_report_path: Path to the v1 model evaluation JSON report.
        v2_report_path: Path to the v2 model evaluation JSON report.
        output_path: Destination path for the comparison Markdown report.
        v1_label: Display label for the v1 model (column header).
        v2_label: Display label for the v2 model (column header).

    Returns:
        Dictionary with keys ``decision``, ``rationale``, ``output_path``,
        ``v1_metrics``, and ``v2_metrics``.
    """
    logger.info(
        "Generating model comparison report: %s vs %s", v1_report_path, v2_report_path
    )

    v1_report = _load_report(v1_report_path)
    v2_report = _load_report(v2_report_path)

    v1_overall = _get_overall(v1_report)
    v2_overall = _get_overall(v2_report)
    v1_draw = _get_draw(v1_report)
    v2_draw = _get_draw(v2_report)

    decision, rationale = _evaluate_promotion(v1_overall, v2_overall)
    logger.info("Promotion gate decision: %s", decision)

    # ── Segment data ─────────────────────────────────────────────────────────
    v1_comp_segs = _get_segments(v1_report, "competition_segments")
    v2_comp_segs = _get_segments(v2_report, "competition_segments")
    v1_conf_segs = _get_segments(v1_report, "home_confederation_segments")
    v2_conf_segs = _get_segments(v2_report, "home_confederation_segments")

    # ── Markdown assembly ─────────────────────────────────────────────────────
    lines: list[str] = [
        "# Model Comparison Report: v1 vs v2",
        "",
        f"> Generated by `reporting_comparison.py` | {v1_label} → {v2_label}",
        "",
        "---",
        "",
        "## Promotion Decision",
        "",
        f"**Decision: `{decision}`**",
        "",
        f"> {rationale}",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        f"| Metric | {v1_label} | {v2_label} | Delta | Direction |",
        "|---|---|---|---|---|",
        _metric_table_row(
            "Accuracy", v1_overall.get("accuracy"), v2_overall.get("accuracy")
        ),
        _metric_table_row(
            "Macro F1", v1_overall.get("macro_f1"), v2_overall.get("macro_f1")
        ),
        _metric_table_row(
            "Weighted F1",
            v1_overall.get("weighted_f1"),
            v2_overall.get("weighted_f1"),
        ),
        _metric_table_row(
            "Balanced Accuracy",
            v1_overall.get("balanced_accuracy"),
            v2_overall.get("balanced_accuracy"),
        ),
        _metric_table_row(
            "MCC",
            v1_overall.get("matthews_corrcoef"),
            v2_overall.get("matthews_corrcoef"),
        ),
        _metric_table_row(
            "Log-Loss",
            v1_overall.get("log_loss"),
            v2_overall.get("log_loss"),
            higher_is_better=False,
        ),
        _metric_table_row(
            "ECE",
            v1_overall.get("expected_calibration_error"),
            v2_overall.get("expected_calibration_error"),
            higher_is_better=False,
        ),
        _metric_table_row(
            "Multiclass Brier",
            v1_overall.get("multiclass_brier_score"),
            v2_overall.get("multiclass_brier_score"),
            higher_is_better=False,
        ),
        "",
        "---",
        "",
        "## Draw Diagnostics (Hard Class)",
        "",
        f"| Metric | {v1_label} | {v2_label} | Delta | Direction |",
        "|---|---|---|---|---|",
        _metric_table_row(
            "Draw Precision", v1_draw.get("precision"), v2_draw.get("precision")
        ),
        _metric_table_row("Draw Recall", v1_draw.get("recall"), v2_draw.get("recall")),
        _metric_table_row("Draw F1", v1_draw.get("f1_score"), v2_draw.get("f1_score")),
        _metric_table_row(
            "Draw Support (rows)",
            float(v1_draw.get("support", 0)),
            float(v2_draw.get("support", 0)),
            decimals=0,
        ),
        "",
        "---",
        "",
        "## Competition Segment Breakdown",
        "",
        *_segment_table(v1_comp_segs, v2_comp_segs, _COMPETITION_SEGMENT_ORDER),
        "",
        "---",
        "",
        "## Confederation Segment Breakdown",
        "",
        *_segment_table(v1_conf_segs, v2_conf_segs, _CONFEDERATION_ORDER),
        "",
        "---",
        "",
        "## Artifacts",
        "",
        f"- v1 evaluation report: `{v1_report_path}`",
        f"- v2 evaluation report: `{v2_report_path}`",
        f"- Comparison report: `{output_path}`",
        "",
        "---",
        "",
        "_Report generated automatically by `src/modeling/reporting_comparison.py`._",
    ]

    markdown = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    logger.info("Comparison report written to %s", output_path)

    return {
        "decision": decision,
        "rationale": rationale,
        "output_path": str(output_path),
        "v1_metrics": {
            "accuracy": v1_overall.get("accuracy"),
            "macro_f1": v1_overall.get("macro_f1"),
            "log_loss": v1_overall.get("log_loss"),
        },
        "v2_metrics": {
            "accuracy": v2_overall.get("accuracy"),
            "macro_f1": v2_overall.get("macro_f1"),
            "log_loss": v2_overall.get("log_loss"),
        },
    }
