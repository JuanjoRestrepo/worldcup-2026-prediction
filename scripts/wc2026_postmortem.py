"""
WC 2026 Post-Tournament Model Analysis
=======================================
Runs our saved model artifact against all 104 WC 2026 matches and computes
per-stage accuracy, calibration, Brier score, and confusion matrices.
Then identifies systematic failure modes to guide model improvement.

Run:
    uv run python scripts/wc2026_postmortem.py
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

# Add project root to sys.path so pickled custom classes in backend module can be imported
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_CSV = PROJECT_ROOT / "data" / "raw" / "international_results.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

# Locate the latest model artifact
def find_latest_artifact() -> Path:
    candidates: list[Path] = []
    models_dir = PROJECT_ROOT / "models"
    if models_dir.exists():
        candidates.extend(sorted(models_dir.glob("*.joblib")))
    if ARTIFACT_DIR.exists():
        candidates.extend(sorted(ARTIFACT_DIR.glob("*.joblib")))
    candidates.extend(sorted(PROJECT_ROOT.glob("*.joblib")))
    if not candidates:
        raise FileNotFoundError(
            "No .joblib model artifact found. Run training first."
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# Load WC 2026 matches
# ---------------------------------------------------------------------------
def load_wc2026_matches() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    wc = df[
        (df["tournament"] == "FIFA World Cup")
        & (df["date"] >= "2026-06-01")
        & (df["date"] <= "2026-07-20")
    ].copy()

    # Derive actual outcome (from home team perspective)
    wc["actual_outcome"] = np.where(
        wc["home_score"] > wc["away_score"], 1,
        np.where(wc["home_score"] < wc["away_score"], -1, 0),
    )

    # Annotate tournament stage
    def stage(d: pd.Timestamp) -> str:
        if d <= pd.Timestamp("2026-06-27"):
            return "Group Stage"
        elif d <= pd.Timestamp("2026-07-04"):
            return "Round of 32"
        elif d <= pd.Timestamp("2026-07-07"):
            return "Round of 16"
        elif d <= pd.Timestamp("2026-07-11"):
            return "Quarter-Finals"
        elif d <= pd.Timestamp("2026-07-15"):
            return "Semi-Finals"
        else:
            return "Final / 3rd Place"

    wc["stage"] = wc["date"].apply(stage)
    logger.info("Loaded %d WC 2026 matches", len(wc))
    return wc.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Predict for all WC matches using the official prediction entrypoint
# ---------------------------------------------------------------------------
def predict_all(bundle: dict, wc: pd.DataFrame) -> pd.DataFrame:
    from backend.modeling.predict import predict_match_outcome

    results = []
    skipped = 0

    # Knockout rounds started 2026-07-02 (Round of 32 onward)
    KNOCKOUT_CUTOFF = pd.Timestamp("2026-07-02")

    # ── Pre-tournament snapshot cap ────────────────────────────────────────────
    # CRITICAL: We must use feature snapshots from BEFORE the tournament started.
    # If we use the actual match date, later knockout rounds will see ELO/form
    # features inflated by earlier WC results — causing 100% home-win confidence.
    # The WC group stage began 2026-06-12, so we cap at 2026-06-11 for all matches.
    PRE_WC_SNAPSHOT_DATE = pd.Timestamp("2026-06-11").date()

    for _, row in wc.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        neutral = bool(row["neutral"])
        tournament = str(row["tournament"])
        is_knockout = bool(row["date"] >= KNOCKOUT_CUTOFF)

        try:
            res = predict_match_outcome(
                home_team=home_team,
                away_team=away_team,
                tournament=tournament,
                neutral=neutral,
                # Use pre-tournament snapshot for all WC matches to prevent leakage
                match_date=PRE_WC_SNAPSHOT_DATE,
                log_inference=False,
                is_knockout=is_knockout,
            )
            prob_dict = res["class_probabilities"]

            # Map predicted_outcome string back to integer outcome (-1, 0, 1)
            outcome_map = {"home_win": 1, "draw": 0, "away_win": -1}
            pred_outcome = outcome_map.get(res["predicted_outcome"])

            results.append(
                {
                    "home_team": home_team,
                    "away_team": away_team,
                    "date": row["date"],
                    "stage": row["stage"],
                    "is_knockout": is_knockout,
                    "actual_outcome": row["actual_outcome"],
                    "predicted_outcome": pred_outcome,
                    "p_home_win": prob_dict.get("home_win", 0.0),
                    "p_draw": prob_dict.get("draw", 0.0),
                    "p_away_win": prob_dict.get("away_win", 0.0),
                }
            )
        except Exception as e:
            logger.warning("Error predicting %s vs %s: %s", home_team, away_team, e)
            skipped += 1
            results.append(
                {
                    "home_team": home_team,
                    "away_team": away_team,
                    "date": row["date"],
                    "stage": row["stage"],
                    "is_knockout": is_knockout,
                    "actual_outcome": row["actual_outcome"],
                    "predicted_outcome": None,
                    "p_home_win": None,
                    "p_draw": None,
                    "p_away_win": None,
                }
            )

    logger.info("Predicted %d matches (%d skipped)", len(results) - skipped, skipped)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Feature lookup from gold dataset
# ---------------------------------------------------------------------------
_GOLD_DF: pd.DataFrame | None = None

def _load_gold() -> pd.DataFrame:
    global _GOLD_DF
    if _GOLD_DF is None:
        gold_path = PROJECT_ROOT / "data" / "gold" / "features_dataset.csv"
        _GOLD_DF = pd.read_csv(gold_path, low_memory=False)
        _GOLD_DF["date"] = pd.to_datetime(_GOLD_DF["date"])
    return _GOLD_DF


def _get_feature_row(
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    feature_columns: list[str],
) -> dict | None:
    gold = _load_gold()

    # Most recent home/away rows for each team before the match date
    h = gold[
        (gold["homeTeam"] == home_team) & (gold["date"] < match_date)
    ].sort_values("date").tail(1)
    a = gold[
        (gold["awayTeam"] == away_team) & (gold["date"] < match_date)
    ].sort_values("date").tail(1)

    if h.empty or a.empty:
        return None

    row: dict = {}
    for col in feature_columns:
        if col in h.columns:
            row[col] = h.iloc[0][col]
        elif col in a.columns:
            row[col] = a.iloc[0][col]
        else:
            row[col] = 0.0

    return row


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyse(preds: pd.DataFrame) -> None:
    valid = preds.dropna(subset=["predicted_outcome"])
    y_true = valid["actual_outcome"].astype(int)
    y_pred = valid["predicted_outcome"].astype(int)
    proba_cols = ["p_home_win", "p_draw", "p_away_win"]

    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 65)
    logger.info("  WC 2026 POST-TOURNAMENT MODEL ANALYSIS")
    logger.info("=" * 65)
    logger.info("Matches evaluated: %d / %d", len(valid), len(preds))

    # Overall accuracy
    overall_acc = accuracy_score(y_true, y_pred)
    logger.info("\n📊 OVERALL ACCURACY:   %.1f%%", overall_acc * 100)

    # Per-stage accuracy
    logger.info("\n📌 ACCURACY BY STAGE:")
    for stage, grp in valid.groupby("stage"):
        acc = accuracy_score(
            grp["actual_outcome"].astype(int),
            grp["predicted_outcome"].astype(int),
        )
        logger.info("  %-20s  %d matches  %.1f%%", stage, len(grp), acc * 100)

    # Log-loss & Brier
    y_proba = valid[proba_cols].values
    ll = log_loss(
        y_true,
        y_proba,
        labels=[-1, 0, 1],
    )
    logger.info("\n📉 LOG-LOSS:           %.4f  (lower=better)", ll)

    # Brier score per class
    for i, label in enumerate([-1, 0, 1]):
        binary_true = (y_true == label).astype(int)
        bs = brier_score_loss(binary_true, y_proba[:, i])
        name = {1: "Home win", 0: "Draw", -1: "Away win"}[label]
        logger.info("    Brier %-12s %.4f", name, bs)

    # Confusion matrix
    logger.info("\n🔲 CONFUSION MATRIX (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
    labels_str = ["Away Win", "Draw    ", "Home Win"]
    logger.info("              %-10s %-10s %-10s", *labels_str)
    for i, row_label in enumerate(labels_str):
        logger.info("  %-12s %10d %10d %10d", row_label, *cm[i])

    # Classification report
    logger.info("\n📋 CLASSIFICATION REPORT:")
    report = classification_report(
        y_true, y_pred, target_names=["Away Win", "Draw", "Home Win"]
    )
    for line in report.split("\n"):
        logger.info("  %s", line)

    # -----------------------------------------------------------------------
    # Upset analysis — where high-confidence predictions were wrong
    logger.info("\n⚡ BIGGEST UPSETS (high-confidence wrong predictions):")
    valid2 = valid.copy()
    valid2["confidence"] = valid2[proba_cols].max(axis=1)
    valid2["correct"] = (valid2["actual_outcome"] == valid2["predicted_outcome"]).astype(int)
    upsets = valid2[valid2["correct"] == 0].sort_values("confidence", ascending=False).head(10)
    for _, r in upsets.iterrows():
        pred_label = {1: "Home Win", 0: "Draw", -1: "Away Win"}[int(r["predicted_outcome"])]
        actual_label = {1: "Home Win", 0: "Draw", -1: "Away Win"}[int(r["actual_outcome"])]
        logger.info(
            "  %s  %-22s vs %-22s  Predicted: %-10s (%.0f%%)  Actual: %s",
            r["date"].strftime("%Y-%m-%d"),
            r["home_team"],
            r["away_team"],
            pred_label,
            r["confidence"] * 100,
            actual_label,
        )

    # -----------------------------------------------------------------------
    # Draw prediction analysis
    logger.info("\n🤝 DRAW ANALYSIS:")
    actual_draws = (y_true == 0).sum()
    pred_draws = (y_pred == 0).sum()
    logger.info("  Actual draws:    %d (%.1f%%)", actual_draws, actual_draws / len(valid) * 100)
    logger.info("  Predicted draws: %d (%.1f%%)", pred_draws, pred_draws / len(valid) * 100)

    # -----------------------------------------------------------------------
    # Group stage vs. knockout accuracy
    gs = valid[valid["stage"] == "Group Stage"]
    ko = valid[valid["stage"] != "Group Stage"]
    if not gs.empty:
        logger.info(
            "\n🏟️  GROUP STAGE acc:    %.1f%%  (%d matches)",
            accuracy_score(gs["actual_outcome"].astype(int), gs["predicted_outcome"].astype(int)) * 100,
            len(gs),
        )
    if not ko.empty:
        logger.info(
            "⚔️   KNOCKOUT acc:       %.1f%%  (%d matches)",
            accuracy_score(ko["actual_outcome"].astype(int), ko["predicted_outcome"].astype(int)) * 100,
            len(ko),
        )

    # -----------------------------------------------------------------------
    # Model improvement suggestions
    logger.info("\n" + "=" * 65)
    logger.info("  IMPROVEMENT RECOMMENDATIONS")
    logger.info("=" * 65)
    draw_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
    home_win_precision = cm[2, 2] / cm[:, 2].sum() if cm[:, 2].sum() > 0 else 0
    if draw_recall < 0.35:
        logger.info(
            "\n1. DRAW UNDER-PREDICTION (recall=%.0f%%):\n"
            "   → Add draw-propensity features (H2H draw rate, possession entropy)\n"
            "   → Use draw-calibrated Platt scaling or isotonic regression\n"
            "   → Increase draw class weight during XGBoost training",
            draw_recall * 100,
        )
    if overall_acc < 0.50:
        logger.info(
            "\n2. OVERALL ACCURACY LOW (%.1f%%):\n"
            "   → Recalibrate ELO with tournament-specific K-factor\n"
            "   → Add 'is_knockout' binary feature — team motivation changes\n"
            "   → Incorporate squad availability / injury data",
            overall_acc * 100,
        )
    if home_win_precision < 0.55:
        logger.info(
            "\n3. HOME WIN OVER-PREDICTION (precision=%.0f%%):\n"
            "   → Most WC matches are neutral-site — home advantage signal is noisy\n"
            "   → Add 'host_country_proximity' feature for CONCACAF teams\n"
            "   → Separate ELO for neutral vs. home matches",
            home_win_precision * 100,
        )
    logger.info(
        "\n4. GENERAL:\n"
        "   → Retrain on post-2026-WC data (now included in international_results.csv)\n"
        "   → Add WC-specific tournament weight in training sample weighting\n"
        "   → Evaluate a goals-based (Dixon-Coles) model as an ensemble member"
    )

    # -----------------------------------------------------------------------
    # Save detailed prediction CSV
    out_path = PROJECT_ROOT / "data" / "raw" / "wc2026_predictions_vs_actuals.csv"
    valid2.to_csv(out_path, index=False)
    logger.info("\n💾 Full predictions saved → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    artifact_path = find_latest_artifact()
    logger.info("Loading model artifact: %s", artifact_path)
    bundle: dict = joblib.load(artifact_path)

    wc = load_wc2026_matches()
    preds = predict_all(bundle, wc)
    analyse(preds)


if __name__ == "__main__":
    main()
