# Segment-Aware Hybrid Ensemble: Implementation Guide

**Status:** Phase 4, Refinement 2 (Post-Uncertainty Analysis)  
**Date:** April 8, 2026  
**Context:** Improving on the 0.45 global threshold → segment-conditional strategy

---

## Executive Summary

The original `HybridDrawOverrideEnsemble` used **one threshold for all data** (0.45).

**Problem:** This was too broad. The ensemble overrode the generalist even on:

- High-confidence World Cup matches (where generalist is strong)
- Matches with clear home advantage
- Situations where uncertainty was low but the specialist still pushed for draws

**Solution:** `SegmentAwareHybridDrawOverrideEnsemble` allows:

- ✅ **Different thresholds per segment** (tournament type, match tier)
- ✅ **Selective specialist activation** (only where generalist struggles)
- ✅ **Per-segment validation metrics** (identifying which segments benefit)
- ✅ **Conservative by default** (narrower uncertainty bands)

---

## Key Concepts

### 1. Segment Detection

A _segment_ is a logical grouping of fixtures based on contextual properties.

```python
def detect_tournament_segment(row: pd.Series) -> str | None:
    """
    Detect segment based on tournament type.

    Examples:
    - "World Cup 2026" → segment "worldcup"
    - "Friendly" → segment "friendlies"
    - "Copa America" → segment "continental"
    """
    tournament = row.get("tournament")

    if tournament and "World Cup" in tournament:
        return "worldcup"
    elif tournament == "Friendly":
        return "friendlies"
    elif tournament in ["Copa America", "EURO"]:
        return "continental"
    else:
        return None  # Not in a configured segment
```

### 2. Segment Configuration

Each segment has **specific thresholds** reflecting domain knowledge.

```python
from src.modeling.hybrid_ensemble_segment_aware import SegmentConfig

# Friendlies: Generalist is often uncertain → specialist can help
friendlies_config = SegmentConfig(
    segment_id="friendlies",
    uncertainty_threshold=0.35,  # NARROW band (activate at 35% uncertainty)
    draw_conviction_threshold=0.55,  # Specialist must be fairly confident
    description="Friendly matches with high uncertainty"
)

# World Cup: Generalist is strong → avoid interference
worldcup_config = SegmentConfig(
    segment_id="worldcup",
    uncertainty_threshold=0.50,  # WIDE band (only activate at 50% uncertainty)
    draw_conviction_threshold=0.60,  # Specialist must be very confident
    description="World Cup matches"
)

# Continental (Copa, EURO): Medium confidence
continental_config = SegmentConfig(
    segment_id="continental",
    uncertainty_threshold=0.42,
    draw_conviction_threshold=0.57,
    description="Continental tournaments (Copa, EURO, etc.)"
)
```

### 3. Override Logic

The specialist ONLY overrides when **ALL** conditions are met:

```
IF fixture in segment S:
    unc_thresh, conv_thresh = config[S]
ELSE:
    unc_thresh, conv_thresh = defaults

IF generalist.max_probability < unc_thresh
   AND specialist.predicts_draw
   AND specialist.p_draw >= conv_thresh:

    USE specialist prediction
ELSE:
    USE generalist prediction
```

---

## Implementation Pattern

### Step 1: Import & Define Segment Detector

```python
import pandas as pd
from src.modeling.hybrid_ensemble_segment_aware import (
    SegmentAwareHybridDrawOverrideEnsemble,
    SegmentConfig,
)

def detect_segment(row: pd.Series) -> str | None:
    """
    Route fixtures to segments based on tournament type.

    Returns segment_id if fixture belongs to configured segment, else None.
    """
    tournament = row.get("tournament")
    if not tournament:
        return None

    if "World Cup" in tournament:
        return "worldcup"
    elif tournament == "Friendly":
        return "friendlies"
    elif tournament in ["Copa America", "EURO", "Africa Cup", "Asian Cup"]:
        return "continental"

    return None  # Not in a special segment
```

### Step 2: Create Segment Configurations

```python
# Configure each segment with domain knowledge
segment_configs = {
    "friendlies": SegmentConfig(
        segment_id="friendlies",
        uncertainty_threshold=0.35,  # Activate at lower confidence
        draw_conviction_threshold=0.55,
        min_samples_for_override=10,
    ),
    "worldcup": SegmentConfig(
        segment_id="worldcup",
        uncertainty_threshold=0.50,  # Only activate when very uncertain
        draw_conviction_threshold=0.60,
        min_samples_for_override=20,
    ),
    "continental": SegmentConfig(
        segment_id="continental",
        uncertainty_threshold=0.42,
        draw_conviction_threshold=0.57,
        min_samples_for_override=15,
    ),
}
```

### Step 3: Instantiate Ensemble

```python
# Assuming generalist & specialist are already trained XGBClassifier, etc.
from src.modeling.hybrid_ensemble_segment_aware import SegmentAwareHybridDrawOverrideEnsemble

ensemble = SegmentAwareHybridDrawOverrideEnsemble(
    generalist_estimator=trained_generalist,
    specialist_estimator=trained_specialist,

    # Defaults (fallback for unmatched segments)
    default_uncertainty_threshold=0.45,
    default_draw_conviction_threshold=0.55,

    # Segment-specific routing
    segment_configs=segment_configs,
    segment_detector_fn=detect_segment,

    # Draw-weighting in specialist training
    specialist_draw_weight_multiplier=1.5,  # Friendlies-focused specialist
)

# Train both generalist & specialist with sample weights
ensemble.fit(X_train, y_train, sample_weight=computed_weights)
```

### Step 4: Make Predictions

```python
# X_test must have "tournament" column for detector to work
y_pred = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)  # (n_samples, 3) array
```

### Step 5: Evaluate Per-Segment

```python
segment_stats = ensemble.segment_statistics(X_test, y_test)

# Example output:
# {
#     "friendlies": {
#         "n_samples": 48,
#         "override_rate": 0.125,  # 12.5% of friendlies use specialist
#         "accuracy": 0.62,
#         "draw_accuracy": 0.55,  # Recall: can specialist catch draws?
#         "draw_precision": 0.67,  # If it predicts draw, is it right?
#     },
#     "worldcup": {
#         "n_samples": 52,
#         "override_rate": 0.02,  # Only 2% of WC matches use specialist
#         "accuracy": 0.77,
#         "draw_accuracy": 0.40,
#         "draw_precision": 0.80,
#     },
#     ...
# }

# Interpretation:
# - If friendlies.override_rate is HIGH but accuracy is LOW → thresholds too loose
# - If worldcup.override_rate is HIGH → specialist is interfering (expected, increase thresholds)
# - If any segment has LOW draw_precision → specialist is overpredicting draws (increase conv_thresh)
```

---

## Example: Training a Segment-Aware Model

```python
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from src.modeling.hybrid_ensemble_segment_aware import (
    SegmentAwareHybridDrawOverrideEnsemble,
    SegmentConfig,
)

# 1. Load data
X_train = pd.read_csv("data/gold/features_dataset.csv")
y_train = X_train["match_outcome"]
X_train = X_train.drop(columns=["match_outcome"])

# 2. Segment detector
def detect_segment(row: pd.Series) -> str | None:
    tournament = row.get("tournament", "")
    if "World Cup" in tournament:
        return "worldcup"
    elif tournament == "Friendly":
        return "friendlies"
    elif tournament in ["Copa America", "EURO"]:
        return "continental"
    return None

# 3. Create segment configs
segment_configs = {
    "friendlies": SegmentConfig(
        segment_id="friendlies",
        uncertainty_threshold=0.35,
        draw_conviction_threshold=0.55,
    ),
    "worldcup": SegmentConfig(
        segment_id="worldcup",
        uncertainty_threshold=0.50,
        draw_conviction_threshold=0.60,
    ),
    "continental": SegmentConfig(
        segment_id="continental",
        uncertainty_threshold=0.42,
        draw_conviction_threshold=0.57,
    ),
}

# 4. Build models
generalist = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
specialist = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)

# 5. Create ensemble
ensemble = SegmentAwareHybridDrawOverrideEnsemble(
    generalist_estimator=generalist,
    specialist_estimator=specialist,
    segment_configs=segment_configs,
    segment_detector_fn=detect_segment,
    specialist_draw_weight_multiplier=1.5,
)

# 6. Fit
ensemble.fit(X_train, y_train)

# 7. Evaluate on validation set
X_val, y_val = load_validation_data()
stats = ensemble.segment_statistics(X_val, y_val)
print("Per-segment validation metrics:")
for segment_id, metrics in stats.items():
    print(f"  {segment_id}: accuracy={metrics['accuracy']:.3f}, "
          f"override_rate={metrics['override_rate']:.1%}")
```

---

## Decision Framework: When to Adjust Thresholds

### Problem: Too Many Specialist Overrides

**Symptoms:**

- `override_rate` > 20% in a segment
- Overall accuracy DOWN compared to generalist-only

**Solution:** Increase thresholds (make specialist activation harder)

```python
# Before
friendlies_config = SegmentConfig(
    segment_id="friendlies",
    uncertainty_threshold=0.35,  # ← Too low (activates too often)
    draw_conviction_threshold=0.55,
)

# After
friendlies_config = SegmentConfig(
    segment_id="friendlies",
    uncertainty_threshold=0.40,  # ← More selective
    draw_conviction_threshold=0.58,
)
```

### Problem: Too Few Overrides / Specialist Not Helping

**Symptoms:**

- `override_rate` < 2% in a segment where generalist is weak
- Accuracy could be improved by specialist activation

**Solution:** Decrease thresholds (make specialist activation easier)

```python
# Before
continental_config = SegmentConfig(
    segment_id="continental",
    uncertainty_threshold=0.48,  # ← Too high (specialist never activates)
    draw_conviction_threshold=0.60,
)

# After
continental_config = SegmentConfig(
    segment_id="continental",
    uncertainty_threshold=0.40,  # ← More permissive
    draw_conviction_threshold=0.55,
)
```

### Problem: Specialist Predicts Too Many Draws

**Symptoms:**

- `draw_precision` is LOW (e.g., 0.30 = 70% false positives)
- Specialist is overconfident about draws

**Solution:** Increase `draw_conviction_threshold`

```python
# Before
SegmentConfig(
    segment_id="friendly",
    uncertainty_threshold=0.35,
    draw_conviction_threshold=0.50,  # ← Too permissive
)

# After
SegmentConfig(
    segment_id="friendly",
    uncertainty_threshold=0.35,
    draw_conviction_threshold=0.60,  # ← Require higher confidence
)
```

---

## Monitoring & Iteration

### Phase 1: Baseline (Initial Segment Thresholds)

Based on domain knowledge:

- **Friendlies:** Generalist often uncertain → aggressive specialist use
- **World Cup:** Generalist strong → minimal specialist use
- **Continental:** Middle ground

### Phase 2: Validation (Per-Segment Metrics)

```python
# After 100+ test predictions, compute segment stats
stats = ensemble.segment_statistics(X_validation, y_validation)

# Review:
# - Is override_rate reasonable per segment?
# - Are metrics improving vs. generalist-only?
# - Which segments benefit from specialist?
```

### Phase 3: Refinement (Threshold Tuning)

Use validation metrics to narrow bands:

- If a segment is over-overriding, increase thresholds
- If a segment needs help, decrease thresholds
- A/B test refined configs against production baseline

### Phase 4: Production (Segment-Aware Serving)

Deploy ensemble with segment awareness:

- Include segment detection in preprocessing pipeline
- Log `override_flag` in inference logs for monitoring
- Track segment-specific metrics in dashboard

---

## Backward Compatibility

The original `HybridDrawOverrideEnsemble` still exists and works. To migrate:

```python
# Old (single global threshold)
from src.modeling.hybrid_ensemble import HybridDrawOverrideEnsemble

ensemble_old = HybridDrawOverrideEnsemble(
    generalist_estimator=gen,
    specialist_estimator=spec,
    uncertainty_threshold=0.45,  # One threshold for all
)

# New (segment-aware)
from src.modeling.hybrid_ensemble_segment_aware import (
    SegmentAwareHybridDrawOverrideEnsemble,
    SegmentConfig,
)

# If you want old behavior, just use default thresholds with no segments:
ensemble_new = SegmentAwareHybridDrawOverrideEnsemble(
    generalist_estimator=gen,
    specialist_estimator=spec,
    default_uncertainty_threshold=0.45,
    # segment_configs=None (no segment routing)
    # segment_detector_fn=None
)
# Behavior will be identical to old version
```

---

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_hybrid_ensemble_segment_aware.py -v

# Coverage:
# ✓ Segment config validation
# ✓ Fit/predict without segmentation (backward compat)
# ✓ Segment detection & routing
# ✓ Override mask computation
# ✓ Per-segment statistics
# ✓ Error handling
```

---

## FAQ

**Q: What if my data doesn't have a "tournament" column?**

A: Provide any segment detector that matches your schema. Examples:

```python
def detect_segment(row: pd.Series) -> str | None:
    # Can route on any column
    match_type = row.get("match_type")
    if match_type == "qualifier":
        return "qualifiers"
    ...
```

**Q: Should I use different specialist models per segment?**

A: Not recommended initially. The current design uses **one specialist** trained with draw-weighted samples, serving all segments. Per-segment specialists would add complexity and training cost. Validate the current approach first.

**Q: How many samples do I need per segment to trust the metrics?**

A: Minimum 30-50 samples per segment. Below that, override rate might be noisy. Use `min_samples_for_override` in SegmentConfig to enforce thresholds.

**Q: Can I have overlapping segments?**

A: No. The detector should return at most one segment per row. If you need hierarchy (e.g., "international_friendly" vs "club_friendly"), flatten it into distinct segment_ids.

---

## References

- **Original analysis:** `PHASE_3B_SENIOR_FEATURES_SUMMARY.md` (0.45 threshold closure)
- **Hybrid design:** `hybrid_ensemble.py` (base class, now deprecated in favor of this)
- **Tests:** `tests/test_hybrid_ensemble_segment_aware.py`
- **Feature source:** `src/modeling/hybrid_ensemble_segment_aware.py`
