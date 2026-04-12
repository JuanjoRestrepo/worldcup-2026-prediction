# Segment-Aware Hybrid Ensemble: Integration into Temporal Search

## Goal

Integrate the `SegmentAwareHybridDrawOverrideEnsemble` as a formal candidate family in the temporal backtesting pipeline (`train.py`), so the segment-conditioned hybrid competes head-to-head against the current champion (`logistic_c2_draw1.2`) under the same multicriteria ranking framework.

The hypothesis: **A hybrid that activates its draw specialist ONLY in segments where the generalist suffers** (Friendlies, certain continental matches) while leaving World Cup / Qualifier predictions untouched should achieve better risk-adjusted performance than both the global-threshold hybrid and the standalone generalist.

---

## Background & Rationale

| Aspect | Current State | Target State |
|--------|--------------|--------------|
| **Global hybrid** (`HybridDrawOverrideEnsemble`) | ✅ Integrated in `train.py`, rank #5 best | Keep as baseline |
| **Segment-aware hybrid** (`SegmentAwareHybridDrawOverrideEnsemble`) | ✅ Code exists, tests pass, **NOT** in temporal search | ⬆ Integrate as candidate family |
| **Tournament column in features** | `tournament` column exists in gold dataset but excluded from `NON_FEATURE_COLUMNS` | Pass `tournament` to segment detector via X metadata |
| **Domain knowledge** | 0.45 global threshold was too broad | Narrower, segment-specific bands from SEGMENT_AWARE_HYBRID_GUIDE.md |

### Statistical Rationale

The dataset has **31,957 rows** with the following segment distribution:
- **Friendlies**: ~10,600 (33%) — highest uncertainty, generalist often weak
- **Qualifiers**: ~6,900+ (22%) — moderate structure, generalist decent
- **World Cup**: ~552 (1.7%) — generalist strong, specialist should NOT interfere
- **Continental**: ~2,500+ (8%) — mixed signals, selective specialist use
- **Other**: ~11,300 (35%) — unconfigured, fallback to default thresholds

The draw class represents **23.5%** of outcomes (7,513/31,957). This prior imbalance is exactly the scenario where segment-conditional routing can help: overriding into draw only when uncertainty is genuinely high AND within segments where draw rates differ from the global average.

---

## Proposed Changes

### Component 1: Training Pipeline

#### [MODIFY] [train.py](file:///c:/Users/restr/Desktop/worldcup-2026-prediction/src/modeling/train.py)

**Changes:**
1. **Import** `SegmentAwareHybridDrawOverrideEnsemble` and `SegmentConfig` from `hybrid_ensemble_segment_aware.py`
2. **Add a segment detector function** (`_tournament_segment_detector`) that maps the `tournament` column to segment IDs
3. **Add segment-aware hybrid variants** to `_build_candidate_specs()` — a compact grid of 4–6 variants with different segment threshold combinations:

   | Variant | Friendlies unc/conv | World Cup unc/conv | Continental unc/conv | Qualifiers unc/conv | Default unc/conv |
   |---------|---------------------|--------------------|----------------------|---------------------|------------------|
   | `seg_hybrid_conservative` | 0.38/0.58 | 0.52/0.62 | 0.44/0.58 | 0.48/0.60 | 0.45/0.55 |
   | `seg_hybrid_friendlies_focus` | 0.32/0.52 | 0.55/0.65 | 0.46/0.58 | 0.50/0.60 | 0.48/0.58 |
   | `seg_hybrid_balanced` | 0.36/0.55 | 0.50/0.60 | 0.42/0.56 | 0.46/0.58 | 0.44/0.55 |
   | `seg_hybrid_narrow_band` | 0.40/0.60 | 0.55/0.65 | 0.48/0.60 | 0.50/0.62 | 0.48/0.58 |

4. **Handle the `tournament` column pass-through**: The segment-aware ensemble needs the `tournament` column during `predict_proba` for segment routing, but the model should NOT train on it as a feature. The existing `SegmentAwareHybridDrawOverrideEnsemble` already handles this via `feature_names_in_` — it filters X to only trained columns for model inference, but uses full X for segment detection.

   **Challenge**: In `evaluate_candidates_with_backtesting`, `X` only contains `feature_columns` (numeric). We need to pass `tournament` alongside for the segment detector.
   
   **Solution**: We'll extend the segment-aware ensemble's `predict_proba` to accept an optional `X_metadata` param. However, a much cleaner approach that works within the current architecture: **temporarily include `tournament` in the DataFrame passed to the ensemble** and let `feature_names_in_` filtering handle the separation. The ensemble already does `X_fit = X[self.feature_names_in_]` during prediction — it will automatically strip `tournament` for model inference and use full X for segment detection. We just need to make the backtesting code pass `tournament` alongside. 

   **Implementation**: Modify `evaluate_candidates_with_backtesting` to accept an optional `metadata_columns` list. When provided, these columns are concatenated to X during evaluation but excluded from training feature names. This is architecturally clean and respects the separation of concerns.

---

### Component 2: Evaluation Pipeline (Metadata Pass-Through)

#### [MODIFY] [evaluation.py](file:///c:/Users/restr/Desktop/worldcup-2026-prediction/src/modeling/evaluation.py)

**Changes:**
1. Add `metadata_columns: list[str] | None = None` parameter to `evaluate_candidates_with_backtesting`
2. When `metadata_columns` is not None, include those columns from `train_df` in X during `predict_proba` and `predict` calls, but **exclude** them from training
3. This is minimal and backward-compatible — existing callers pass `metadata_columns=None` and behavior is unchanged

---

### Component 3: Tests

#### [NEW] [test_segment_aware_training_integration.py](file:///c:/Users/restr/Desktop/worldcup-2026-prediction/tests/test_segment_aware_training_integration.py)

Test coverage:
- `_tournament_segment_detector` maps tournament strings correctly
- Segment-aware candidate specs are built correctly with expected hyperparameters
- Segment-aware ensemble can fit/predict through the full pipeline
- End-to-end: segment-aware candidate runs through backtesting without error
- Metadata column pass-through works correctly in evaluation

---

## User Review Required

> [!IMPORTANT]
> **Segment threshold grid**: The 4 variants above are based on domain knowledge from `SEGMENT_AWARE_HYBRID_GUIDE.md` and the historical 0.45 analysis. The idea is:
> - **Friendlies**: Most aggressive specialist activation (low uncertainty threshold) because the generalist is genuinely uncertain here
> - **World Cup**: Most conservative (high threshold) — generalist performance is strong, don't mess with it
> - **Qualifiers/Continental**: Middle ground, leaning conservative
> 
> Do you want to adjust these ranges, or should I proceed with these as starting points?

> [!WARNING]
> **Training time impact**: Adding 4 segment-aware hybrid variants to the 21+ existing candidates will increase backtesting time. Each variant trains 2 sub-models (generalist + specialist) per fold × 5 folds = 10 fits per variant × 4 variants = **40 additional model fits**. Based on current pipeline speed (~30s for 113 tests), this should add **~2–4 minutes** to the full training run. This is acceptable for a one-time retrain before deploy.

---

## Open Questions

1. **Specialist base model**: The current global hybrid uses `LogisticRegression(C=2.0)` as the generalist and `TwoStageDrawClassifier` as the specialist. Should the segment-aware hybrid use the same composition, or should we test with a simpler `LogisticRegression` specialist too? I recommend **keeping the same composition** for fair comparison, and adding one variant with a simpler specialist as an experiment.

2. **Qualifiers segment**: The feature dataset has `"FIFA World Cup qualification"`, `"UEFA Euro qualification"`, `"African Cup of Nations qualification"` etc. Should all these map to the `"qualifiers"` segment, or should we split "WC qualifiers" vs "continental qualifiers"? I recommend **unified `qualifiers` segment** initially — we can split later if metrics suggest differential performance.

---

## Verification Plan

### Automated Tests
```bash
# 1. Full lint check
uv run ruff check src/modeling/ tests/

# 2. Type checking
uv run mypy src/modeling/train.py src/modeling/evaluation.py src/modeling/hybrid_ensemble_segment_aware.py

# 3. Existing + new tests
uv run python -m pytest tests/ -x --tb=short -q

# 4. Full retraining (validates end-to-end pipeline with new candidates)
uv run python -m src.modeling.train
```

### Manual Verification
- Review the training metrics JSON for segment-aware candidates
- Compare segment-aware hybrid rank vs. global hybrid rank vs. generalist champion
- Inspect segment-specific override rates from the training output
