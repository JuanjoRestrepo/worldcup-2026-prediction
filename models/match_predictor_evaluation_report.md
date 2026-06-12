# Model Evaluation Report

## Overall
- Selected model: `seg_hybrid_auto_tuned` (SegmentAwareHybridDrawOverrideEnsemble)
- Deployed variant: `temperature`
- Accuracy: `0.5709`
- Macro F1: `0.5183`
- Weighted F1: `0.5660`
- Balanced accuracy: `0.5260`
- MCC: `0.3259`
- Log loss: `0.9066`
- ECE: `0.0290`

## Draw Diagnostics
- Precision: `0.2939`
- Recall: `0.2473`
- F1: `0.2686`

## Top Candidate Search Results
- `seg_hybrid_auto_tuned` (segment_aware_hybrid): rank=1, macro_f1=0.4923, draw_f1=0.2835, log_loss=0.9596
- `hybrid_override_u0.42_d0.5` (hybrid_draw_override_ensemble): rank=2, macro_f1=0.4928, draw_f1=0.2956, log_loss=0.9714
- `xgboost_n400_d4_lr0.03_lambda2.0_draw1` (xgboost): rank=3, macro_f1=0.4923, draw_f1=0.2682, log_loss=0.9507
- `hybrid_override_u0.48_d0.5` (hybrid_draw_override_ensemble): rank=4, macro_f1=0.4919, draw_f1=0.3310, log_loss=0.9946
- `hybrid_override_u0.42_d0.6` (hybrid_draw_override_ensemble): rank=5, macro_f1=0.4900, draw_f1=0.2729, log_loss=0.9581

## Competition Segments
- `World Cup`: rows=1834, macro_f1=0.5426, log_loss=0.8246
- `Friendly`: rows=1677, macro_f1=0.4840, log_loss=0.9642
- `Other`: rows=1504, macro_f1=0.4623, log_loss=0.9773
- `Qualifier`: rows=1023, macro_f1=0.5313, log_loss=0.8460
- `Continental`: rows=419, macro_f1=0.5191, log_loss=0.9285

## Confederation Segments
- `UEFA`: rows=2042, macro_f1=0.5314, draw_recall=0.2179
- `CAF`: rows=1498, macro_f1=0.4960, draw_recall=0.2419
- `AFC`: rows=1298, macro_f1=0.5086, draw_recall=0.2436
- `CONCACAF`: rows=1017, macro_f1=0.5308, draw_recall=0.3080
- `CONMEBOL`: rows=355, macro_f1=0.5211, draw_recall=0.3049

## Highest-Confidence Errors
- 2024-06-07 | Romania vs Liechtenstein | actual=draw predicted=home_win conf=0.97
- 2021-03-30 | Senegal vs Eswatini | actual=draw predicted=home_win conf=0.94
- 2022-11-17 | Saint Lucia vs San Marino | actual=draw predicted=home_win conf=0.93
- 2021-09-02 | Italy vs Bulgaria | actual=draw predicted=home_win conf=0.93
- 2024-03-21 | South Africa vs Andorra | actual=draw predicted=home_win conf=0.92
- 2026-03-26 | Tanzania vs Liechtenstein | actual=away_win predicted=home_win conf=0.90
- 2021-03-30 | Turkey vs Latvia | actual=draw predicted=home_win conf=0.90
- 2021-03-31 | Ukraine vs Kazakhstan | actual=draw predicted=home_win conf=0.89
- 2024-11-14 | France vs Israel | actual=draw predicted=home_win conf=0.89
- 2023-09-07 | Austria vs Moldova | actual=draw predicted=home_win conf=0.89

## Artifact Files
- Confusion matrix: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_confusion_matrix.png`
- Calibration curves: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_calibration_curves.png`
- JSON report: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_evaluation_report.json`