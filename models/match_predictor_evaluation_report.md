# Model Evaluation Report

## Overall
- Selected model: `seg_hybrid_auto_tuned` (SegmentAwareHybridDrawOverrideEnsemble)
- Deployed variant: `uncalibrated`
- Accuracy: `0.5674`
- Macro F1: `0.5192`
- Weighted F1: `0.5647`
- Balanced accuracy: `0.5269`
- MCC: `0.3242`
- Log loss: `0.9102`
- ECE: `0.0358`

## Draw Diagnostics
- Precision: `0.2919`
- Recall: `0.2595`
- F1: `0.2747`

## Top Candidate Search Results
- `seg_hybrid_auto_tuned` (segment_aware_hybrid): rank=1, macro_f1=0.4975, draw_f1=0.2964, log_loss=0.9597
- `logistic_c2_draw1` (logistic_regression): rank=2, macro_f1=0.4945, draw_f1=0.2796, log_loss=0.9565
- `logistic_c2_draw1.2` (logistic_regression): rank=3, macro_f1=0.4945, draw_f1=0.2796, log_loss=0.9565
- `xgboost_n400_d4_lr0.03_lambda2.0_draw1` (xgboost): rank=4, macro_f1=0.4930, draw_f1=0.2829, log_loss=0.9543
- `hybrid_override_u0.42_d0.5` (hybrid_draw_override_ensemble): rank=5, macro_f1=0.4961, draw_f1=0.2988, log_loss=0.9705

## Competition Segments
- `World Cup`: rows=1833, macro_f1=0.5471, log_loss=0.8315
- `Friendly`: rows=1679, macro_f1=0.4765, log_loss=0.9688
- `Other`: rows=1504, macro_f1=0.4658, log_loss=0.9780
- `Qualifier`: rows=1023, macro_f1=0.5397, log_loss=0.8463
- `Continental`: rows=419, macro_f1=0.5246, log_loss=0.9327

## Confederation Segments
- `UEFA`: rows=2042, macro_f1=0.5329, draw_recall=0.2244
- `CAF`: rows=1498, macro_f1=0.4952, draw_recall=0.2643
- `AFC`: rows=1298, macro_f1=0.5218, draw_recall=0.2727
- `CONCACAF`: rows=1018, macro_f1=0.5165, draw_recall=0.2812
- `CONMEBOL`: rows=355, macro_f1=0.5280, draw_recall=0.3293

## Highest-Confidence Errors
- 2024-06-07 | Romania vs Liechtenstein | actual=draw predicted=home_win conf=0.96
- 2021-03-30 | Senegal vs Eswatini | actual=draw predicted=home_win conf=0.93
- 2021-09-02 | Italy vs Bulgaria | actual=draw predicted=home_win conf=0.92
- 2022-11-17 | Saint Lucia vs San Marino | actual=draw predicted=home_win conf=0.91
- 2024-03-21 | South Africa vs Andorra | actual=draw predicted=home_win conf=0.91
- 2024-11-14 | France vs Israel | actual=draw predicted=home_win conf=0.90
- 2021-03-31 | Ukraine vs Kazakhstan | actual=draw predicted=home_win conf=0.88
- 2021-03-30 | Turkey vs Latvia | actual=draw predicted=home_win conf=0.87
- 2026-03-26 | Tanzania vs Liechtenstein | actual=away_win predicted=home_win conf=0.86
- 2023-09-07 | Austria vs Moldova | actual=draw predicted=home_win conf=0.85

## Artifact Files
- Confusion matrix: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_confusion_matrix.png`
- Calibration curves: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_calibration_curves.png`
- JSON report: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_evaluation_report.json`