# Model Evaluation Report

## Overall
- Selected model: `hybrid_override_u0.45_d0.5` (HybridDrawOverrideEnsemble)
- Deployed variant: `uncalibrated`
- Accuracy: `0.5419`
- Macro F1: `0.5096`
- Weighted F1: `0.5491`
- Balanced accuracy: `0.5113`
- MCC: `0.2933`
- Log loss: `0.9533`
- ECE: `0.0726`

## Draw Diagnostics
- Precision: `0.2846`
- Recall: `0.3293`
- F1: `0.3053`

## Top Candidate Search Results
- `hybrid_override_u0.45_d0.5` (hybrid_draw_override_ensemble): rank=1, macro_f1=0.4873, draw_f1=0.3207, log_loss=1.0025
- `hybrid_override_u0.42_d0.5` (hybrid_draw_override_ensemble): rank=2, macro_f1=0.4868, draw_f1=0.2989, log_loss=0.9890
- `seg_hybrid_auto_tuned` (segment_aware_hybrid): rank=3, macro_f1=0.4848, draw_f1=0.2834, log_loss=0.9752
- `hybrid_override_u0.48_d0.5` (hybrid_draw_override_ensemble): rank=4, macro_f1=0.4839, draw_f1=0.3338, log_loss=1.0135
- `logistic_c2_draw1` (logistic_regression): rank=5, macro_f1=0.4826, draw_f1=0.2738, log_loss=0.9728

## Competition Segments
- `World Cup`: rows=1869, macro_f1=0.5448, log_loss=0.8612
- `Friendly`: rows=1557, macro_f1=0.4363, log_loss=1.0463
- `Other`: rows=1518, macro_f1=0.4647, log_loss=1.0276
- `Qualifier`: rows=1064, macro_f1=0.5252, log_loss=0.8658
- `Continental`: rows=419, macro_f1=0.5185, log_loss=0.9714

## Confederation Segments
- `UEFA`: rows=2022, macro_f1=0.5360, draw_recall=0.3326
- `CAF`: rows=1494, macro_f1=0.4887, draw_recall=0.2800
- `AFC`: rows=1322, macro_f1=0.5086, draw_recall=0.3381
- `CONCACAF`: rows=1028, macro_f1=0.5095, draw_recall=0.4027
- `CONMEBOL`: rows=346, macro_f1=0.4924, draw_recall=0.3690

## Highest-Confidence Errors
- 2024-06-07 | Romania vs Liechtenstein | actual=draw predicted=home_win conf=0.96
- 2022-11-17 | Saint Lucia vs San Marino | actual=draw predicted=home_win conf=0.93
- 2024-03-21 | South Africa vs Andorra | actual=draw predicted=home_win conf=0.90
- 2021-09-02 | Italy vs Bulgaria | actual=draw predicted=home_win conf=0.89
- 2021-03-30 | Senegal vs Eswatini | actual=draw predicted=home_win conf=0.89
- 2021-03-30 | Turkey vs Latvia | actual=draw predicted=home_win conf=0.88
- 2024-06-06 | Gibraltar vs Wales | actual=draw predicted=away_win conf=0.87
- 2019-09-08 | Greece vs Liechtenstein | actual=draw predicted=home_win conf=0.87
- 2026-03-26 | Tanzania vs Liechtenstein | actual=away_win predicted=home_win conf=0.87
- 2021-09-08 | Armenia vs Liechtenstein | actual=draw predicted=home_win conf=0.86

## Artifact Files
- Confusion matrix: `models\match_predictor_confusion_matrix.png`
- Calibration curves: `models\match_predictor_calibration_curves.png`
- JSON report: `models\match_predictor_evaluation_report.json`