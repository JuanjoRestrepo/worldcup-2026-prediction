# Model Evaluation Report

## Overall
- Selected model: `hybrid_override_u0.45_d0.5` (HybridDrawOverrideEnsemble)
- Deployed variant: `temperature`
- Accuracy: `0.5455`
- Macro F1: `0.5189`
- Weighted F1: `0.5566`
- Balanced accuracy: `0.5212`
- MCC: `0.3063`
- Log loss: `0.9442`
- ECE: `0.0554`

## Draw Diagnostics
- Precision: `0.2846`
- Recall: `0.3619`
- F1: `0.3186`

## Top Candidate Search Results
- `hybrid_override_u0.45_d0.5` (hybrid_draw_override_ensemble): rank=1, macro_f1=0.4972, draw_f1=0.3212, log_loss=0.9830
- `hybrid_override_u0.42_d0.5` (hybrid_draw_override_ensemble): rank=2, macro_f1=0.4971, draw_f1=0.3041, log_loss=0.9711
- `seg_hybrid_auto_tuned` (segment_aware_hybrid): rank=3, macro_f1=0.4951, draw_f1=0.2900, log_loss=0.9599
- `logistic_c0.5_draw1` (logistic_regression): rank=4, macro_f1=0.4937, draw_f1=0.2808, log_loss=0.9576
- `hybrid_override_u0.48_d0.5` (hybrid_draw_override_ensemble): rank=5, macro_f1=0.4938, draw_f1=0.3341, log_loss=0.9942

## Competition Segments
- `World Cup`: rows=1903, macro_f1=0.5617, log_loss=0.8649
- `Friendly`: rows=1654, macro_f1=0.4691, log_loss=1.0088
- `Other`: rows=1492, macro_f1=0.4647, log_loss=1.0148
- `Qualifier`: rows=1003, macro_f1=0.5367, log_loss=0.8759
- `Continental`: rows=428, macro_f1=0.4941, log_loss=0.9619

## Confederation Segments
- `UEFA`: rows=2064, macro_f1=0.5354, draw_recall=0.3269
- `CAF`: rows=1494, macro_f1=0.5016, draw_recall=0.3550
- `AFC`: rows=1278, macro_f1=0.5201, draw_recall=0.3957
- `CONCACAF`: rows=1021, macro_f1=0.5196, draw_recall=0.4110
- `CONMEBOL`: rows=368, macro_f1=0.5109, draw_recall=0.4070

## Highest-Confidence Errors
- 2024-06-07 | Romania vs Liechtenstein | actual=draw predicted=home_win conf=0.95
- 2021-03-30 | Senegal vs Eswatini | actual=draw predicted=home_win conf=0.89
- 2024-03-21 | South Africa vs Andorra | actual=draw predicted=home_win conf=0.86
- 2021-09-02 | Italy vs Bulgaria | actual=draw predicted=home_win conf=0.86
- 2022-11-17 | Saint Lucia vs San Marino | actual=draw predicted=home_win conf=0.85
- 2021-03-31 | Ukraine vs Kazakhstan | actual=draw predicted=home_win conf=0.83
- 2023-09-07 | Austria vs Moldova | actual=draw predicted=home_win conf=0.83
- 2024-06-06 | Gibraltar vs Wales | actual=draw predicted=away_win conf=0.81
- 2024-11-14 | France vs Israel | actual=draw predicted=home_win conf=0.81
- 2024-06-07 | England vs Iceland | actual=away_win predicted=home_win conf=0.81

## Artifact Files
- Confusion matrix: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_confusion_matrix.png`
- Calibration curves: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_calibration_curves.png`
- JSON report: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_evaluation_report.json`