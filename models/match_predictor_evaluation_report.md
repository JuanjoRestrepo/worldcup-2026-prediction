# Model Evaluation Report

## Overall
- Selected model: `logistic_c2_draw1.2` (LogisticRegression)
- Deployed variant: `uncalibrated`
- Accuracy: `0.5778`
- Macro F1: `0.5165`
- Weighted F1: `0.5666`
- Balanced accuracy: `0.5282`
- MCC: `0.3349`
- Log loss: `0.8952`
- ECE: `0.0289`

## Draw Diagnostics
- Precision: `0.2974`
- Recall: `0.2119`
- F1: `0.2475`

## Top Candidate Search Results
- `logistic_c2_draw1.2` (logistic_regression): rank=1, macro_f1=0.4912, draw_f1=0.2794, log_loss=0.9642
- `logistic_c2_draw1` (logistic_regression): rank=2, macro_f1=0.4912, draw_f1=0.2794, log_loss=0.9642
- `logistic_c0.5_draw1` (logistic_regression): rank=3, macro_f1=0.4912, draw_f1=0.2787, log_loss=0.9643
- `seg_hybrid_balanced` (segment_aware_hybrid): rank=4, macro_f1=0.4915, draw_f1=0.2827, log_loss=0.9671
- `logistic_c1_draw1` (logistic_regression): rank=5, macro_f1=0.4911, draw_f1=0.2792, log_loss=0.9643

## Competition Segments
- `World Cup`: rows=1827, macro_f1=0.5407, log_loss=0.8124
- `Other`: rows=1542, macro_f1=0.4776, log_loss=0.9666
- `Friendly`: rows=1500, macro_f1=0.4857, log_loss=0.9586
- `Qualifier`: rows=1066, macro_f1=0.5145, log_loss=0.8320
- `Continental`: rows=456, macro_f1=0.5001, log_loss=0.9243

## Confederation Segments
- `UEFA`: rows=1978, macro_f1=0.5321, draw_recall=0.2278
- `CAF`: rows=1507, macro_f1=0.4761, draw_recall=0.1617
- `AFC`: rows=1340, macro_f1=0.5290, draw_recall=0.2265
- `CONCACAF`: rows=1003, macro_f1=0.5249, draw_recall=0.2258
- `CONMEBOL`: rows=339, macro_f1=0.5336, draw_recall=0.3373

## Highest-Confidence Errors
- 2024-06-07 | Romania vs Liechtenstein | actual=draw predicted=home_win conf=0.97
- 2024-11-17 | Thailand vs Laos | actual=draw predicted=home_win conf=0.94
- 2022-11-17 | Saint Lucia vs San Marino | actual=draw predicted=home_win conf=0.93
- 2019-09-08 | Greece vs Liechtenstein | actual=draw predicted=home_win conf=0.93
- 2025-11-26 | Oman vs Somalia | actual=draw predicted=home_win conf=0.93
- 2024-03-21 | South Africa vs Andorra | actual=draw predicted=home_win conf=0.93
- 2025-03-21 | Guinea vs Somalia | actual=draw predicted=home_win conf=0.93
- 2025-11-17 | Bahrain vs Somalia | actual=away_win predicted=home_win conf=0.92
- 2021-03-30 | Senegal vs Eswatini | actual=draw predicted=home_win conf=0.91
- 2021-03-30 | Turkey vs Latvia | actual=draw predicted=home_win conf=0.91

## Artifact Files
- Confusion matrix: `C:\Users\restr\Desktop\worldcup-2026-prediction\models\match_predictor_confusion_matrix.png`
- Calibration curves: `C:\Users\restr\Desktop\worldcup-2026-prediction\models\match_predictor_calibration_curves.png`
- JSON report: `C:\Users\restr\Desktop\worldcup-2026-prediction\models\match_predictor_evaluation_report.json`