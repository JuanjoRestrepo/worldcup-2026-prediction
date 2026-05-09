# Model Evaluation Report

## Overall
- Selected model: `logistic_c1_draw1` (LogisticRegression)
- Deployed variant: `uncalibrated`
- Accuracy: `0.5587`
- Macro F1: `0.5037`
- Weighted F1: `0.5510`
- Balanced accuracy: `0.5123`
- MCC: `0.3051`
- Log loss: `0.9246`
- ECE: `0.0347`

## Draw Diagnostics
- Precision: `0.2876`
- Recall: `0.2247`
- F1: `0.2523`

## Top Candidate Search Results
- `logistic_c1_draw1` (logistic_regression): rank=1, macro_f1=0.4827, draw_f1=0.2736, log_loss=0.9728
- `logistic_c2_draw1` (logistic_regression): rank=2, macro_f1=0.4826, draw_f1=0.2738, log_loss=0.9728
- `logistic_c1_draw1.25` (logistic_regression): rank=3, macro_f1=0.4827, draw_f1=0.2736, log_loss=0.9728
- `logistic_c2_draw1.2` (logistic_regression): rank=4, macro_f1=0.4826, draw_f1=0.2738, log_loss=0.9728
- `seg_hybrid_auto_tuned` (segment_aware_hybrid): rank=5, macro_f1=0.4848, draw_f1=0.2834, log_loss=0.9752

## Competition Segments
- `World Cup`: rows=1869, macro_f1=0.5324, log_loss=0.8498
- `Friendly`: rows=1557, macro_f1=0.4516, log_loss=0.9905
- `Other`: rows=1518, macro_f1=0.4589, log_loss=0.9950
- `Qualifier`: rows=1064, macro_f1=0.5053, log_loss=0.8536
- `Continental`: rows=419, macro_f1=0.5072, log_loss=0.9383

## Confederation Segments
- `UEFA`: rows=2022, macro_f1=0.5296, draw_recall=0.2313
- `CAF`: rows=1494, macro_f1=0.4710, draw_recall=0.1825
- `AFC`: rows=1322, macro_f1=0.5073, draw_recall=0.2278
- `CONCACAF`: rows=1028, macro_f1=0.5063, draw_recall=0.2788
- `CONMEBOL`: rows=346, macro_f1=0.4946, draw_recall=0.2738

## Highest-Confidence Errors
- 2024-06-07 | Romania vs Liechtenstein | actual=draw predicted=home_win conf=0.96
- 2022-11-17 | Saint Lucia vs San Marino | actual=draw predicted=home_win conf=0.93
- 2024-03-21 | South Africa vs Andorra | actual=draw predicted=home_win conf=0.90
- 2021-09-02 | Italy vs Bulgaria | actual=draw predicted=home_win conf=0.89
- 2021-03-30 | Senegal vs Eswatini | actual=draw predicted=home_win conf=0.88
- 2021-03-30 | Turkey vs Latvia | actual=draw predicted=home_win conf=0.88
- 2024-06-06 | Gibraltar vs Wales | actual=draw predicted=away_win conf=0.87
- 2019-09-08 | Greece vs Liechtenstein | actual=draw predicted=home_win conf=0.87
- 2026-03-26 | Tanzania vs Liechtenstein | actual=away_win predicted=home_win conf=0.86
- 2021-09-08 | Armenia vs Liechtenstein | actual=draw predicted=home_win conf=0.85

## Artifact Files
- Confusion matrix: `models\match_predictor_confusion_matrix.png`
- Calibration curves: `models\match_predictor_calibration_curves.png`
- JSON report: `models\match_predictor_evaluation_report.json`