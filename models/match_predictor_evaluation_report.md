# Model Evaluation Report

## Overall
- Selected model: `xgboost_n400_d4_lr0.03_lambda2.0_draw1` (XGBClassifier)
- Deployed variant: `uncalibrated`
- Accuracy: `0.5569`
- Macro F1: `0.5156`
- Weighted F1: `0.5587`
- Balanced accuracy: `0.5184`
- MCC: `0.3074`
- Log loss: `0.9185`
- ECE: `0.0370`

## Draw Diagnostics
- Precision: `0.2905`
- Recall: `0.2938`
- F1: `0.2922`

## Top Candidate Search Results
- `xgboost_n400_d4_lr0.03_lambda2.0_draw1` (xgboost): rank=1, macro_f1=0.4870, draw_f1=0.2919, log_loss=0.9653
- `seg_hybrid_auto_tuned` (segment_aware_hybrid): rank=2, macro_f1=0.4867, draw_f1=0.2888, log_loss=0.9681
- `hybrid_override_u0.42_d0.5` (hybrid_draw_override_ensemble): rank=3, macro_f1=0.4872, draw_f1=0.3011, log_loss=0.9804
- `hybrid_override_u0.45_d0.5` (hybrid_draw_override_ensemble): rank=4, macro_f1=0.4859, draw_f1=0.3161, log_loss=0.9944
- `hybrid_override_u0.48_d0.5` (hybrid_draw_override_ensemble): rank=5, macro_f1=0.4844, draw_f1=0.3341, log_loss=1.0059

## Competition Segments
- `World Cup`: rows=1834, macro_f1=0.5529, log_loss=0.8333
- `Friendly`: rows=1677, macro_f1=0.4738, log_loss=0.9807
- `Other`: rows=1504, macro_f1=0.4735, log_loss=0.9937
- `Qualifier`: rows=1023, macro_f1=0.5357, log_loss=0.8508
- `Continental`: rows=419, macro_f1=0.4884, log_loss=0.9372

## Confederation Segments
- `UEFA`: rows=2042, macro_f1=0.5370, draw_recall=0.3050
- `CAF`: rows=1499, macro_f1=0.4913, draw_recall=0.2612
- `AFC`: rows=1298, macro_f1=0.5173, draw_recall=0.2800
- `CONCACAF`: rows=1016, macro_f1=0.5040, draw_recall=0.3348
- `CONMEBOL`: rows=355, macro_f1=0.5321, draw_recall=0.3293

## Highest-Confidence Errors
- 2024-06-07 | Romania vs Liechtenstein | actual=draw predicted=home_win conf=0.94
- 2021-03-30 | Senegal vs Eswatini | actual=draw predicted=home_win conf=0.92
- 2019-11-19 | Latvia vs Austria | actual=home_win predicted=away_win conf=0.91
- 2024-03-21 | South Africa vs Andorra | actual=draw predicted=home_win conf=0.90
- 2021-03-31 | Ukraine vs Kazakhstan | actual=draw predicted=home_win conf=0.89
- 2024-06-06 | Gibraltar vs Wales | actual=draw predicted=away_win conf=0.88
- 2023-10-17 | Lithuania vs Hungary | actual=draw predicted=away_win conf=0.88
- 2019-11-14 | Albania vs Andorra | actual=draw predicted=home_win conf=0.87
- 2023-11-20 | Mali vs Central African Republic | actual=draw predicted=home_win conf=0.87
- 2025-11-26 | Oman vs Somalia | actual=draw predicted=home_win conf=0.87

## Artifact Files
- Confusion matrix: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_confusion_matrix.png`
- Calibration curves: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_calibration_curves.png`
- JSON report: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_evaluation_report.json`