# Model Evaluation Report

## Overall
- Selected model: `hybrid_override_u0.42_d0.5` (HybridDrawOverrideEnsemble)
- Deployed variant: `temperature`
- Accuracy: `0.5565`
- Macro F1: `0.5216`
- Weighted F1: `0.5622`
- Balanced accuracy: `0.5255`
- MCC: `0.3156`
- Log loss: `0.9436`
- ECE: `0.0495`

## Draw Diagnostics
- Precision: `0.2912`
- Recall: `0.3214`
- F1: `0.3056`

## Top Candidate Search Results
- `hybrid_override_u0.42_d0.5` (hybrid_draw_override_ensemble): rank=1, macro_f1=0.5020, draw_f1=0.3104, log_loss=0.9640
- `hybrid_override_u0.45_d0.5` (hybrid_draw_override_ensemble): rank=2, macro_f1=0.5017, draw_f1=0.3275, log_loss=0.9771
- `seg_hybrid_auto_tuned` (segment_aware_hybrid): rank=3, macro_f1=0.5005, draw_f1=0.2989, log_loss=0.9526
- `xgboost_n400_d4_lr0.03_lambda2.0_draw1` (xgboost): rank=4, macro_f1=0.5005, draw_f1=0.2842, log_loss=0.9428
- `hybrid_override_u0.42_d0.6` (hybrid_draw_override_ensemble): rank=5, macro_f1=0.4988, draw_f1=0.2893, log_loss=0.9498

## Competition Segments
- `World Cup`: rows=1903, macro_f1=0.5575, log_loss=0.9034
- `Friendly`: rows=1654, macro_f1=0.4794, log_loss=0.9940
- `Other`: rows=1492, macro_f1=0.4703, log_loss=0.9943
- `Qualifier`: rows=1003, macro_f1=0.5310, log_loss=0.8614
- `Continental`: rows=428, macro_f1=0.5156, log_loss=0.9427

## Confederation Segments
- `UEFA`: rows=2064, macro_f1=0.5330, draw_recall=0.2731
- `CAF`: rows=1494, macro_f1=0.4939, draw_recall=0.2950
- `AFC`: rows=1278, macro_f1=0.5319, draw_recall=0.3597
- `CONCACAF`: rows=1021, macro_f1=0.5330, draw_recall=0.4018
- `CONMEBOL`: rows=368, macro_f1=0.5114, draw_recall=0.4070

## Highest-Confidence Errors
- 2026-07-05 | Brazil vs Norway | actual=away_win predicted=home_win conf=1.00
- 2026-07-07 | Switzerland vs Colombia | actual=draw predicted=home_win conf=1.00
- 2026-07-06 | United States vs Belgium | actual=away_win predicted=home_win conf=1.00
- 2026-07-03 | Australia vs Egypt | actual=draw predicted=home_win conf=1.00
- 2026-07-18 | France vs England | actual=away_win predicted=home_win conf=1.00
- 2026-07-15 | England vs Argentina | actual=away_win predicted=home_win conf=1.00
- 2026-07-14 | France vs Spain | actual=away_win predicted=home_win conf=1.00
- 2026-07-06 | Portugal vs Spain | actual=away_win predicted=home_win conf=1.00
- 2026-07-04 | Canada vs Morocco | actual=away_win predicted=home_win conf=1.00
- 2026-07-05 | Mexico vs England | actual=away_win predicted=home_win conf=1.00

## Artifact Files
- Confusion matrix: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_confusion_matrix.png`
- Calibration curves: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_calibration_curves.png`
- JSON report: `/Users/jorgerestrepo/Desktop/worldcup-2026-prediction/models/match_predictor_evaluation_report.json`