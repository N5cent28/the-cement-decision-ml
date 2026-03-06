# Comparative Summary Across All Four Exercises

## Majority-vote family (directly comparable)
- Logistic Regression ROC-AUC: CT-only 0.7405 -> CT + demographics 0.8690 (delta +0.1285)
- Logistic Regression ROC-AUC: CT-only 0.7405 -> demographics-only 0.7857 (delta +0.0452)
- Logistic Regression Accuracy: CT-only 0.6341 -> CT + demographics 0.7561

Best model by ROC-AUC in each majority experiment:
- Majority: CT + demographics: Logistic Regression (ROC-AUC 0.8690, Accuracy 0.7561, PR-AUC 0.8780)
- Majority: CT-only: Logistic Regression (ROC-AUC 0.7405, Accuracy 0.6341, PR-AUC 0.7539)
- Majority: demographics-only: Logistic Regression (ROC-AUC 0.7857, Accuracy 0.6829, PR-AUC 0.7673)

## Vote-fraction exercise (different target, not directly equivalent)
- Best RMSE: Linear Regression (RMSE 0.3058, MAE 0.2576, R2 0.1831)
- Because vote-fraction uses a continuous target, it is best compared via RMSE/MAE/R2 rather than directly against majority-vote ROC-AUC.
