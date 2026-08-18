# Comparative Summary Across All Four Exercises

## Majority-vote family (directly comparable)
- Logistic Regression ROC-AUC: CT-only 0.6469 -> CT + demographics 0.7193 (delta +0.0724)
- Logistic Regression ROC-AUC: CT-only 0.6469 -> demographics-only 0.7061 (delta +0.0592)
- Logistic Regression Accuracy: CT-only 0.6512 -> CT + demographics 0.6279

Best model by ROC-AUC in each majority experiment:
- Majority: CT + demographics: SVM (RBF) (ROC-AUC 0.7566, Accuracy 0.6977, PR-AUC 0.7392)
- Majority: CT-only: Random Forest (ROC-AUC 0.7368, Accuracy 0.6047, PR-AUC 0.6309)
- Majority: demographics-only: Logistic Regression (ROC-AUC 0.7061, Accuracy 0.6047, PR-AUC 0.7014)

## Vote-fraction exercise (different target, not directly equivalent)
- Best RMSE: Linear Regression (RMSE 0.3173, MAE 0.2544, R2 0.1060)
- Because vote-fraction uses a continuous target, it is best compared via RMSE/MAE/R2 rather than directly against majority-vote ROC-AUC.
