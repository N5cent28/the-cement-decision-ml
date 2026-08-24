# All Models: Accuracy, Precision, Recall, F1, ROC-AUC by Scenario

95% CIs are computed from 30 repeated patient-grouped train/test splits (see `results_repeated_splits_all_scenarios/`), not from the single split used for the point-estimate columns. A blank CI means the repeated-split run hasn't been regenerated for that scenario/model.

| scenario | ground_truth | feature_set | model | n_train | n_test | accuracy | precision | recall | f1 | roc_auc | roc_auc_95ci_lower | roc_auc_95ci_upper | ci_repeats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Majority vote | CT-only | Gradient Boosting | 157 | 43 | 0.6279 | 0.5882 | 0.5263 | 0.5556 | 0.7171 | 0.4960 | 0.7963 | 30 |
| 1 | Majority vote | CT-only | Logistic Regression | 157 | 43 | 0.6512 | 0.5909 | 0.6842 | 0.6341 | 0.6469 | 0.5598 | 0.8842 | 30 |
| 1 | Majority vote | CT-only | Random Forest | 157 | 43 | 0.6047 | 0.5714 | 0.4211 | 0.4848 | 0.7368 | 0.5622 | 0.8158 | 30 |
| 1 | Majority vote | CT-only | SVM (RBF) | 157 | 43 | 0.6977 | 0.6875 | 0.5789 | 0.6286 | 0.6996 | 0.5511 | 0.7932 | 30 |
| 2 | Vote fraction | CT-only | Gradient Boosting Regressor | 158 | 42 | 0.4524 | 0.3333 | 0.2105 | 0.2581 | 0.4874 | 0.4361 | 0.7680 | 30 |
| 2 | Vote fraction | CT-only | Linear Regression | 158 | 42 | 0.6190 | 0.7143 | 0.2632 | 0.3846 | 0.6430 | 0.4629 | 0.7868 | 30 |
| 2 | Vote fraction | CT-only | Random Forest Regressor | 158 | 42 | 0.5238 | 0.4444 | 0.2105 | 0.2857 | 0.5195 | 0.5297 | 0.7813 | 30 |
| 2 | Vote fraction | CT-only | SVR (RBF) | 158 | 42 | 0.4762 | 0.3333 | 0.1579 | 0.2143 | 0.4622 | 0.4545 | 0.7832 | 30 |
| 3 | Majority vote | CT + demographics | Gradient Boosting | 157 | 43 | 0.6279 | 0.5789 | 0.5789 | 0.5789 | 0.6754 | 0.4227 | 0.8211 | 30 |
| 3 | Majority vote | CT + demographics | Logistic Regression | 157 | 43 | 0.6279 | 0.5789 | 0.5789 | 0.5789 | 0.7193 | 0.5554 | 0.9000 | 30 |
| 3 | Majority vote | CT + demographics | Random Forest | 157 | 43 | 0.6512 | 0.6250 | 0.5263 | 0.5714 | 0.7237 | 0.5270 | 0.8180 | 30 |
| 3 | Majority vote | CT + demographics | SVM (RBF) | 157 | 43 | 0.6977 | 0.6500 | 0.6842 | 0.6667 | 0.7566 | 0.5100 | 0.8199 | 30 |
| 4 | Majority vote | Demographics-only | Gradient Boosting | 157 | 43 | 0.5581 | 0.5000 | 0.4737 | 0.4865 | 0.4978 | 0.3170 | 0.7231 | 30 |
| 4 | Majority vote | Demographics-only | Logistic Regression | 157 | 43 | 0.6047 | 0.5556 | 0.5263 | 0.5405 | 0.7061 | 0.4612 | 0.8910 | 30 |
| 4 | Majority vote | Demographics-only | Random Forest | 157 | 43 | 0.5581 | 0.5000 | 0.4737 | 0.4865 | 0.6425 | 0.3516 | 0.7362 | 30 |
| 4 | Majority vote | Demographics-only | SVM (RBF) | 157 | 43 | 0.5814 | 0.5200 | 0.6842 | 0.5909 | 0.6009 | 0.4496 | 0.8114 | 30 |
| 5 | Halldor+3D agree (2/3) | CT-only | Gradient Boosting | 119 | 31 | 0.7742 | 0.7778 | 0.5833 | 0.6667 | 0.7325 | 0.6547 | 0.9234 | 30 |
| 5 | Halldor+3D agree (2/3) | CT-only | Logistic Regression | 119 | 31 | 0.7419 | 0.8333 | 0.4167 | 0.5556 | 0.8421 | 0.6823 | 0.9177 | 30 |
| 5 | Halldor+3D agree (2/3) | CT-only | Random Forest | 119 | 31 | 0.7419 | 0.8333 | 0.4167 | 0.5556 | 0.8860 | 0.6595 | 0.9315 | 30 |
| 5 | Halldor+3D agree (2/3) | CT-only | SVM (RBF) | 119 | 31 | 0.7419 | 0.8333 | 0.4167 | 0.5556 | 0.7851 | 0.6452 | 0.8947 | 30 |
| 6 | Original surgeon only | CT + demographics | Gradient Boosting | 161 | 39 | 0.6923 | 0.6154 | 0.8889 | 0.7273 | 0.7407 | 0.7017 | 0.9545 | 30 |
| 6 | Original surgeon only | CT + demographics | Logistic Regression | 161 | 39 | 0.7179 | 0.6522 | 0.8333 | 0.7317 | 0.7963 | 0.8084 | 0.9860 | 30 |
| 6 | Original surgeon only | CT + demographics | Random Forest | 161 | 39 | 0.7436 | 0.6818 | 0.8333 | 0.7500 | 0.6852 | 0.6722 | 0.9763 | 30 |
| 6 | Original surgeon only | CT + demographics | SVM (RBF) | 161 | 39 | 0.6410 | 0.5769 | 0.8333 | 0.6818 | 0.7063 | 0.6970 | 0.9755 | 30 |
| 7 | Unanimous (3/3) | CT-only | Gradient Boosting | 68 | 19 | 0.6316 | 0.5714 | 0.5000 | 0.5333 | 0.8295 | 0.4799 | 0.9299 | 30 |
| 7 | Unanimous (3/3) | CT-only | Logistic Regression | 68 | 19 | 0.8421 | 1.0000 | 0.6250 | 0.7692 | 0.9773 | 0.6410 | 1.0000 | 30 |
| 7 | Unanimous (3/3) | CT-only | Random Forest | 68 | 19 | 0.7368 | 0.8000 | 0.5000 | 0.6154 | 0.9432 | 0.6570 | 0.9642 | 30 |
| 7 | Unanimous (3/3) | CT-only | SVM (RBF) | 68 | 19 | 0.8421 | 1.0000 | 0.6250 | 0.7692 | 0.8977 | 0.6075 | 0.9718 | 30 |
| 8 | Unanimous (3/3) | Demographics-only | Gradient Boosting | 68 | 19 | 0.6842 | 0.6667 | 0.5000 | 0.5714 | 0.7045 | 0.6601 | 1.0000 | 30 |
| 8 | Unanimous (3/3) | Demographics-only | Logistic Regression | 68 | 19 | 0.8421 | 0.8571 | 0.7500 | 0.8000 | 0.8864 | 0.8333 | 1.0000 | 30 |
| 8 | Unanimous (3/3) | Demographics-only | Random Forest | 68 | 19 | 0.6842 | 0.6667 | 0.5000 | 0.5714 | 0.8636 | 0.8158 | 1.0000 | 30 |
| 8 | Unanimous (3/3) | Demographics-only | SVM (RBF) | 68 | 19 | 0.8421 | 0.7273 | 1.0000 | 0.8421 | 0.9091 | 0.8024 | 1.0000 | 30 |
| 9 | Unanimous (3/3) | CT + demographics | Gradient Boosting | 68 | 19 | 0.6842 | 1.0000 | 0.2500 | 0.4000 | 0.7045 | 0.5807 | 1.0000 | 30 |
| 9 | Unanimous (3/3) | CT + demographics | Logistic Regression | 68 | 19 | 0.8421 | 0.8571 | 0.7500 | 0.8000 | 0.9432 | 0.8118 | 1.0000 | 30 |
| 9 | Unanimous (3/3) | CT + demographics | Random Forest | 68 | 19 | 0.7895 | 1.0000 | 0.5000 | 0.6667 | 0.8864 | 0.7637 | 1.0000 | 30 |
| 9 | Unanimous (3/3) | CT + demographics | SVM (RBF) | 68 | 19 | 0.8421 | 0.8571 | 0.7500 | 0.8000 | 0.9205 | 0.7314 | 1.0000 | 30 |
| 10 | Original surgeon only | CT-only | Gradient Boosting | 161 | 39 | 0.5128 | 0.4762 | 0.5556 | 0.5128 | 0.5370 | 0.3991 | 0.7982 | 30 |
| 10 | Original surgeon only | CT-only | Logistic Regression | 161 | 39 | 0.5128 | 0.4783 | 0.6111 | 0.5366 | 0.5212 | 0.3645 | 0.8089 | 30 |
| 10 | Original surgeon only | CT-only | Random Forest | 161 | 39 | 0.5385 | 0.5000 | 0.5000 | 0.5000 | 0.4577 | 0.3988 | 0.7219 | 30 |
| 10 | Original surgeon only | CT-only | SVM (RBF) | 161 | 39 | 0.4872 | 0.4375 | 0.3889 | 0.4118 | 0.5503 | 0.4328 | 0.7633 | 30 |
| 11 | Original surgeon only | Demographics-only | Gradient Boosting | 161 | 39 | 0.7949 | 0.7500 | 0.8333 | 0.7895 | 0.7540 | 0.7545 | 0.9565 | 30 |
| 11 | Original surgeon only | Demographics-only | Logistic Regression | 161 | 39 | 0.8205 | 0.7619 | 0.8889 | 0.8205 | 0.8095 | 0.8445 | 1.0000 | 30 |
| 11 | Original surgeon only | Demographics-only | Random Forest | 161 | 39 | 0.7949 | 0.7500 | 0.8333 | 0.7895 | 0.7460 | 0.7728 | 0.9871 | 30 |
| 11 | Original surgeon only | Demographics-only | SVM (RBF) | 161 | 39 | 0.6923 | 0.6250 | 0.8333 | 0.7143 | 0.7143 | 0.7999 | 0.9703 | 30 |
| 12 | Vote fraction | Demographics-only | Gradient Boosting Regressor | 158 | 42 | 0.5476 | 0.5000 | 0.5263 | 0.5128 | 0.5927 | 0.4027 | 0.7899 | 30 |
| 12 | Vote fraction | Demographics-only | Linear Regression | 158 | 42 | 0.5476 | 0.5000 | 0.5263 | 0.5128 | 0.4554 | 0.4652 | 0.8215 | 30 |
| 12 | Vote fraction | Demographics-only | Random Forest Regressor | 158 | 42 | 0.6190 | 0.5789 | 0.5789 | 0.5789 | 0.5469 | 0.4008 | 0.8369 | 30 |
| 12 | Vote fraction | Demographics-only | SVR (RBF) | 158 | 42 | 0.5476 | 0.5000 | 0.4737 | 0.4865 | 0.5606 | 0.4091 | 0.7446 | 30 |
| 13 | Vote fraction | CT + demographics | Gradient Boosting Regressor | 158 | 42 | 0.4762 | 0.3333 | 0.1579 | 0.2143 | 0.5721 | 0.4854 | 0.7964 | 30 |
| 13 | Vote fraction | CT + demographics | Linear Regression | 158 | 42 | 0.5714 | 0.5333 | 0.4211 | 0.4706 | 0.6590 | 0.5379 | 0.8570 | 30 |
| 13 | Vote fraction | CT + demographics | Random Forest Regressor | 158 | 42 | 0.5000 | 0.4167 | 0.2632 | 0.3226 | 0.5629 | 0.5267 | 0.8319 | 30 |
| 13 | Vote fraction | CT + demographics | SVR (RBF) | 158 | 42 | 0.5238 | 0.4545 | 0.2632 | 0.3333 | 0.5423 | 0.4515 | 0.8048 | 30 |
