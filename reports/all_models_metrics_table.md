# All Models: Accuracy, Precision, Recall, F1, ROC-AUC by Scenario

| scenario | ground_truth | feature_set | model | n_train | n_test | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Majority vote | CT-only | Gradient Boosting | 157 | 43 | 0.6279 | 0.5882 | 0.5263 | 0.5556 | 0.7171 |
| 1 | Majority vote | CT-only | Logistic Regression | 157 | 43 | 0.6512 | 0.5909 | 0.6842 | 0.6341 | 0.6469 |
| 1 | Majority vote | CT-only | Random Forest | 157 | 43 | 0.6047 | 0.5714 | 0.4211 | 0.4848 | 0.7368 |
| 1 | Majority vote | CT-only | SVM (RBF) | 157 | 43 | 0.6977 | 0.6875 | 0.5789 | 0.6286 | 0.6996 |
| 2 | Vote fraction | CT-only | Gradient Boosting Regressor | 158 | 42 | 0.4524 | 0.3333 | 0.2105 | 0.2581 | 0.4874 |
| 2 | Vote fraction | CT-only | Linear Regression | 158 | 42 | 0.6190 | 0.7143 | 0.2632 | 0.3846 | 0.6430 |
| 2 | Vote fraction | CT-only | Random Forest Regressor | 158 | 42 | 0.5238 | 0.4444 | 0.2105 | 0.2857 | 0.5195 |
| 2 | Vote fraction | CT-only | SVR (RBF) | 158 | 42 | 0.4762 | 0.3333 | 0.1579 | 0.2143 | 0.4622 |
| 3 | Majority vote | CT + demographics | Gradient Boosting | 157 | 43 | 0.6279 | 0.5789 | 0.5789 | 0.5789 | 0.6754 |
| 3 | Majority vote | CT + demographics | Logistic Regression | 157 | 43 | 0.6279 | 0.5789 | 0.5789 | 0.5789 | 0.7193 |
| 3 | Majority vote | CT + demographics | Random Forest | 157 | 43 | 0.6512 | 0.6250 | 0.5263 | 0.5714 | 0.7237 |
| 3 | Majority vote | CT + demographics | SVM (RBF) | 157 | 43 | 0.6977 | 0.6500 | 0.6842 | 0.6667 | 0.7566 |
| 4 | Majority vote | Demographics-only | Gradient Boosting | 157 | 43 | 0.5581 | 0.5000 | 0.4737 | 0.4865 | 0.4978 |
| 4 | Majority vote | Demographics-only | Logistic Regression | 157 | 43 | 0.6047 | 0.5556 | 0.5263 | 0.5405 | 0.7061 |
| 4 | Majority vote | Demographics-only | Random Forest | 157 | 43 | 0.5581 | 0.5000 | 0.4737 | 0.4865 | 0.6425 |
| 4 | Majority vote | Demographics-only | SVM (RBF) | 157 | 43 | 0.5814 | 0.5200 | 0.6842 | 0.5909 | 0.6009 |
| 5 | Halldor+3D agree (2/3) | CT-only | Gradient Boosting | 119 | 31 | 0.7742 | 0.7778 | 0.5833 | 0.6667 | 0.7325 |
| 5 | Halldor+3D agree (2/3) | CT-only | Logistic Regression | 119 | 31 | 0.7419 | 0.8333 | 0.4167 | 0.5556 | 0.8421 |
| 5 | Halldor+3D agree (2/3) | CT-only | Random Forest | 119 | 31 | 0.7419 | 0.8333 | 0.4167 | 0.5556 | 0.8860 |
| 5 | Halldor+3D agree (2/3) | CT-only | SVM (RBF) | 119 | 31 | 0.7419 | 0.8333 | 0.4167 | 0.5556 | 0.7851 |
| 6 | Original surgeon only | CT + demographics | Gradient Boosting | 161 | 39 | 0.6923 | 0.6154 | 0.8889 | 0.7273 | 0.7407 |
| 6 | Original surgeon only | CT + demographics | Logistic Regression | 161 | 39 | 0.7179 | 0.6522 | 0.8333 | 0.7317 | 0.7963 |
| 6 | Original surgeon only | CT + demographics | Random Forest | 161 | 39 | 0.7436 | 0.6818 | 0.8333 | 0.7500 | 0.6852 |
| 6 | Original surgeon only | CT + demographics | SVM (RBF) | 161 | 39 | 0.6410 | 0.5769 | 0.8333 | 0.6818 | 0.7063 |
| 7 | Unanimous (3/3) | CT-only | Gradient Boosting | 68 | 19 | 0.6316 | 0.5714 | 0.5000 | 0.5333 | 0.8295 |
| 7 | Unanimous (3/3) | CT-only | Logistic Regression | 68 | 19 | 0.8421 | 1.0000 | 0.6250 | 0.7692 | 0.9773 |
| 7 | Unanimous (3/3) | CT-only | Random Forest | 68 | 19 | 0.7368 | 0.8000 | 0.5000 | 0.6154 | 0.9432 |
| 7 | Unanimous (3/3) | CT-only | SVM (RBF) | 68 | 19 | 0.8421 | 1.0000 | 0.6250 | 0.7692 | 0.8977 |
| 8 | Unanimous (3/3) | Demographics-only | Gradient Boosting | 68 | 19 | 0.6842 | 0.6667 | 0.5000 | 0.5714 | 0.7045 |
| 8 | Unanimous (3/3) | Demographics-only | Logistic Regression | 68 | 19 | 0.8421 | 0.8571 | 0.7500 | 0.8000 | 0.8864 |
| 8 | Unanimous (3/3) | Demographics-only | Random Forest | 68 | 19 | 0.6842 | 0.6667 | 0.5000 | 0.5714 | 0.8636 |
| 8 | Unanimous (3/3) | Demographics-only | SVM (RBF) | 68 | 19 | 0.8421 | 0.7273 | 1.0000 | 0.8421 | 0.9091 |
| 9 | Unanimous (3/3) | CT + demographics | Gradient Boosting | 68 | 19 | 0.6842 | 1.0000 | 0.2500 | 0.4000 | 0.7045 |
| 9 | Unanimous (3/3) | CT + demographics | Logistic Regression | 68 | 19 | 0.8421 | 0.8571 | 0.7500 | 0.8000 | 0.9432 |
| 9 | Unanimous (3/3) | CT + demographics | Random Forest | 68 | 19 | 0.7895 | 1.0000 | 0.5000 | 0.6667 | 0.8864 |
| 9 | Unanimous (3/3) | CT + demographics | SVM (RBF) | 68 | 19 | 0.8421 | 0.8571 | 0.7500 | 0.8000 | 0.9205 |
| 10 | Original surgeon only | CT-only | Gradient Boosting | 161 | 39 | 0.5128 | 0.4762 | 0.5556 | 0.5128 | 0.5370 |
| 10 | Original surgeon only | CT-only | Logistic Regression | 161 | 39 | 0.5128 | 0.4783 | 0.6111 | 0.5366 | 0.5212 |
| 10 | Original surgeon only | CT-only | Random Forest | 161 | 39 | 0.5385 | 0.5000 | 0.5000 | 0.5000 | 0.4577 |
| 10 | Original surgeon only | CT-only | SVM (RBF) | 161 | 39 | 0.4872 | 0.4375 | 0.3889 | 0.4118 | 0.5503 |
| 11 | Original surgeon only | Demographics-only | Gradient Boosting | 161 | 39 | 0.7949 | 0.7500 | 0.8333 | 0.7895 | 0.7540 |
| 11 | Original surgeon only | Demographics-only | Logistic Regression | 161 | 39 | 0.8205 | 0.7619 | 0.8889 | 0.8205 | 0.8095 |
| 11 | Original surgeon only | Demographics-only | Random Forest | 161 | 39 | 0.7949 | 0.7500 | 0.8333 | 0.7895 | 0.7460 |
| 11 | Original surgeon only | Demographics-only | SVM (RBF) | 161 | 39 | 0.6923 | 0.6250 | 0.8333 | 0.7143 | 0.7143 |
| 12 | Vote fraction | Demographics-only | Gradient Boosting Regressor | 158 | 42 | 0.5476 | 0.5000 | 0.5263 | 0.5128 | 0.5927 |
| 12 | Vote fraction | Demographics-only | Linear Regression | 158 | 42 | 0.5476 | 0.5000 | 0.5263 | 0.5128 | 0.4554 |
| 12 | Vote fraction | Demographics-only | Random Forest Regressor | 158 | 42 | 0.6190 | 0.5789 | 0.5789 | 0.5789 | 0.5469 |
| 12 | Vote fraction | Demographics-only | SVR (RBF) | 158 | 42 | 0.5476 | 0.5000 | 0.4737 | 0.4865 | 0.5606 |
| 13 | Vote fraction | CT + demographics | Gradient Boosting Regressor | 158 | 42 | 0.4762 | 0.3333 | 0.1579 | 0.2143 | 0.5721 |
| 13 | Vote fraction | CT + demographics | Linear Regression | 158 | 42 | 0.5714 | 0.5333 | 0.4211 | 0.4706 | 0.6590 |
| 13 | Vote fraction | CT + demographics | Random Forest Regressor | 158 | 42 | 0.5000 | 0.4167 | 0.2632 | 0.3226 | 0.5629 |
| 13 | Vote fraction | CT + demographics | SVR (RBF) | 158 | 42 | 0.5238 | 0.4545 | 0.2632 | 0.3333 | 0.5423 |
