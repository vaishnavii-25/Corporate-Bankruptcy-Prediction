# Corporate Bankruptcy Prediction

**Course:** UE24CS352A --- Machine Learning\
**Team:** 5\
**Team members:** Sri Vaishnavi Peri, Sneha Anand

## Problem statement

Predict whether a company is bankrupt using its financial attributes and
compare machine-learning classifiers.

## Dataset

The project uses `3year.arff`, the third-year forecasting file from the
UCI Polish Companies Bankruptcy dataset:
https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy. It
contains 10,503 records, 64 input attributes, 495 bankrupt companies and
10,008 non-bankrupt companies. The data contains missing values and
duplicate records.

## Approach

The data is split into 80% training and 20% testing with stratification.
Infinite values are treated as missing, missing values are imputed using
training-set medians, and features are standardized for Logistic
Regression and SVM. Logistic Regression, RBF-kernel SVM, Random Forest
and XGBoost are compared using accuracy, precision, recall, F1-score,
ROC-AUC and PR-AUC. The model with the highest PR-AUC is selected.

## Implementation

`Corporate_Bankruptcy_Prediction.ipynb` includes data exploration,
preprocessing, model training, evaluation plots, feature-importance
analysis, model saving and sample predictions. On a fresh run with the
supplied dataset, XGBoost achieved the highest PR-AUC (0.6089) and was
selected.


## Dataset attribution

Dataset: Polish Companies Bankruptcy, UCI Machine Learning Repository.
Check the dataset page for licence and attribution requirements before
redistributing the data file.
