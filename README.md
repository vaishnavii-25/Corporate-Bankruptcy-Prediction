# Corporate Bankruptcy Prediction

**Course:** UE24CS352A — Machine Learning (Mini-Project)
**Team:** 5

| Name | SRN |
|------|-----|
| Sri Vaishnavi Peri | PES2UG24CS517 |
| Sneha Anand | PES2UG24CS506 |

## Problem Statement

Predict whether a company is bankrupt or not bankrupt using its financial attributes, and compare machine-learning classifiers to find the best one.

## Dataset

The project uses `3year.arff`, the third-year forecasting file from the UCI Polish Companies Bankruptcy dataset: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy

- 10,503 company records and 64 input attributes
- 495 bankrupt and 10,008 non-bankrupt companies (strong class imbalance)
- Contains missing values and duplicate records

## Approach

- 80:20 stratified train/test split.
- Infinite values treated as missing; missing values imputed with training-set medians.
- Features standardized for Logistic Regression and SVM.
- Models compared: Logistic Regression, RBF-kernel SVM, Random Forest and XGBoost.
- Metrics: accuracy, precision, recall, F1-score, ROC-AUC and PR-AUC. The model with the highest PR-AUC is selected.

## Implementation

`Corporate_Bankruptcy_Prediction.ipynb` covers data exploration, preprocessing, model training, evaluation plots, feature-importance analysis, model saving and sample predictions.

On a fresh run with the supplied dataset, **XGBoost achieved the highest PR-AUC (0.6089)** and was selected as the final model.

## Repository Contents

| File | Description |
|------|-------------|
| `Corporate_Bankruptcy_Prediction.ipynb` | Main notebook (already run, outputs included) |
| `3year.arff` | Dataset |
| `requirements.txt` | Python libraries used |
| `Corporate_Bankruptcy_Prediction_writeup.pdf` | One-page project write-up |

## Setup and Run

1. Clone the repository and move into the folder:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start Jupyter and open the notebook:
   ```bash
   jupyter notebook Corporate_Bankruptcy_Prediction.ipynb
   ```
5. Run all cells from top to bottom (**Kernel → Restart & Run All**). Keep `3year.arff` in the same folder as the notebook.

The notebook can also be opened in Google Colab; upload `3year.arff` to the session before running.

## Dataset Attribution

Dataset: Polish Companies Bankruptcy, UCI Machine Learning Repository. See the dataset page linked above for licence and attribution terms.
