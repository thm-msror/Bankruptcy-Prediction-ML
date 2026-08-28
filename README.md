# Bankruptcy Prediction - AutoGluon vs XGBoost

Undergraduate research project submitted to the **QCPA Fintech & AI Research Competition**, hosted by the Qatar Association of Certified Public Accountants.

> _Enhancing Bankruptcy Prediction with Automated Learning and Interpretability: A Comparative Study of Metaheuristics, XGBoost, and AutoML_

## Problem

Predicting corporate bankruptcy from financial statement data remains a critical challenge in finance. We address this as a binary classification task on highly imbalanced data (bankrupt firms represent roughly 3-5% of samples).

## Dataset

**Polish Companies Bankruptcy Data** from the UCI ML Repository. Five subsets correspond to forecasting horizons of 1-5 years before bankruptcy, totaling 43,405 company-year observations with 64 financial ratios (profitability, leverage, liquidity, efficiency).

## Methods

| Component                | Details                                                                                   |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| Baseline                 | Random Forest classifier                                                                  |
| Model 1                  | XGBoost with GridSearchCV tuning, threshold calibration, and SHAP interpretability        |
| Model 2                  | AutoGluon TabularPredictor (`best_quality` preset) with automated ensembling and stacking |
| Imbalance handling       | SMOTE and ADASYN oversampling on the training set                                         |
| Dimensionality reduction | PCA (95% variance retained) applied after oversampling                                    |
| Interpretability         | SHAP values, feature importance plots, risk heatmaps                                      |

## Key Results

| Model                    | ROC AUC              | Bankrupt-class F1 |
| ------------------------ | -------------------- | ----------------- |
| Random Forest (baseline) | 0.93                 | 0.55              |
| XGBoost (tuned, SMOTE)   | 0.97                 | 0.76              |
| AutoGluon ensemble       | best across horizons | see notebook      |

## Project Structure

```
Enhancing Bankruptcy Prediction.pdf             Research paper

notebooks/
    01_EDA_and_Random_Forest_Baseline.ipynb     Data aggregation, EDA, RF baseline
    02_XGBoost_SHAP_and_Resampling.ipynb        SMOTE/ADASYN, XGBoost + SHAP, LightGBM/CatBoost
    03_AutoGluon_AutoML_Pipeline.ipynb          AutoGluon with SMOTE + PCA

datasets/
    raw/           1year.csv .. 5year.csv (original UCI data)
    clean/         combined_polish_bankruptcy_data.csv, col_mapping_*.csv
    processed/     processed_1year.csv .. processed_combined.csv (model-ready)
```

## How to Run

```bash
pip install pandas scikit-learn xgboost autogluon imbalanced-learn shap seaborn matplotlib
```

Run the notebooks in order: `01_EDA_and_Random_Forest_Baseline` → `02_XGBoost_SHAP_and_Resampling` → `03_AutoGluon_AutoML_Pipeline`. The processed datasets are included in `datasets/`.
