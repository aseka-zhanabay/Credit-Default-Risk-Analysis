# Credit Default Risk Analysis

## Overview

This project explores credit-related data and prepares it for a potential machine learning model to predict borrower default. The notebook focuses on exploratory data analysis (EDA), feature scaling, stratified train/test splitting, and addressing class imbalance with SMOTE.

> **Project status:** Data exploration and preprocessing are implemented. Model training and evaluation are not yet included in the notebook.

## Objectives

- Inspect the dataset structure, descriptive statistics, and missing values.
- Explore feature relationships and the distribution of credit scores.
- Define `default` as the target variable.
- Split data into training and test sets with an 80/20 stratified split.
- Standardize numeric features using `StandardScaler` fitted on the training set.
- Apply SMOTE to the training data to address class imbalance.

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- imbalanced-learn (SMOTE)
- Google Colab / Jupyter Notebook

## Notebook Workflow

1. Import data analysis, visualization, and machine learning libraries.
2. Load a CSV dataset from Google Drive.
3. Inspect rows, dimensions, summary statistics, and missing values.
4. Plot a correlation heatmap and `credit_score` histogram.
5. Separate features (`X`) from the `default` target (`y`).
6. Create a stratified train/test split (`test_size=0.2`, `random_state=42`).
7. Fit `StandardScaler` on the training data and transform both splits.
8. Use SMOTE to rebalance the training set.


## Getting Started

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn notebook
```

Open `Practice.ipynb` in Google Colab or Jupyter Notebook. The notebook currently loads a CSV from a Google Drive path (`/content/drive/MyDrive/Colab Notebooks/11.03.25.csv`). Update this path to your own dataset location before running it.

**Important:** Add `from imblearn.over_sampling import SMOTE` before the SMOTE step. The current notebook uses `SMOTE` without importing it. Also ensure that the input features are numeric and any missing values are handled before scaling and resampling.

## Current Limitations and Next Steps

- The source CSV is not included in this repository description; reproducibility requires access to the dataset.
- The notebook does not yet train or evaluate a classifier, despite importing several classification models and metrics.
- Future work could include comparing Logistic Regression, Decision Tree, and Random Forest using precision, recall, F1-score, ROC-AUC, and confusion matrices.
- For reliable evaluation, preprocessing and resampling should be confined to training folds when cross-validation is introduced.

## Notes

This README describes the work currently present in the notebook and does not claim predictive performance results that have not been generated.
