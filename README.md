# House Price Prediction — EDA & Regression Modeling

Exploratory data analysis and machine learning project predicting residential home prices using the **Ames Housing dataset** (Kaggle: House Prices – Advanced Regression Techniques).

## 📊 Overview

This project walks through a complete, small-scale analytics workflow: exploring a real dataset, engineering features from domain understanding, and comparing multiple regression models to predict `SalePrice`.

**Best model: Gradient Boosting Regressor — R² = 0.898, MAE ≈ $16,155**

## 🔍 What’s inside

- **Exploratory Data Analysis** — target distribution, missing data patterns, correlation analysis, and key feature relationships (living area, quality, year built, neighborhood)
- **Feature Engineering** — handling of “meaningful missingness,” derived features (`TotalSF`, `TotalBathrooms`, `HouseAge`, `RemodAge`)
- **Model Comparison** — Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting, evaluated with 5-fold cross-validation
- **Feature Importance** — identifying which factors actually drive the model’s predictions

## 📈 Results

|Model                |R² Score |CV R² (mean)|RMSE ($)|MAE ($)|
|---------------------|---------|------------|--------|-------|
|**Gradient Boosting**|**0.898**|0.889       |28,825  |16,155 |
|Random Forest        |0.887    |0.869       |29,409  |17,182 |
|Ridge Regression     |0.872    |0.841       |27,879  |18,163 |
|Linear Regression    |0.871    |0.840       |27,889  |18,266 |

**Top predictive features:** `TotalSF` (engineered total square footage), `OverallQual`, `TotalBathrooms`, `GrLivArea`, `GarageCars`

## 🛠️ Tech stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

## 📁 Structure

```
├── data/
│   └── train.csv                          # Ames Housing dataset
├── notebook/
│   └── House_Price_Prediction.ipynb       # Full analysis notebook
├── outputs/                                # Saved charts
└── analysis.py                             # Standalone script version
```

## 🚀 Key takeaways

- Combining raw columns into a single engineered feature (`TotalSF` = basement + 1st floor + 2nd floor area) proved more predictive than the individual columns alone — a useful reminder that feature engineering often matters as much as model choice.
- Tree-based ensembles outperformed linear models, suggesting the relationship between features and price isn’t purely additive — quality, size, and location interact.
- Missing data in this dataset is often *informative* (e.g., a missing `PoolQC` means “no pool,” not “unknown”), which shaped the cleaning strategy.

## 📌 Dataset source

[Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
