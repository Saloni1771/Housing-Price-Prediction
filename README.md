# House Price Prediction — EDA & Regression Modeling

Predicting residential house prices using the Ames Housing dataset.

I wanted to work through a complete regression problem — starting with understanding the data, figuring out which variables were actually useful, creating a few features based on the house characteristics, and then comparing different regression models.

The target variable is `SalePrice`.

## What I worked on

The dataset contains information about different aspects of a house, including its size, quality, age, garage, neighborhood, basement, and other features.

I started with exploratory analysis to understand:

* How `SalePrice` is distributed
* Which columns had missing values
* How house size and quality relate to price
* Which features were strongly correlated with the target
* Whether some of the missing values actually had a useful meaning

Some of the strongest relationships were around overall quality, living area, total house size, and garage capacity.

## Feature engineering

I created a few features instead of relying only on the original columns:

* `TotalSF` — basement + 1st floor + 2nd floor area
* `TotalBathrooms` — combined full and half bathrooms
* `HouseAge` — age of the house at the time of sale
* `RemodAge` — years since the house was last remodelled

`TotalSF` was particularly useful. Combining the different floor areas into one measure gave the models a better representation of the overall size of the house.

## Handling missing data

One thing I found interesting in this dataset is that a missing value doesn't always mean the data is unknown.

For example, a missing `PoolQC` generally means the house doesn't have a pool. Treating that as simply "missing" would lose some information.

I therefore handled missing values based on what the particular feature represented instead of applying the same missing-value treatment to every column.

## Models

I compared four regression models:

* Linear Regression
* Ridge Regression
* Random Forest
* Gradient Boosting

I used 5-fold cross-validation to get a better idea of how the models performed beyond a single train/test split.

## Results

| Model                 |  R² Score | CV R² (mean) |   RMSE ($) |    MAE ($) |
| --------------------- | --------: | -----------: | ---------: | ---------: |
| **Gradient Boosting** | **0.898** |    **0.889** |     28,826 | **16,158** |
| Random Forest         |     0.887 |        0.869 |     29,341 |     17,079 |
| Ridge Regression      |     0.872 |        0.841 | **27,879** |     18,163 |
| Linear Regression     |     0.871 |        0.840 |     27,889 |     18,266 |

Gradient Boosting gave the best overall R² and MAE in my comparison, with an R² of 0.898 and MAE of about $16,158.

The models were fairly close in some of the metrics, so I didn't want to judge them only by one score.

## Features that mattered most

Some of the features that came out as important were:

* `TotalSF`
* `OverallQual`
* `TotalBathrooms`
* `GrLivArea`
* `GarageCars`

The interesting part was that an engineered feature like `TotalSF` ended up being one of the strongest predictors.

## What I took from the project

The main takeaway for me was that feature engineering can make a noticeable difference even before changing the model.

Looking at the individual floor-area columns separately doesn't give exactly the same picture as looking at the total usable area of the house. Combining them into `TotalSF` gave the models a more useful representation of house size.

The other thing that stood out was the missing-value handling. In a real dataset, "missing" doesn't always mean the same thing. Sometimes it can actually represent the absence of a feature.

The tree-based models performed better overall than the linear models, which also suggests that the relationship between things like house quality, size, and other characteristics isn't simply a straight-line relationship.

## Tech stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

## Project structure

```text
├── data/
│   └── train.csv                          # Ames Housing dataset
├── notebook/
│   └── House_Price_Prediction.ipynb       # Full analysis notebook
├── outputs/                               # Saved charts
└── analysis.py                            # Standalone script version
```

## Dataset

Kaggle — House Prices: Advanced Regression Techniques
https://www.kaggle.com/c/house-prices-advanced-regression-techniques
