“””
House Price Prediction - EDA, Feature Engineering & Modeling
Dataset: Ames Housing (Kaggle House Prices - Advanced Regression Techniques)
“””
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings(‘ignore’)

sns.set_style(‘whitegrid’)
plt.rcParams[‘figure.dpi’] = 110

df = pd.read_csv(‘data/train.csv’)
print(f”Dataset shape: {df.shape}”)

# ============================================================

# 1. EDA

# ============================================================

# 1a. Target variable distribution

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df[‘SalePrice’], kde=True, ax=axes[0], color=’#2563eb’)
axes[0].set_title(‘SalePrice Distribution (Right-Skewed)’)
axes[0].set_xlabel(‘Sale Price ($)’)

sns.histplot(np.log1p(df[‘SalePrice’]), kde=True, ax=axes[1], color=’#16a34a’)
axes[1].set_title(‘Log-Transformed SalePrice (Near-Normal)’)
axes[1].set_xlabel(‘log(1 + Sale Price)’)
plt.tight_layout()
plt.savefig(‘outputs/01_target_distribution.png’, bbox_inches=‘tight’)
plt.close()

# 1b. Missing values

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]

plt.figure(figsize=(9, 7))
sns.barplot(x=missing_pct.values, y=missing_pct.index, color=’#dc2626’)
plt.xlabel(’% Missing’)
plt.title(‘Missing Data by Feature’)
plt.tight_layout()
plt.savefig(‘outputs/02_missing_values.png’, bbox_inches=‘tight’)
plt.close()

# 1c. Correlation with target (numeric features)

numeric_df = df.select_dtypes(include=[np.number])
corr_with_target = numeric_df.corr()[‘SalePrice’].sort_values(ascending=False)
top_corr_features = corr_with_target.head(11).index.tolist()  # top 10 + SalePrice itself

plt.figure(figsize=(9, 8))
sns.heatmap(numeric_df[top_corr_features].corr(), annot=True, fmt=’.2f’,
cmap=‘RdBu_r’, center=0, square=True, linewidths=0.5)
plt.title(‘Correlation Heatmap — Top Features vs SalePrice’)
plt.tight_layout()
plt.savefig(‘outputs/03_correlation_heatmap.png’, bbox_inches=‘tight’)
plt.close()

print(”\nTop 10 features correlated with SalePrice:”)
print(corr_with_target.head(11)[1:])

# 1d. Key relationships

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

sns.scatterplot(data=df, x=‘GrLivArea’, y=‘SalePrice’, alpha=0.5, ax=axes[0,0], color=’#2563eb’)
axes[0,0].set_title(‘Living Area vs Sale Price’)

sns.boxplot(data=df, x=‘OverallQual’, y=‘SalePrice’, ax=axes[0,1], color=’#16a34a’)
axes[0,1].set_title(‘Overall Quality vs Sale Price’)

sns.scatterplot(data=df, x=‘YearBuilt’, y=‘SalePrice’, alpha=0.5, ax=axes[1,0], color=’#f59e0b’)
axes[1,0].set_title(‘Year Built vs Sale Price’)

neighborhood_avg = df.groupby(‘Neighborhood’)[‘SalePrice’].median().sort_values(ascending=False)
sns.barplot(x=neighborhood_avg.values, y=neighborhood_avg.index, ax=axes[1,1], color=’#7c3aed’)
axes[1,1].set_title(‘Median Sale Price by Neighborhood’)
axes[1,1].set_xlabel(‘Median Sale Price ($)’)

plt.tight_layout()
plt.savefig(‘outputs/04_key_relationships.png’, bbox_inches=‘tight’)
plt.close()

print(”\nEDA visuals saved to outputs/”)

# ============================================================

# 2. FEATURE ENGINEERING

# ============================================================

data = df.copy()

# Drop columns that are >80% missing (unreliable signal)

high_missing_cols = missing_pct[missing_pct > 80].index.tolist()
data = data.drop(columns=high_missing_cols)
print(f”\nDropped high-missing columns: {high_missing_cols}”)

# Fill categorical NAs with ‘None’ (NA often means “feature absent”, e.g. no garage)

cat_cols = data.select_dtypes(include=‘object’).columns
for col in cat_cols:
data[col] = data[col].fillna(‘None’)

# Fill numeric NAs with median

num_cols = data.select_dtypes(include=[np.number]).columns
for col in num_cols:
data[col] = data[col].fillna(data[col].median())

# Derived features (common domain-informed engineering for this dataset)

data[‘HouseAge’] = data[‘YrSold’] - data[‘YearBuilt’]
data[‘RemodAge’] = data[‘YrSold’] - data[‘YearRemodAdd’]
data[‘TotalSF’] = data[‘TotalBsmtSF’] + data[‘1stFlrSF’] + data[‘2ndFlrSF’]
data[‘TotalBathrooms’] = (data[‘FullBath’] + 0.5 * data[‘HalfBath’] +
data[‘BsmtFullBath’] + 0.5 * data[‘BsmtHalfBath’])

# Log-transform target to handle skew

data[‘SalePrice_log’] = np.log1p(data[‘SalePrice’])

# Label-encode categoricals (simple, effective for tree models)

le_cols = data.select_dtypes(include=‘object’).columns
encoders = {}
for col in le_cols:
le = LabelEncoder()
data[col] = le.fit_transform(data[col].astype(str))
encoders[col] = le

print(f”\nFinal feature set shape: {data.shape}”)

# ============================================================

# 3. MODELING

# ============================================================

drop_cols = [‘Id’, ‘SalePrice’, ‘SalePrice_log’]
X = data.drop(columns=drop_cols)
y = data[‘SalePrice_log’]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
‘Linear Regression’: LinearRegression(),
‘Ridge Regression’: Ridge(alpha=10),
‘Random Forest’: RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1),
‘Gradient Boosting’: GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
}

results = []
for name, model in models.items():
model.fit(X_train, y_train)
preds = model.predict(X_test)

```
# Convert back from log scale for interpretable $ metrics
preds_actual = np.expm1(preds)
y_test_actual = np.expm1(y_test)

rmse_log = np.sqrt(mean_squared_error(y_test, preds))
rmse_dollar = np.sqrt(mean_squared_error(y_test_actual, preds_actual))
mae_dollar = mean_absolute_error(y_test_actual, preds_actual)
r2 = r2_score(y_test, preds)

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

results.append({
    'Model': name,
    'R2 Score': round(r2, 4),
    'CV R2 (mean)': round(cv_scores.mean(), 4),
    'RMSE (log)': round(rmse_log, 4),
    'RMSE ($)': round(rmse_dollar, 0),
    'MAE ($)': round(mae_dollar, 0)
})
```

results_df = pd.DataFrame(results).sort_values(‘R2 Score’, ascending=False)
print(”\n” + “=”*70)
print(“MODEL COMPARISON”)
print(”=”*70)
print(results_df.to_string(index=False))
results_df.to_csv(‘outputs/model_comparison.csv’, index=False)

# Model comparison chart

plt.figure(figsize=(9, 5))
sns.barplot(data=results_df, x=‘R2 Score’, y=‘Model’, color=’#2563eb’)
plt.title(‘Model Comparison — R² Score (higher is better)’)
plt.xlim(0.7, 1.0)
plt.tight_layout()
plt.savefig(‘outputs/05_model_comparison.png’, bbox_inches=‘tight’)
plt.close()

# ============================================================

# 4. FEATURE IMPORTANCE (best model)

# ============================================================

best_model_name = results_df.iloc[0][‘Model’]
best_model = models[best_model_name]

if hasattr(best_model, ‘feature_importances_’):
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)

```
plt.figure(figsize=(9, 7))
sns.barplot(x=importances.values, y=importances.index, color='#16a34a')
plt.title(f'Top 15 Feature Importances — {best_model_name}')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('outputs/06_feature_importance.png', bbox_inches='tight')
plt.close()

print(f"\nTop 10 most important features ({best_model_name}):")
print(importances.head(10))
```

print(”\nDone. All outputs saved to outputs/”)
