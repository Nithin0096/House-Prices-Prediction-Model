# House Prices Prediction Model

Regression model to predict residential house prices in Ames, Iowa using the [Kaggle House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

**Kaggle Score: 0.13861 RMSE**

---

## Problem Statement

Predict the final sale price of homes based on 79 explanatory variables describing almost every aspect of residential homes in Ames, Iowa.

---

## Project Pipeline

### 1. Data Cleaning
- Handled 19 columns with null values — distinguished between "feature doesn't exist" (filled with `None`/`0`) and "genuinely missing data" (filled with median/mode)
- Fixed data types: `MSSubClass`, `YrSold`, `MoSold` converted from int to string (categorical labels, not quantities)
- Removed 2 outliers — houses with 4000+ sq ft but unusually low sale prices (likely non-market transactions)

### 2. Exploratory Data Analysis
- Applied log transformation to `SalePrice` to fix right skew and normalize distribution
- Top correlated features with SalePrice: `OverallQual` (0.82), `GrLivArea` (0.73), `GarageCars` (0.68)
- Identified multicollinearity pairs: `GarageArea/GarageCars` and `TotalBsmtSF/1stFlrSF`
- Neighborhood significantly affects price — `NridgHt` and `NoRidge` are the most expensive

### 3. Feature Engineering
| New Feature | Description |
|---|---|
| `TotalSF` | Basement + 1st floor + 2nd floor square footage |
| `TotalBath` | Full baths + 0.5 × half baths (all floors) |
| `HouseAge` | Year sold minus year built |
| `Remodelled` | Binary flag — was the house remodelled? |

Dropped 6 redundant columns: `GarageArea`, `1stFlrSF`, `TotalBsmtSF`, `YearBuilt`, `YearRemodAdd`, `Id`

One hot encoded 46 categorical columns → final feature set: 283 columns

### 4. Modelling

5 models tested with GridSearchCV tuning on best performers:

| Model | Default RMSE | Tuned RMSE |
|---|---|---|
| Linear Regression | 0.1406 | - |
| Ridge | 0.1300 | 0.1262 |
| Lasso | 0.1886 | **0.1235** ✅ |
| Random Forest | 0.1454 | - |
| Gradient Boosting | 0.1253 | 0.1267 |

**Winner: Tuned Lasso (α=0.001) with RMSE 0.1235**

Default Lasso performed worst — tuning revealed that α=0.001 (very gentle penalty) keeps most features and wins.

---

## Key Learnings

- Null values require domain reasoning, not just mechanical filling
- Log transforming a skewed target variable significantly improves model performance
- Default model parameters are rarely optimal — GridSearchCV is essential
- Lasso's built-in feature selection outperformed more complex ensemble methods after tuning

---

## Tech Stack

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (LinearRegression, Ridge, Lasso, RandomForestRegressor, GradientBoostingRegressor, GridSearchCV)

---

## Dataset

[Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
