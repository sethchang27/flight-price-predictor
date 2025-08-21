
# Flight Price Prediction: A Statistical ML Approach

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Description

This project implements a **statistically-driven machine learning pipeline** for predicting flight prices in the Indian aviation market. Unlike typical ML projects that rely on basic feature engineering, this implementation emphasises **rigorous statistical analysis** at every stage of the workflow, from feature selection to model validation.

The system predicts flight prices using a comprehensive set of features including airline, route, timing, duration, and seasonality factors. The model achieves **83.4% explained variance (R²)** with an RMSE of 0.209 on log-transformed prices, demonstrating strong predictive performance across diverse flight scenarios.

### Key Differentiators

- **Statistical Feature Selection**: Multi-method correlation analysis using Pearson, Spearman, Mutual Information, Point-Biserial, and ANOVA F-tests
- **Advanced Feature Engineering**: Cyclical encoding, temporal categorisation, and domain-specific transformations
- **Encoding Strategy Optimisation**: Cardinality-based encoding with target encoding and regularisation techniques
- **Rigorous Model Validation**: Cross-validation with hyperparameter optimisation and multicollinearity analysis

---

## Statistical Methodology & ML Workflow

### 1. Exploratory Data Analysis with Statistical Rigor
- **Distribution Analysis**: Log-transformation of target variable to achieve normality
- **Outlier Detection**: Statistical identification using boxplots and IQR methods
- **Temporal Pattern Analysis**: Day-of-week and seasonal price variations
- **Initial Data Visualisations**: Boxplots and graphs for data understanding and visualisation


### 2. Feature Engineering with Domain Knowledge
```python
# Cyclical Encoding for Temporal Features
day_sin = sin(2π × day_of_week / 7)
day_cos = cos(2π × day_of_week / 7)

# Time-of-Day Categorisation
time_categories = {
    'Morning': [5-11], 'Afternoon': [12-15], 
    'Evening': [16-18], 'Night': [19-4]
}

# Holiday Impact Analysis
holiday_window = ±1 day around public holidays
```

### 3. Statistical Feature Selection Framework
- **Target Correlation Analysis**: Performed correlation analysis between features and target variable to identify features with low predictive power

| Feature Type | Statistical Test | Threshold | Rationale |
|--------------|------------------|-----------|-----------|
| **Continuous** | Pearson & Spearman Correlation | r ≥ 0.3, p < 0.05 | Linear and monotonic relationships |
| **High-Cardinality Categorical** | Mutual Information | MI > 0.8 | Non-linear dependencies |
| **Binary** | Point-Biserial Correlation | r ≥ 0.2, p < 0.05 | Binary-continuous associations |
| **Low-Cardinality Categorical** | ANOVA F-test | F ≥ 2.0, p < 0.05 | Group mean differences |


- **Multicollinearity Analysis**: Performed pairwise correlation analysis between features to identify similar features that can be dropped to reduce model complexity


### 4. Feature Encodings
- **One-Hot Encodings**: Performed one-hot encodings for low-cardinality discrete features
- **Target Encodings (With Regularisation)**: Performed target encodings for high-cardinality features. Included regularisation to prevent overfitting.
```python
# Smoothed target encoding to prevent overfitting
smoothed_mean = (count × category_mean + smoothing × global_mean) / (count + smoothing)
encoded_value = smoothed_mean + gaussian_noise(σ=0.01)
```

### 5. Model Selection & Validation
#### Cross-Validation Strategy
- **5-Fold Cross-Validation** with `neg_mean_squared_error` scoring
- **Stratified approach** for consistent performance estimation
- **Grid Search CV** for hyperparameter optimisation on top performers

#### Initial Model Screening
| Algorithm | CV RMSE | Standard Deviation | Performance Rank |
|-----------|---------|-------------------|------------------|
| **Gradient Boosting** | **0.2304** | ±0.0076 | **1st** |
| Random Forest | 0.2322 | ±0.0096 | 2nd |
| Ridge Regression | 0.2723 | ±0.0073 | 3rd |
| Linear Regression | 0.2723 | ±0.0073 | 4th |
| KNN | 0.2793 | ±0.0107 | 5th |

#### Hyperparameter Optimisation Process

**Top 3 models selected for tuning**: Gradient Boosting, Random Forest, Ridge

```python
# Grid search parameters for selected models
param_grids = {
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2]
    },
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    },
    'Ridge': {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    }
}
```
#### Final Model Comparison (Post Grid Search)
| Algorithm | Best CV RMSE | Optimal Hyperparameters | Improvement |
|-----------|--------------|-------------------------|-------------|
| **Gradient Boosting** | **0.2129** | n_estimators=200, max_depth=5, learning_rate=0.1 | **-7.6%** |
| Random Forest | 0.2176 | n_estimators=200, max_depth=10, min_samples_split=5 | -6.3% |
| Ridge Regression | 0.2724 | alpha=1.0 | -0.04% |

#### Final Test Set Performance
- **RMSE**: 0.2090 (better than CV estimate)
- **MAE**: 0.1513
- **R² Score**: 0.8338
- **Model Generalisation**: Test RMSE < CV RMSE indicates good model generalisation

