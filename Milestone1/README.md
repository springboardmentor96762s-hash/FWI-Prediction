# Milestone 1 Submission: Mishti Kalra

## Project: Tempest FWI Predictor – A Machine Learning Model to Predict Fire Weather

Completed data preprocessing milestone for FWI regression modeling.

## List of Steps to Achieve the Goal
1. **Downloading FWI Data**: Sourced FWI_UPDATE.csv (Bejaia region, 2012 summer; ~246 rows with features like Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, and target FWI).
2. **Using Pandas to Clean the Data**: 
   - Skipped header, stripped columns, added 'Region' as 'Bejaia'.
   - Converted numerics using loc command.
   - Created 'date' column, sorted chronologically.
   - Handled missings with linear interpolation (mean of neighbors – preserves time-series trends).
   - Dropped invalids; result: 244 rows, 0 missings, no categoricals in numerics.
3. **Analyzing the Data**: 
   - Histograms: Distributions (e.g., FWI right-skewed, skew~1.2; Rain zero-inflated).
   - Correlation: Pearson matrix (e.g., ISI +0.91 with FWI – strong positive; RH -0.57 – strong negative).

## Learnings from Week 1 and Week 2
1. **Understanding Linear Regression**: Supervised model for continuous FWI prediction: $$ FWI = \beta_0 + \beta_1 \cdot Temp + \beta_2 \cdot RH + \dots + \epsilon $$. Assumptions: linearity (from corr), normality (from histograms), homoscedasticity.
2. **How the Solution for Regressors is Obtained**: Ordinary Least Squares (OLS) minimizes MSE: $$ \hat{\beta} = (X^T X)^{-1} X^T y $$ (matrix form; e.g., β_ISI ~0.45 from high corr).
3. **Systems of Linear Equations**: Normal equations $$ X^T X \beta = X^T y $$; solve via inversion (e.g., for 2 features: nβ₀ + Σx β₁ = Σy). Handles multicollinearity (e.g., DMC-DC >0.9).

## Files Included
- `Algerian_forest_fires_dataset_CLEANED`: Processed dataset (ready for regression).
- `Data Preprocessing.ipynb':Preprocessing code 
- 'Histogram Plots and Correlation Matrix.ipynb`: EDA script (histograms/correlation plots).

## Key Insights for Regression
- Skewed FWI: Suggest log-transform.
- Top Predictors: ISI/DMC (positive corr) for feature selection.
- Multicollinearity: Combine DMC/BUI/DC.
