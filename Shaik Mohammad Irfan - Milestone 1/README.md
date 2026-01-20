# Tempest FWI Predictor - Milestone 1: Data Preprocessing

## Overview
This folder covers preprocessing the FWI dataset for linear regression modeling.

## Steps Achieved
1. **Downloaded FWI Data**: Used FWI_UPDATE.csv (Bejaia region, 2012).
2. **Cleaned with Pandas**: Handled missings via linear interpolation; ensured no categoricals in numerics.
3. **Analyzed Data**: Histograms show distributions (e.g., FWI right-skewed); correlations highlight predictors (e.g., ISI +0.91 with FWI).

## Files
- Data Preprocessing.py: Cleaning script.
- Histograms and Correlation.py: EDA with plots.
- cleaned_fwi_dataset.csv: Processed data (244 rows, 0 missings).
- histogram-features.png / correlation_heatmap.png: Visuals.
- week 1 & 2 learnings.pdf: Regression fundamentals.

Run: `python Data Preprocessing.py` then `python Histograms and Correlation.py`.