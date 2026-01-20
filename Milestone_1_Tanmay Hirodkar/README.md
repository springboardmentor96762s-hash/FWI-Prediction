# Milestone 1: Forest Fire Weather Index (FWI) - Initial Data Analysis

**Author:** Tanmay Hirodkar  
**Date:** December 2025  
**Internship:** FWI Prediction Project  

## 📋 Overview

This milestone focuses on performing initial exploratory data analysis (EDA) on the Forest Fire Weather Index dataset. The main objective is to understand the data structure, clean the dataset, and generate basic visualizations to identify patterns and relationships between different fire weather variables.

## 🎯 Objectives

1. **Data Loading & Preprocessing**: Load the forest fire dataset and handle data quality issues
2. **Data Cleaning**: Convert data types, handle missing values, and remove unnecessary columns
3. **Visual Analysis**: Generate histograms to understand feature distributions
4. **Correlation Analysis**: Create correlation matrix to identify relationships between variables

## 📁 Files Description

### Main Script
- **`main2.py`**: Primary analysis script containing all data processing and visualization code

### Input Data
- **`ff2.csv`**: Forest fire dataset containing weather measurements and fire indices

### Generated Outputs
- **`fwi_histograms_short.png`**: Histogram plots showing distribution of all fire weather features
- **`fwi_correlation_short.png`**: Correlation heatmap showing relationships between variables

## 🔧 Technical Implementation

### Libraries Used
```python
import pandas as pd      # Data manipulation and analysis
import numpy as np       # Numerical computations
import matplotlib.pyplot as plt  # Basic plotting
import seaborn as sns    # Advanced statistical visualizations
```

### Code Structure

#### 1. Data Loading
```python
df = pd.read_csv(FILE_NAME)
```
- Implements error handling for missing files
- Uses try-except block to gracefully handle FileNotFoundError

#### 2. Data Preprocessing & Cleaning
- **Type Conversion**: Converts all columns (except date and class columns) to numeric format
- **Missing Value Handling**: Fills NaN values with column means using `fillna(df.mean())`
- **Feature Selection**: Removes non-predictive columns (`Classes`, `day`, `month`, `year`)
- **Error Handling**: Uses `pd.to_numeric()` with `errors='coerce'` for robust type conversion

#### 3. Visualization Generation

**Histogram Analysis:**
- Creates distribution plots for all numerical features
- Uses 20 bins for detailed distribution visualization
- Saves output as `fwi_histograms_short.png`

**Correlation Analysis:**
- Generates correlation matrix using `df.corr()`
- Creates heatmap with Seaborn for better visualization
- Includes correlation coefficients with 2 decimal precision
- Saves output as `fwi_correlation_short.png`

## 📊 Key Features Analyzed

The analysis focuses on fire weather variables including:
- **Temperature**: Air temperature measurements
- **RH**: Relative humidity levels
- **Ws**: Wind speed measurements  
- **Rain**: Precipitation amounts
- **FFMC**: Fine Fuel Moisture Code
- **DMC**: Duff Moisture Code
- **DC**: Drought Code
- **ISI**: Initial Spread Index
- **BUI**: Buildup Index
- **FWI**: Fire Weather Index (target variable)

## 🎓 Student Learnings

### 1. **Data Quality Management**
- Learned importance of handling different data types in real-world datasets
- Understanding how `pd.to_numeric()` with `errors='coerce'` handles conversion failures
- Realized significance of missing value treatment using statistical measures (mean imputation)

### 2. **Exploratory Data Analysis Fundamentals**
- **Distribution Analysis**: Histograms revealed the shape and spread of each variable
- **Relationship Discovery**: Correlation heatmaps helped identify which variables are related
- **Feature Engineering**: Learned to exclude non-predictive features (dates, categorical classes)

### 3. **Visualization Best Practices**
- **Matplotlib**: Understood subplot management and figure sizing
- **Seaborn**: Learned advanced heatmap customization with annotations
- **Output Management**: Practiced saving plots programmatically for documentation

### 4. **Code Organization & Error Handling**
- Structured code with clear sections (Loading → Preprocessing → Analysis)
- Implemented robust error handling for file operations
- Used descriptive variable names and comments for code readability

### 5. **Statistical Understanding**
- **Correlation Interpretation**: Learned how correlation coefficients indicate linear relationships
- **Distribution Shapes**: Identified normal, skewed, and bimodal distributions in the data
- **Data Preprocessing Impact**: Understood how cleaning steps affect analysis results

## 🔍 Key Insights from Analysis

1. **Data Structure**: Successfully processed fire weather measurements with multiple indices
2. **Missing Values**: Handled missing data using mean imputation strategy
3. **Feature Relationships**: Correlation analysis revealed which weather factors are most interconnected
4. **Distribution Patterns**: Histograms showed the statistical properties of each fire weather variable

## 🚀 Next Steps (Milestone 2)

Based on this initial analysis, the next milestone should focus on:
- More detailed statistical analysis
- Advanced visualization techniques
- Feature selection for predictive modeling
- Correlation analysis specifically with target variables

## 💻 How to Run

1. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

2. Place `ff2.csv` in the same directory as `main2.py`

3. Execute the script:
   ```bash
   python main2.py
   ```

4. Check generated visualizations: `fwi_histograms_short.png` and `fwi_correlation_short.png`

## 📝 Notes

- This milestone serves as the foundation for understanding the FWI dataset
- All preprocessing decisions (like removing date columns) were made to focus on numerical relationships
- The correlation analysis will inform feature selection in subsequent modeling phases
- Error handling ensures the script runs robustly across different environments

---

*This README documents the initial data exploration phase of the FWI Prediction project, establishing the groundwork for advanced analysis in upcoming milestones.*
