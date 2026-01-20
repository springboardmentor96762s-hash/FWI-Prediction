# Milestone 2: Advanced Exploratory Data Analysis (EDA) - FWI Predictor

**Author:** Tanmay Hirodkar  
**Date:** December 2025  
**Internship:** FWI Prediction Project  

## 📋 Overview

This milestone builds upon Milestone 1 by conducting comprehensive exploratory data analysis (EDA) on the Forest Fire Weather Index dataset. The focus is on deep statistical analysis, advanced visualizations, target variable encoding, correlation studies, and identifying multicollinearity issues for future predictive modeling.

## 🎯 Objectives

1. **Enhanced Data Processing**: Advanced data loading with comprehensive info analysis
2. **Target Variable Engineering**: Binary encoding of fire occurrence classes
3. **Statistical Deep Dive**: Detailed descriptive statistics and missing value analysis
4. **Advanced Visualizations**: Multi-subplot histograms and custom correlation heatmaps
5. **Feature Relationship Analysis**: Correlation studies with target variable (FWI)
6. **Multicollinearity Detection**: Identifying highly correlated feature pairs

## 📁 Files Description

### Main Script
- **`main.py`**: Advanced EDA script with comprehensive statistical analysis

### Input Data
- **`ff2.csv`**: Forest fire dataset containing weather measurements and fire indices

### Generated Outputs
- **`Figure_1.png`**: Multi-panel histogram visualization of all features
- **`Figure_2.png`**: Custom correlation heatmap with annotated values
- Console outputs: Statistical summaries, correlation rankings, multicollinearity warnings

## 🔧 Technical Implementation

### Libraries Used
```python
import pandas as pd           # Data manipulation and analysis
import numpy as np            # Numerical computations and binary encoding
import matplotlib.pyplot as plt  # Advanced plotting and subplots
```

### Code Structure & Key Enhancements

#### 1. Advanced Data Loading & Inspection
```python
df = pd.read_csv("ff2.csv")
print(df.head())           # First 5 rows preview
print("Shape:", df.shape)   # Dataset dimensions
print(df.info())           # Data types and memory usage
```

#### 2. Target Variable Engineering
```python
df['Classes'] = df['Classes'].astype(str).str.strip()
df['Classes_binary'] = np.where(df['Classes'] == 'fire', 1, 0)
```
- **String Processing**: Handles whitespace and case sensitivity issues
- **Binary Encoding**: Converts 'fire'/'not fire' to 1/0 for modeling compatibility
- **Data Type Management**: Ensures consistent string format before encoding

#### 3. Comprehensive Feature Selection
```python
numeric_cols = [
    'day','month','year','Temperature','RH','Ws','Rain',
    'FFMC','DMC','DC','ISI','BUI','FWI'
]
```
- **13 Numerical Features**: Includes temporal, weather, and fire index variables
- **Robust Type Conversion**: Uses `pd.to_numeric()` with error handling
- **Missing Value Detection**: Systematic check across all features

#### 4. Advanced Statistical Analysis
- **Descriptive Statistics**: Mean, std, min, max, quartiles for all features
- **Missing Value Audit**: Complete assessment of data quality
- **Distribution Analysis**: Statistical summary with pandas `.describe()`

#### 5. Multi-Panel Histogram Visualization
```python
plt.figure(figsize=(18, 12))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(4, 4, i)
    plt.hist(df[col], bins=10, edgecolor='black')
```
- **Grid Layout**: 4x4 subplot arrangement for 13 features
- **Consistent Styling**: Black edges for better visual separation
- **Large Figure Size**: (18,12) for detailed visualization

#### 6. Custom Correlation Heatmap
```python
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                 ha='center', va='center', fontsize=8)
```
- **Manual Heatmap Creation**: Using `imshow()` for custom control
- **Value Annotations**: Correlation coefficients displayed on each cell
- **Color Mapping**: Cool-warm palette for intuitive correlation visualization

#### 7. Target Variable Correlation Analysis
```python
fwi_corr = corr_matrix['FWI'].sort_values(ascending=False)
print(fwi_corr.round(3))
```
- **FWI-Specific Analysis**: Ranking all features by correlation with Fire Weather Index
- **Sorted Output**: Descending order to identify strongest predictors

#### 8. Multicollinearity Detection
```python
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        if abs(corr_matrix.iloc[i, j]) > 0.85:
            print(numeric_cols[i], "<->", numeric_cols[j], ":", round(corr_matrix.iloc[i, j], 3))
```
- **Threshold-Based Detection**: |correlation| > 0.85 indicates high multicollinearity
- **Pairwise Analysis**: Systematic check of all feature combinations
- **Model Risk Assessment**: Identifying problematic feature pairs for regression

## 📊 Key Features Analyzed

### Temporal Features
- **day**, **month**, **year**: Date components for seasonal pattern analysis

### Weather Measurements  
- **Temperature**: Air temperature (°C)
- **RH**: Relative humidity (%)
- **Ws**: Wind speed (km/h)
- **Rain**: Precipitation (mm)

### Fire Weather Indices
- **FFMC**: Fine Fuel Moisture Code (surface litter moisture)
- **DMC**: Duff Moisture Code (organic layer moisture) 
- **DC**: Drought Code (deep organic layer moisture)
- **ISI**: Initial Spread Index (fire spread rate)
- **BUI**: Buildup Index (fuel consumption)
- **FWI**: Fire Weather Index (overall fire danger - TARGET)

### Target Variable
- **Classes**: Original categorical fire occurrence
- **Classes_binary**: Encoded binary target (1=fire, 0=no fire)

## 🎓 Student Learnings

### 1. **Advanced Data Processing Techniques**
- **String Manipulation**: Using `.str.strip()` for data cleaning
- **Conditional Encoding**: `np.where()` for efficient binary transformation
- **Type Conversion**: Robust handling of mixed data types with error management
- **Data Quality Assessment**: Systematic missing value and data type analysis

### 2. **Statistical Analysis Skills**
- **Descriptive Statistics**: Understanding distribution properties (mean, median, std, quartiles)
- **Missing Data Patterns**: Learning to identify and quantify data quality issues
- **Distribution Shapes**: Interpreting histogram patterns (normal, skewed, multimodal)

### 3. **Advanced Visualization Techniques**
- **Subplot Management**: Creating complex multi-panel layouts with matplotlib
- **Custom Heatmaps**: Manual correlation matrix visualization with annotations
- **Figure Sizing**: Optimizing plot dimensions for readability and detail
- **Color Mapping**: Using diverging color schemes for correlation data

### 4. **Feature Engineering & Selection**
- **Target Encoding**: Converting categorical outcomes to numerical format
- **Feature Categorization**: Organizing variables by type (temporal, weather, indices)
- **Correlation Analysis**: Understanding linear relationships between variables
- **Multicollinearity Detection**: Identifying redundant features before modeling

### 5. **Predictive Modeling Preparation**
- **Target Variable Creation**: Preparing binary classification targets
- **Feature Correlation Ranking**: Identifying most predictive variables
- **Redundancy Detection**: Finding highly correlated feature pairs (>0.85)
- **Data Quality Assurance**: Ensuring clean, consistent data for modeling

### 6. **Code Organization & Best Practices**
- **Modular Structure**: Clear separation of loading, processing, analysis, and visualization
- **Comprehensive Output**: Multiple analysis perspectives (visual, statistical, textual)
- **Documentation**: Detailed comments and section headers
- **Error Prevention**: Robust type conversion and data validation

## 🔍 Key Analysis Outputs

### Statistical Insights
1. **Dataset Dimensions**: Understanding the scale of available data
2. **Missing Value Assessment**: Data completeness evaluation
3. **Distribution Characteristics**: Statistical properties of each feature
4. **Correlation Rankings**: Features most/least correlated with FWI

### Visual Insights
1. **Feature Distributions**: Shape and spread patterns across all variables
2. **Correlation Patterns**: Visual identification of strong relationships
3. **Multicollinearity Visualization**: Heatmap highlighting redundant features

### Modeling Insights
1. **Target Variable Distribution**: Balance of fire vs. no-fire cases
2. **Predictor Importance**: FWI correlation rankings for feature selection
3. **Multicollinearity Warnings**: Feature pairs requiring attention in modeling

## 🚀 Advanced Techniques Demonstrated

### 1. **Binary Target Creation**
- Converting categorical fire occurrence to numerical binary format
- Essential preprocessing for classification algorithms

### 2. **Comprehensive Correlation Study**
- Matrix-wide correlation analysis
- Target-specific correlation ranking
- Multicollinearity threshold testing

### 3. **Custom Visualization Development**
- Manual heatmap construction with matplotlib
- Multi-panel histogram layouts
- Annotation integration for data-rich visualizations

### 4. **Systematic Data Quality Assessment**
- Missing value quantification across all features
- Data type consistency verification
- Statistical distribution analysis

## 💻 How to Run

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy matplotlib
   ```

2. **Prepare Data:**
   - Place `ff2.csv` in the same directory as `main.py`

3. **Execute Analysis:**
   ```bash
   python main.py
   ```

4. **Review Outputs:**
   - Console: Statistical summaries, correlation rankings, multicollinearity warnings
   - `Figure_1.png`: Feature distribution histograms
   - `Figure_2.png`: Correlation heatmap with annotations

## 🔄 Comparison with Milestone 1

| Aspect | Milestone 1 | Milestone 2 |
|--------|-------------|-------------|
| **Scope** | Basic EDA | Advanced Statistical Analysis |
| **Target Variable** | Removed classes | Binary encoding for modeling |
| **Visualizations** | Simple histograms & heatmap | Multi-panel layouts with annotations |
| **Correlation Analysis** | Basic matrix | Target-specific ranking + multicollinearity |
| **Statistical Depth** | Distribution overview | Comprehensive descriptive statistics |
| **Modeling Preparation** | Data cleaning focus | Feature selection & target preparation |

## 📝 Key Findings & Insights

1. **Data Quality**: Assessment of missing values and data completeness
2. **Feature Relationships**: Identification of strongly correlated variables
3. **Target Preparation**: Binary classification setup for fire occurrence prediction
4. **Multicollinearity Issues**: Detection of redundant features requiring attention
5. **Predictive Features**: Ranking of variables by correlation with FWI target

## 🎯 Next Steps (Future Milestones)

Based on this comprehensive EDA:
- **Feature Selection**: Use correlation rankings and multicollinearity analysis
- **Model Development**: Apply binary classification algorithms
- **Feature Engineering**: Create derived variables based on correlation insights
- **Model Validation**: Use statistical insights for robust model evaluation

---

*This README documents the advanced exploratory data analysis phase, providing deep statistical insights and preparing the foundation for predictive modeling in the FWI Prediction project.*
