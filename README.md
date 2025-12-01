Milestone 1 – FWI Prediction
In this milestone, I performed exploratory data analysis (EDA) on the Portuguese Forest Fires Dataset to understand the relationships between environmental features and the Fire Weather Index (FWI).
1.The tasks included:

Loading the dataset

Cleaning and preprocessing

Generating histograms

Creating correlation plots

Selecting important features

📂 2. Files Included

Inside this milestone folder, you will find:

Raw Dataset
The original CSV file used for analysis.

Python Scripts

Data loading & inspection

Preprocessing & missing value handling

Histogram generation

Correlation heatmap

Boxplots & pairplots

Feature selection

Output Files
Saved visualizations (histograms, heatmaps, etc.)

This README
Explaining everything I have done.

🧪 3. Steps Performed
Step 1: Data Loading

Loaded the dataset using Pandas and printed:

First 5 rows

Dataset info

Summary statistics

Step 2: Missing Values Check

Checked missing values in:

The whole dataset

Individual columns (especially temp)

Step 3: Data Cleaning

Dropped unnecessary columns (X, Y)

Renamed or corrected column names where needed

Selected key columns for FWI analysis:

FFMC, DMC, ISI, temp, wind, FWI

Step 4: Visualizations Created

I generated multiple visual insights:

📊 Histograms

To understand distribution of numeric features.

📦 Boxplots

To detect outliers and spread.

🔥 Correlation Heatmap

To find relationships between variables such as:

temp vs FWI

wind vs FWI

🔗 Pairplot

To visually inspect feature relationships.

📈 4. Insights from Analysis

Some features show strong correlation with FWI.

Temperature and wind appear to influence fire intensity.

Distributions show certain skewed variables.

These insights will help in building the machine learning model in upcoming milestones.

🚀 5. Conclusion

This milestone focused on:

Understanding the dataset

Cleaning & preparing it
