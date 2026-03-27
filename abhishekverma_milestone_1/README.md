##  Project Overview

This project focuses on preprocessing, cleaning, and performing exploratory data analysis (EDA) on a forest fire dataset to prepare it for predictive modeling. The goal is to ensure the data is reliable, structured, and insightful for understanding fire occurrences across different regions.

Key Steps Undertaken

Data Loading

Imported the dataset using pandas and reviewed the first few records to understand its structure and content.

Handling Missing Values

Numeric missing values were filled with column mean.

Categorical missing values were filled with mode, ensuring no data loss and a complete dataset.

Encoding Categorical Features

Converted the target column Classes to numeric: fire=1, not fire=0.

Encoded Region manually to numeric values (East=0, North=1, South=2, West=3) for computational efficiency.

Data Verification

Checked missing values after cleaning and verified that all columns are complete and ready for modeling.

Exploratory Data Analysis (EDA)

Plotted histograms for all numeric features to understand distributions.

Generated a correlation matrix and heatmap to explore relationships between features, aiding in feature selection and modeling decisions.

Data Export

Saved the cleaned and encoded dataset as cleaned_forest_fire_dataset.csv for future machine learning tasks.

Key Impact

Transformed raw, incomplete data into a clean, structured dataset ready for predictive modeling.

Provided visual insights into feature distributions and correlations to guide feature engineering.

Established a solid foundation for building accurate forest fire prediction models.