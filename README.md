# California Wildfire Incident Analysis (2013-2019): Predictive Modeling Comparison

This repository contains a Google Colab notebook (`.ipynb`) that performs a comprehensive analysis of historical California wildfire incidents between 2013 and 2019. The primary goal was to assess the feasibility of predicting wildfire outcomes (final size and high-risk classification) using commonly available incident report data and to evaluate the impact of data limitations.

## 1. Overview & Context

Wildfires pose a significant and growing threat in California. Understanding the factors that influence their size and potential risk is crucial for effective management and public safety. This project leverages a publicly available dataset to:

1.  Explore historical fire incident patterns.
2.  Clean and preprocess the data, addressing significant quality issues like missing values.
3.  Engineer relevant features from time and location data.
4.  Build and compare machine learning models for two distinct tasks:
    *   **Regression:** Predicting the final `AcresBurned` (using a log transformation).
    *   **Classification:** Predicting whether a fire will exceed a 1000-acre threshold ('High Risk').
5.  Evaluate model performance using appropriate metrics.
6.  Identify key predictive features within the available data.
7.  Discuss the limitations encountered, particularly those related to data quality, and suggest future directions.

## 2. Dataset

*   **Source:** California Wildfire Incidents 2013-2020 dataset from Kaggle.
    *   Link: [https://www.kaggle.com/datasets/ananthu017/california-wildfire-incidents-20132020](https://www.kaggle.com/datasets/ananthu017/california-wildfire-incidents-20132020)
    *   *Note: The analysis primarily focused on incidents within the 2013-2019 timeframe based on initial data exploration.*
*   **Key Features Used:** `Latitude`, `Longitude`, `Started` (for time features), `AdminUnit`, `Counties`, `CalFireIncident`, `MajorIncident`.
*   **Target Variables:**
    *   `AcresBurned` (Original, used to derive targets)
    *   `AcresBurned_log` (Log-transformed for regression)
    *   `RiskLevel_1000` (Binary classification: > 1000 acres = 1, else 0)

## 3. Objectives

*   Assess the predictive accuracy achievable for final fire size (regression) and high-risk classification (>1000 acres) using the available historical data.
*   Identify the most influential features driving these predictions within the dataset's constraints.
*   Compare different preprocessing strategies (coordinate imputation vs. dropping) and modeling algorithms (Linear, Random Forest, Gradient Boosting).
*   Quantify the limitations imposed by data quality, particularly missing values.

## 4. Requirements & Dependencies

The analysis was performed in a Google Colab environment (Python 3.x). Key libraries required:

*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `category_encoders` (install via `!pip install category_encoders`)
*   `lightgbm` (install via `!pip install lightgbm`)
*   `catboost` (install via `!pip install catboost`)
*   `missingno` (install via `!pip install missingno`)
*   `scipy`

The notebook includes installation commands (`!pip install ... -q`).

## 5. How to Run

1.  **Environment:** Open the `.ipynb` file in Google Colab or a local environment with the required libraries installed.
2.  **Data:** Download the `California_Fire_Incidents 7 (1).csv` file from the Kaggle source. Upload it to your Colab session's root directory (`/content/`).
3.  **File Path:** Ensure the `FILE_PATH` variable in the "Configuration" section of the code correctly points to `/content/California_Fire_Incidents 7 (1).csv`.
4.  **Run Cells:** Execute the notebook cells sequentially from top to bottom.
    *   Cells are grouped by analysis step (Setup, Load Data, EDA, Preprocessing, Regression, Classification, Visualization).
    *   **Note:** The hyperparameter tuning cells (`RandomizedSearchCV`) can take a significant amount of time to run (potentially 15-30 minutes or more per model depending on Colab resources).
5.  **Outputs:**
    *   Results (tables, metrics) will be printed directly in the notebook output.
    *   Generated plots (Missing Values, Confusion Matrix, Feature Importance) will be saved to the `/content/results/` directory within the Colab session and also displayed inline.

## 6. Methodology Summary

1.  **Data Loading & EDA:** Loaded the dataset, standardized column names, performed initial exploration including missing value analysis (Figure 1) and target variable distribution analysis (Figure 2 justification).
2.  **Cleaning:** Dropped columns with >80% missing values (e.g., resources, impacts, fuel type) and irrelevant/redundant columns. Removed records with missing target or invalid start dates.
3.  **Feature Engineering:** Extracted year, day-of-week, and cyclical sine/cosine features for month and day-of-year from the `Started` date.
4.  **Coordinate Handling:** Cleaned invalid coordinates. Compared dropping rows with missing coordinates vs. KNN imputation for the regression task. Used the "drop missing" strategy for the final classification task based on prior results.
5.  **Encoding:** Grouped rare categories (<10 occurrences) in `AdminUnit` and `Counties`. Applied Target Encoding within pipelines. Converted boolean flags to integers.
6.  **Target Definition:** Used `log1p(AcresBurned)` for regression and a binary flag (`AcresBurned > 1000`) for classification.
7.  **Modeling Pipelines:** Used `scikit-learn` Pipelines to combine preprocessing (imputation, scaling for KNN/linear, encoding) and model fitting.
8.  **Model Training & Tuning:** Trained and tuned Ridge/Logistic Regression, RandomForest, CatBoost, and LightGBM (for classification) using `RandomizedSearchCV` with 5-fold CV.
9.  **Evaluation:** Assessed regression models using R², MAE (log and original scale), and RMSE (log scale). Assessed classification models using ROC AUC, Accuracy, Precision, Recall, F1-score (especially for the 'High Risk' class), and Confusion Matrices.

## 7. Key Findings & Results Summary

*   **Data Quality:** The dataset suffers from severe missingness (>80%) in critical variables like resource deployment, structural impacts, and fuel type, fundamentally limiting predictive potential (Figure 1).
*   **Regression Performance:** Predicting exact (log-transformed) `AcresBurned` proved difficult. The best model (tuned CatBoost with KNN imputed coordinates) achieved a Test R² of only **~0.28**, with a high MAE of **~4200 acres** (Table 1, Figure 3 - Actual vs Predicted plot).
*   **Classification Performance:** Classifying fires as 'High Risk' (>1000 acres) yielded moderate discriminative ability (best Test ROC AUC **~0.80** by LGBM/CatBoost/RF). However, models struggled to reliably identify these critical events, with low **Recall (~0.46)** and low **Precision (~0.47-0.55)** for the 'High Risk' class (Table 2, Figure 4 - Confusion Matrix).
*   **Feature Importance:** Across tasks and models, **location** (Latitude, Longitude, encoded AdminUnit/County) and **time** (Start Year, DayOfYear cyclical features, DayOfWeek) were consistently the most influential predictors among the available data (Figure 5 - Classification Importance). The `MajorIncident` flag also showed relevance.
*   **Main Conclusion:** Despite robust preprocessing and advanced modeling techniques, the predictive accuracy for both fire size and high-risk classification is low, primarily constrained by the **lack of essential information** in the dataset.

## 8. Limitations

*   **Missing Data:** The exclusion of heavily missing, potentially crucial predictors (resources, impacts, fuel) is the primary limitation.
*   **Static Data:** The dataset reflects final incident summaries, lacking dynamic real-time information (weather, spread).
*   **Coordinate Issues:** ~20% missing/invalid coordinates required dropping data or imputation.
*   **Encoding Choices:** Target Encoding performance can be sensitive to parameters and implementation details.
*   **Limited Timeframe:** The 2013-2019 focus might not capture all long-term trends.

## 9. Future Work

*   **Data Enrichment:** Integrate with external datasets (weather, fuel maps, topography, WUI, lightning).
*   **Improved Data Collection:** Advocate for more complete and standardized reporting in official incident databases.
*   **Advanced Feature Engineering:** Develop more sophisticated spatial and temporal features.
*   **Specialized Models:** Explore spatio-temporal or specialized deep learning models *if* data quality significantly improves.

---
