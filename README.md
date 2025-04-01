# Loan Status Prediction with Optimized KNN and Ensemble Techniques

## Overview
This project aims to predict loan status (`loan_status`: 0 for non-default, 1 for default) using an optimized K-Nearest Neighbors (KNN) model and ensemble techniques (bagging) based on the "Loan Approval Classification Data" dataset from Kaggle. The workflow includes comprehensive data exploration, preprocessing, feature engineering, KNN modeling, hyperparameter tuning, cross-validation, and ensemble methods to achieve high accuracy (up to 99.05% with bagging). The goal is to deliver an accurate and robust solution for loan risk prediction.

## Dataset
- **Source**: [Loan Approval Classification Data on Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
- **File**: `loan_data.csv`

## Workflow and Steps

### 1. Exploratory Data Analysis (EDA)
- **Purpose**: Understand data structure, distribution, and characteristics.
- **Methods**:
  - Descriptive statistics, skewness, kurtosis, percentiles.
  - Visualizations: histograms, boxplots, correlation heatmap, class separability scatterplot.
- **Findings**:
  - High skewness in `person_income`, `loan_amnt`, etc.
    ```
    Skewness Kolom Numerik:
    person_age                     2.548154
    person_income                 34.137583
    person_emp_exp                 2.594917
    loan_amnt                      1.179731
    loan_int_rate                  0.213784
    loan_percent_income            1.034512
    cb_person_cred_hist_length     1.631720
    credit_score                  -0.610261
    dtype: float64
    ```
  - Class imbalance detected.
    ```
    --- Distribusi loan_status ---
    Jumlah per Kelas:
    loan_status
    0    35000
    1    10000
    Name: count, dtype: int64
    
    Proporsi per Kelas (%):
    loan_status
    0    77.777778
    1    22.222222
    Name: proportion, dtype: float64
    ```
  - Low correlation between numeric features, but separability observed (e.g., `credit_score` vs `loan_percent_income`).
- **Reason**: Identify challenges (skewness, outliers, imbalance) affecting KNN's distance-based predictions.

### 2. Data Preprocessing
#### a. Missing Value Imputation
- **Method**: Median for numeric, mode for categorical.
- **Reason**: KNN cannot handle missing values; median is robust to outliers, mode suits categorical data.
- **Alternatives Rejected**: Mean (outlier-sensitive), KNN Imputation (computationally expensive).

#### b. Duplicate Removal
- **Method**: `drop_duplicates()`.
- **Reason**: Duplicates bias KNN by over-representing identical points in nearest neighbors.
- **Alternatives Rejected**: No realistic alternative.

#### c. Categorical Encoding
- **Method**: One-hot encoding with `get_dummies`.
- **Reason**: KNN requires numeric data; one-hot encoding suits non-ordinal categorical variables.
- **Alternatives Rejected**: Label Encoding (assumes ordinality), Target Encoding (risk of data leakage).

#### d. Feature Engineering
- **Method**: Added `income_to_loan_ratio`, `credit_to_income`, `age_to_credit_length`.
- **Reason**: Capture domain-specific relationships to enhance class separability.
- **Alternatives Rejected**: Polynomial features (increase dimensionality).

#### e. Skewness Transformation
- **Method**: `log1p` on skewed columns (e.g., `person_income`).
  ```
  --- Skewness Sebelum Transformasi ---
  Skewness per kolom:
  person_age                     2.548154
  person_income                 34.137583
  person_emp_exp                 2.594917
  loan_amnt                      1.179731
  loan_percent_income            1.034512
  cb_person_cred_hist_length     1.631720
  dtype: float64
  
  Transformasi untuk person_age:
  Skewness sebelum: 2.55
  Skewness sesudah: 1.22
  
  Transformasi untuk person_income:
  Skewness sebelum: 34.14
  Skewness sesudah: 0.22
  
  Transformasi untuk person_emp_exp:
  Skewness sebelum: 2.59
  Skewness sesudah: -0.12
  
  Transformasi untuk loan_amnt:
  Skewness sebelum: 1.18
  Skewness sesudah: -0.44
  
  Transformasi untuk loan_percent_income:
  Skewness sebelum: 1.03
  Skewness sesudah: 0.83
  
  Transformasi untuk cb_person_cred_hist_length:
  Skewness sebelum: 1.63
  Skewness sesudah: 0.44
  
  --- Skewness Sesudah Transformasi ---
  Skewness per kolom:
  person_age                    1.220204
  person_income                 0.224018
  person_emp_exp               -0.122990
  loan_amnt                    -0.438128
  loan_percent_income           0.829301
  cb_person_cred_hist_length    0.442656
  dtype: float64
  ```
- **Reason**: Reduce skewness for more meaningful KNN distance calculations.
- **Alternatives Rejected**: Box-Cox (complex), Power Transform (less intuitive).

#### f. Outlier Removal
- **Method**: Isolation Score, removing top 10% outliers.
- **Reason**: Outliers distort KNN distances; Isolation Score is effective for multi-dimensional data.
- **Alternatives Rejected**: Z-Score/IQR (univariate), DBSCAN (slow).

#### g. Standardization
- **Method**: Min-Max Scaling to [0, 1].
- **Reason**: Ensure uniform feature scales for fair KNN distance computation; compatible with binary features.
- **Alternatives Rejected**: Z-Score (unbounded, assumes normality).

#### h. SMOTE
- **Method**: Synthetic Minority Oversampling for class 1, up to 75% of majority-minority gap.
  ```
  --- Distribusi Kelas Sebelum SMOTE ---
  Jumlah per kelas:
  loan_status
  0    32223
  1     8242
  Name: count, dtype: int64
  
  --- Distribusi Kelas Sesudah SMOTE ---
  Jumlah per kelas:
  loan_status
  0.0    32223
  1.0    26227
  Name: count, dtype: int64
  SMOTE: Menambahkan 17985 sampel sintetis
  ```
- **Reason**: Address class imbalance to prevent KNN bias toward majority class.
- **Alternatives Rejected**: Random Oversampling (less realistic), Undersampling (data loss).

#### i. Train-Test Split
- **Method**: Stratified split (80% train, 20% test).
- **Reason**: Maintain class distribution for representative KNN evaluation.
- **Alternatives Rejected**: Random split (risk of imbalance).

### 3. KNN Modeling
#### a. KNN Implementation
- **Method**: Custom `knn_predict` with `k`, `distance`, and `weights` parameters.
- **Reason**: Flexibility for tuning and evaluation.
- **Initial Results**:
  - Accuracy: 0.9061
  - Confusion Matrix: [[5562 883] [215 5030]]
  - Precision: (0.9628, 0.8507), Recall: (0.8630, 0.9590), F1: (0.9102, 0.9016)

#### b. Hyperparameter Tuning
- **Method**: Manual iteration over `k` (3-21), `distance` (Euclidean/Manhattan), `weights` (uniform, distance, gaussian).
- **Reason**: Optimize KNN for maximum accuracy.
- **Best Results**:
  - Parameters: k=3, Euclidean, gaussian
  - Accuracy: 0.9234
  - Confusion Matrix: [[5860 585] [311 4934]]
  - Precision: (0.9496, 0.8940), Recall: (0.9092, 0.9407), F1: (0.9290, 0.9168)
- **Alternatives Rejected**: Automated Grid Search (less transparent).

#### c. Cross-Validation
- **Method**: Stratified 5-fold CV.
- **Reason**: Robust performance estimation, reduces overfitting risk.
- **Alternatives Rejected**: Standard K-Fold (doesn’t preserve class ratio).

### 4. Ensemble with Bagging
- **Method**: Bagging with 10 KNN models on bootstrap samples.
- **Reason**: Reduce KNN variance and enhance stability.
- **Results**:
  - Accuracy: 0.9905
  - Confusion Matrix: [[6405 40] [71 5174]]
  - Precision: (0.9890, 0.9923), Recall: (0.9938, 0.9865), F1: (0.9914, 0.9894)
- **Alternatives Rejected**: Boosting (unsuitable for KNN).

### 5. Evaluation
- **Method**: Confusion matrix, accuracy, precision, recall, F1-score.
- **Reason**: Comprehensive performance assessment, focusing on minority class.

## Key Results
- **Initial KNN**: Accuracy 90.61%
  ```
  --- Confusion Matrix Awal ---
  Confusion Matrix:
  [[5562  883]
   [ 215 5030]]
  
  --- Metrik Evaluasi Awal ---
  Accuracy: 0.9061
  Precision (kelas 0, kelas 1): 0.9628, 0.8507
  Recall (kelas 0, kelas 1): 0.8630, 0.9590
  F1-Score (kelas 0, kelas 1): 0.9102, 0.9016
  ```
  
  ![image](https://github.com/user-attachments/assets/ac07a877-263f-46e7-8e9f-69406dbc1510)


  ```
  --- Hasil Cross-Validation ---
  
  --- Memulai Cross-Validation ---
  Jumlah split: 5
  
  Fold 1 dari 5
  
  --- Memulai Prediksi KNN ---
  Jumlah data train: 46761, Jumlah data test: 11689
  Parameter: k=5, distance=euclidean, weights=distance
  Prediksi 0 dari 11689 selesai.
  Prediksi 1000 dari 11689 selesai.
  Prediksi 2000 dari 11689 selesai.
  Prediksi 3000 dari 11689 selesai.
  Prediksi 4000 dari 11689 selesai.
  Prediksi 5000 dari 11689 selesai.
  Prediksi 6000 dari 11689 selesai.
  Prediksi 7000 dari 11689 selesai.
  Prediksi 8000 dari 11689 selesai.
  Prediksi 9000 dari 11689 selesai.
  Prediksi 10000 dari 11689 selesai.
  Prediksi 11000 dari 11689 selesai.
  
  Prediksi KNN selesai.
  Akurasi fold 1: 0.9063
  
  Fold 2 dari 5
  
  --- Memulai Prediksi KNN ---
  Jumlah data train: 46761, Jumlah data test: 11689
  Parameter: k=5, distance=euclidean, weights=distance
  Prediksi 0 dari 11689 selesai.
  Prediksi 1000 dari 11689 selesai.
  Prediksi 2000 dari 11689 selesai.
  Prediksi 3000 dari 11689 selesai.
  Prediksi 4000 dari 11689 selesai.
  Prediksi 5000 dari 11689 selesai.
  Prediksi 6000 dari 11689 selesai.
  Prediksi 7000 dari 11689 selesai.
  Prediksi 8000 dari 11689 selesai.
  Prediksi 9000 dari 11689 selesai.
  Prediksi 10000 dari 11689 selesai.
  Prediksi 11000 dari 11689 selesai.
  
  Prediksi KNN selesai.
  Akurasi fold 2: 0.9069
  
  Fold 3 dari 5
  
  --- Memulai Prediksi KNN ---
  Jumlah data train: 46761, Jumlah data test: 11689
  Parameter: k=5, distance=euclidean, weights=distance
  Prediksi 0 dari 11689 selesai.
  Prediksi 1000 dari 11689 selesai.
  Prediksi 2000 dari 11689 selesai.
  Prediksi 3000 dari 11689 selesai.
  Prediksi 4000 dari 11689 selesai.
  Prediksi 5000 dari 11689 selesai.
  Prediksi 6000 dari 11689 selesai.
  Prediksi 7000 dari 11689 selesai.
  Prediksi 8000 dari 11689 selesai.
  Prediksi 9000 dari 11689 selesai.
  Prediksi 10000 dari 11689 selesai.
  Prediksi 11000 dari 11689 selesai.
  
  Prediksi KNN selesai.
  Akurasi fold 3: 0.9119
  
  Fold 4 dari 5
  
  --- Memulai Prediksi KNN ---
  Jumlah data train: 46761, Jumlah data test: 11689
  Parameter: k=5, distance=euclidean, weights=distance
  Prediksi 0 dari 11689 selesai.
  Prediksi 1000 dari 11689 selesai.
  Prediksi 2000 dari 11689 selesai.
  Prediksi 3000 dari 11689 selesai.
  Prediksi 4000 dari 11689 selesai.
  Prediksi 5000 dari 11689 selesai.
  Prediksi 6000 dari 11689 selesai.
  Prediksi 7000 dari 11689 selesai.
  Prediksi 8000 dari 11689 selesai.
  Prediksi 9000 dari 11689 selesai.
  Prediksi 10000 dari 11689 selesai.
  Prediksi 11000 dari 11689 selesai.
  
  Prediksi KNN selesai.
  Akurasi fold 4: 0.9111
  
  Fold 5 dari 5
  
  --- Memulai Prediksi KNN ---
  Jumlah data train: 46756, Jumlah data test: 11694
  Parameter: k=5, distance=euclidean, weights=distance
  Prediksi 0 dari 11694 selesai.
  Prediksi 1000 dari 11694 selesai.
  Prediksi 2000 dari 11694 selesai.
  Prediksi 3000 dari 11694 selesai.
  Prediksi 4000 dari 11694 selesai.
  Prediksi 5000 dari 11694 selesai.
  Prediksi 6000 dari 11694 selesai.
  Prediksi 7000 dari 11694 selesai.
  Prediksi 8000 dari 11694 selesai.
  Prediksi 9000 dari 11694 selesai.
  Prediksi 10000 dari 11694 selesai.
  Prediksi 11000 dari 11694 selesai.
  
  Prediksi KNN selesai.
  Akurasi fold 5: 0.9060
  
  Cross-Validation selesai.
  Mean Accuracy: 0.9085
  Standard Deviation: 0.0025
  ```
- **Optimized KNN**: Accuracy 92.34%
  ```
  --- Hasil Tuning Hyperparameter ---
  Parameter terbaik: k=3, distance=euclidean, weights=gaussian, Akurasi=0.9234
  
  --- Prediksi dengan Parameter Terbaik ---
  
  --- Memulai Prediksi KNN ---
  Jumlah data train: 46760, Jumlah data test: 11690
  Parameter: k=3, distance=euclidean, weights=gaussian
  Prediksi 0 dari 11690 selesai.
  Prediksi 1000 dari 11690 selesai.
  Prediksi 2000 dari 11690 selesai.
  Prediksi 3000 dari 11690 selesai.
  Prediksi 4000 dari 11690 selesai.
  Prediksi 5000 dari 11690 selesai.
  Prediksi 6000 dari 11690 selesai.
  Prediksi 7000 dari 11690 selesai.
  Prediksi 8000 dari 11690 selesai.
  Prediksi 9000 dari 11690 selesai.
  Prediksi 10000 dari 11690 selesai.
  Prediksi 11000 dari 11690 selesai.
  
  Prediksi KNN selesai.
  
  Confusion Matrix Terbaik:
  [[5860  585]
   [ 311 4934]]
  
  --- Metrik Evaluasi Terbaik ---
  Accuracy: 0.9234
  Precision (kelas 0, kelas 1): 0.9496, 0.8940
  Recall (kelas 0, kelas 1): 0.9092, 0.9407
  F1-Score (kelas 0, kelas 1): 0.9290, 0.9168
  ```

  ![image](https://github.com/user-attachments/assets/90787ecb-1b56-4d4e-8b5a-a3fddd78555c)

- **Bagging KNN**: Accuracy 99.05% - highest performance, showcasing ensemble strength.
  ```
  Confusion Matrix Ensemble:
  [[6405   40]
   [  71 5174]]
  
  --- Metrik Evaluasi Ensemble ---
  Accuracy: 0.9905
  Precision (kelas 0, kelas 1): 0.9890, 0.9923
  Recall (kelas 0, kelas 1): 0.9938, 0.9865
  F1-Score (kelas 0, kelas 1): 0.9914, 0.9894
  ```

  ![image](https://github.com/user-attachments/assets/c75bdf85-e6c5-4d1a-ba40-473715c282dd)


## Why These Methods?
- **KNN**: Simple, flexible, non-parametric, ideal for baseline modeling.
- **Preprocessing**: Addresses skewness, outliers, imbalance, and scale to ensure accurate KNN distances.
- **Bagging**: Boosts stability and accuracy by reducing variance.

## Additional Insights and Suggestions
1. **Feature Selection**: Consider Recursive Feature Elimination (RFE) to reduce dimensionality further.
2. **Distance Metrics**: Experiment with cosine similarity for mixed data types.
3. **Advanced Tuning**: Use Bayesian Optimization for efficient hyperparameter search.
4. **Benchmarking**: Compare with Random Forest or XGBoost for broader insights.
5. **Deployment**: Implement a Flask or FastAPI API for real-time predictions.

## Free to clone
1. Clone the repository:
   ```bash
   git clone https://github.com/nnichaelangello/Loan-Status-Prediction-with-Optimized-KNN-and-Ensemble-Techniques.git
   cd Loan-Status-Prediction-with-Optimized-KNN-and-Ensemble-Techniques
   ```

## Requirements
- Python 3.8+
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`

## Contributing
Feel free to open an issue or submit a pull request for suggestions or improvements.
