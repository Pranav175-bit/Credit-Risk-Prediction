# Credit Risk Prediction Using Machine Learning

## Project Overview

This project focuses on predicting whether a loan applicant is likely to be risky or non-risky using machine learning. Credit risk prediction is important for banks and financial institutions because it helps identify high-risk applicants before loan approval.

The project includes data cleaning, exploratory data analysis, feature transformation, preprocessing, model training, model comparison, final model selection, business interpretation, and deployment preparation.

## Problem Statement

The objective of this project is to build a machine learning model that predicts loan applicant credit risk based on financial, employment, loan, and credit-related features.

The target variable is `Status`:

- `0` = Non-risky applicant
- `1` = Risky applicant

## Dataset Description

The dataset contains information about loan applicants, including income, employment length, loan amount, interest rate, home ownership, loan intent, previous default history, and credit history length.

| Column | Description |
|---|---|
| `Age` | Age of the loan applicant |
| `Income` | Annual income of the applicant |
| `Home` | Home ownership status |
| `Emp_length` | Employment length of the applicant |
| `Intent` | Purpose of the loan |
| `Amount` | Loan amount requested |
| `Rate` | Interest rate of the loan |
| `Status` | Target variable indicating loan risk |
| `Percent_income` | Loan amount as a percentage of income |
| `Default` | Previous loan default history; whether the applicant has failed to repay a loan before |
| `Cred_length` | Credit history length |

## Tools and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Jupyter Notebook

## Project Workflow

1. Problem Statement
2. Dataset Overview
3. Data Inspection
4. Data Cleaning
5. Exploratory Data Analysis
6. Data Preprocessing and Feature Engineering
7. Model Building and Evaluation
8. Final Model Selection
9. Business Interpretation
10. Conclusion
11. Prediction on New Applicant Data
12. Saving Model and Preprocessing Objects

## Data Cleaning

The dataset was cleaned by:

- Removing unnecessary identifier columns
- Handling missing values in `Emp_length` and `Rate`
- Removing clearly unrealistic values such as very high age and employment length
- Retaining valid financial outliers because they may represent real applicant behavior

## Exploratory Data Analysis

EDA was performed to understand the relationship between applicant features and credit risk.

Key observations:

- The dataset is imbalanced, with more non-risky applicants than risky applicants.
- Several numerical features contain skewness and extreme values.
- Previous default history is an important risk indicator, but it does not directly determine the final loan status.
- Risk patterns differ across categorical features such as home ownership, loan intent, and previous default history.
- Interest rate, loan-to-income ratio, and applicant background features are useful for predicting credit risk.

## Data Preprocessing and Feature Engineering

The preprocessing steps included:

- Creating income category feature
- Applying log transformation to skewed money-related features
- Splitting data using stratified train-test split
- Encoding categorical features using OneHotEncoder
- Scaling numerical features using StandardScaler

## Models Used

The following classification models were trained and compared:

- Logistic Regression
- Decision Tree
- Tuned Decision Tree
- Random Forest
- Tuned Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- XGBoost

## Model Performance

XGBoost achieved the best overall performance among the tested models.

| Model | Test Accuracy | Class 1 Precision | Class 1 Recall | Class 1 F1-score |
|---|---:|---:|---:|---:|
| XGBoost | 92.52% | 0.94 | 0.71 | 0.80 |
| Tuned Random Forest | 91.25% | 0.88 | 0.70 | 0.78 |
| Random Forest | 91.77% | 0.95 | 0.66 | 0.78 |
| Tuned Decision Tree | 87.95% | 0.72 | 0.72 | 0.72 |
| SVM | 86.02% | 0.65 | 0.76 | 0.70 |
| KNN | 87.28% | 0.79 | 0.57 | 0.66 |
| Decision Tree | 88.40% | 0.88 | 0.54 | 0.67 |
| Logistic Regression | 84.80% | 0.73 | 0.48 | 0.58 |

## Final Model Selection

XGBoost was selected as the final model because it achieved the best balance between test accuracy, precision, recall, and F1-score for the risky class.

Final XGBoost performance:

- Test Accuracy: 92.52%
- Class 1 Precision: 0.94
- Class 1 Recall: 0.71
- Class 1 F1-score: 0.80

Although some models achieved higher recall for class 1, XGBoost provided the best overall balance between identifying risky applicants and avoiding excessive false positives.

## Business Interpretation

In credit risk prediction, class `1` represents risky applicants. A false negative means a risky applicant is incorrectly predicted as non-risky, which can lead to financial loss for lenders.

The final XGBoost model correctly identified 71% of risky applicants and achieved a class 1 F1-score of 0.80. This model can support financial institutions by helping identify high-risk applicants before loan approval.

## Deployment

The final XGBoost model and preprocessing objects were saved using pickle for deployment. A Streamlit app was created to allow users to enter applicant details and receive a credit risk prediction.

The deployed app uses the same preprocessing steps used during model training:

- Income category creation
- Log transformation
- One-hot encoding
- Feature scaling
- Column order matching
- XGBoost prediction

## How to Run the Project Locally

Clone the repository:

```bash
git clone https://github.com/Pranav175-bit/Credit-Risk-Prediction.git
```

Navigate to the project folder:

```bash
cd Credit-Risk-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

If Streamlit is not recognized on Windows, use:

```bash
python -m streamlit run app.py
```

## Project Files

| File | Description |
|---|---|
| `Credit_Risk_Prediction.ipynb` | Complete machine learning notebook |
| `credit_risk.csv` | Dataset used for the project |
| `credit_risk_model.pkl` | Saved model and preprocessing objects |
| `app.py` | Streamlit application |
| `requirements.txt` | Required Python packages |
| `README.md` | Project documentation |

## Conclusion

This project successfully built and evaluated multiple machine learning models for credit risk prediction. XGBoost achieved the best overall performance and was selected as the final model.

The project demonstrates a complete machine learning workflow from data cleaning and EDA to model comparison, final model selection, prediction on new applicant data, and deployment preparation.