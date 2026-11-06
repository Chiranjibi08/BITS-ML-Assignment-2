# Breast Cancer Classification Using Five Machine Learning Models

## 1. Project Overview

This project implements and compares five machine learning classification algorithms using the Breast Cancer Wisconsin dataset.

The objective is to classify breast cancer cases into:
- Malignant
- Benign

A Streamlit dashboard is developed to demonstrate model predictions and compare evaluation metrics.

## 2. Dataset Information

- Dataset: Breast Cancer Wisconsin Dataset
- Total observations: 569
- Total features: 30
- Target variable: Diagnosis
- Missing values: 0

## 3. Machine Learning Algorithms

The following five algorithms are implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 4. Model Evaluation

The models are evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## 5. Model Comparison Results

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| SVM | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

## 6. Best Model

Logistic Regression and Support Vector Machine achieved the highest accuracy of 98.25% and MCC of 0.9623. However, Logistic Regression achieved the highest AUC of 0.9954. Therefore, Logistic Regression is selected as the overall best-performing model.

## 7. Project Structure

```text
BITS-ML-Assignment-2/
├── ML_Assignment_2.ipynb
├── app.py
├── requirements.txt
├── test_data.csv
├── model_comparison.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## 8. Running the Application

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 9. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## 10. Model performance Observations

| Model | Observation |
|---|---|
| Logistic Regression | Achieved 98.25% accuracy and 0.9954 AUC. Strong overall performance. |
| Decision Tree | Achieved 91.23% accuracy. Lowest accuracy among the five models. |
| KNN | Achieved 95.61% accuracy with 0.9722 recall. Good classification performance. |
| Gaussian Naive Bayes | Achieved 93.86% accuracy and 0.9878 AUC. |
| Support Vector Machine (SVM) | Achieved 98.25% accuracy and 0.9950 AUC. Performs almost identically to Logistic Regression. |
| Overall Winner | Logistic Regression – Achieved the highest AUC (0.9954), with 98.25% accuracy and 0.9623 MCC. Selected as the best-performing model overall. |

### Overall Observation

Logistic Regression and Support Vector Machine achieved the highest accuracy (98.25%). Logistic Regression achieved the highest AUC (0.9954), making it the best-performing model overall based on the reported metrics.


## 11. Repository

GitHub: https://github.com/Chiranjibi08/BITS-ML-Assignment-2


---

*Academic Machine Learning Assignment – BITS Pilani*
