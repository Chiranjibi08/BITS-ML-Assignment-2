
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Breast Cancer ML Dashboard",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Breast Cancer Classification Dashboard")
st.write(
    "Compare five machine learning models "
    "using the Breast Cancer Wisconsin dataset."
)

# Model paths
model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

# Load saved models
@st.cache_resource
def load_models():
    loaded_models = {}

    for name, path in model_files.items():
        loaded_models[name] = joblib.load(path)

    return loaded_models

# Check model files
if not all(os.path.exists(path) for path in model_files.values()):
    st.error("Model files are missing. Please check the model folder.")
    st.stop()

models = load_models()

# Dataset upload
st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test_data.csv file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.write("Dataset shape:", data.shape)

    # Target column
    target_column = "diagnosis"

    if target_column not in data.columns:
        st.error("Target column 'diagnosis' is missing.")
        st.stop()

    # Get required feature names
    feature_names = list(models["Logistic Regression"].feature_names_in_)

    missing_features = [
        col for col in feature_names if col not in data.columns
    ]

    if missing_features:
        st.error(f"Missing features: {missing_features}")
        st.stop()

    X = data[feature_names]
    y_true = data[target_column]

    if not set(y_true.unique()).issubset({0, 1}):
        st.error("Diagnosis must contain only 0 and 1.")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Calculate metrics for all models
    all_results = {}

    for name, model in models.items():

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        all_results[name] = {
            "Accuracy": accuracy_score(y_true, predictions),
            "AUC": roc_auc_score(y_true, probabilities),
            "Precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_true, predictions, zero_division=0
            ),
            "MCC": matthews_corrcoef(y_true, predictions)
        }

    # Model selection
    st.header("2. Select Machine Learning Model")

    selected_model = st.selectbox(
        "Choose a model",
        list(models.keys())
    )

    model = models[selected_model]

    y_pred = model.predict(X)
    y_probability = model.predict_proba(X)[:, 1]

    # Evaluation metrics
    st.header("3. Model Evaluation Metrics")

    metrics = all_results[selected_model]

    columns = st.columns(3)

    for col, (metric, value) in zip(columns + st.columns(3), metrics.items()):
        col.metric(metric, f"{value:.4f}")

    # Confusion matrix
    st.header("4. Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Malignant", "Actual Benign"],
        columns=["Predicted Malignant", "Predicted Benign"]
    )

    st.dataframe(cm_df)

    # Classification report
    st.header("5. Classification Report")

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Malignant", "Benign"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(pd.DataFrame(report).transpose())

    # Compare all five models
    st.header("6. Comparison of All Five Models")

    comparison = pd.DataFrame(all_results).transpose()

    st.dataframe(
        comparison.style.format("{:.4f}"),
        use_container_width=True
    )

    st.success("Evaluation completed successfully!")

else:
    st.info("Please upload your test_data.csv file to begin.")
