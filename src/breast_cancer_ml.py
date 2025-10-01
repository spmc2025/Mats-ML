# File: breast_cancer_web_app.py
import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import BytesIO

# =========================
# Output folder
# =========================
OUTPUT_DIR = "out_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Streamlit App
# =========================
st.title("Breast Cancer Prediction ML Web App")
st.write("Upload a CSV file with patient data to predict cancer vs non-cancer.")

# =========================
# File upload
# =========================
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # =========================
    # Prepare features and target
    # =========================
    X = df.drop(['target', 'Patient_ID', 'Patient_Name'], axis=1)
    y = df['target'].map({'cancer': 0, 'no cancer': 1})

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # =========================
    # XGBoost training
    # =========================
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'eta': 0.1,
        'eval_metric': 'logloss',
        'seed': 42
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    model = xgb.train(params, dtrain, num_boost_round=50)

    # =========================
    # Predictions
    # =========================
    y_pred_prob = model.predict(dtest)
    y_pred = [1 if p > 0.7 else 0 for p in y_pred_prob]
    y_pred_label = ['cancer' if pred == 0 else 'no cancer' for pred in y_pred]

    # Prepare output DataFrame
    df_out = df.loc[y_test.index].copy()
    df_out['predicted'] = y_pred_label
    df_out.drop(columns=['target'], inplace=True)

    st.subheader("Prediction Counts")
    st.write(pd.Series(y_pred_label).value_counts())

    # =========================
    # Download buttons
    # =========================
    def get_csv_download(df):
        return df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download All Predictions",
        data=get_csv_download(df_out),
        file_name="All_Patient_Predictions.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Cancer Predictions",
        data=get_csv_download(df_out[df_out['predicted']=='cancer']),
        file_name="Cancer_Patient_Predictions.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Non-Cancer Predictions",
        data=get_csv_download(df_out[df_out['predicted']=='no cancer']),
        file_name="Non_Cancer_Predictions.csv",
        mime="text/csv"
    )

    # =========================
    # Evaluation
    # =========================
    accuracy = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, target_names=["cancer","no cancer"])

    st.subheader(f"Accuracy: {accuracy:.4f}")
    st.text(cls_report)

    # =========================
    # Plots
    # =========================
    # Feature Importance
    fig1, ax1 = plt.subplots()
    xgb.plot_importance(model, max_num_features=10, height=0.5, importance_type='weight', ax=ax1)
    plt.title("Top 10 Feature Importances")
    st.pyplot(fig1)

    # Prediction counts bar chart
    fig2, ax2 = plt.subplots()
    pred_counts = pd.Series(y_pred_label).value_counts()
    sns.barplot(x=pred_counts.index, y=pred_counts.values, palette="Set2", ax=ax2)
    plt.title("Predicted Cancer vs Non-Cancer")
    plt.xlabel("Prediction")
    plt.ylabel("Number of Patients")
    st.pyplot(fig2)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['cancer','no cancer'],
                yticklabels=['cancer','no cancer'], ax=ax3)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig3)

    st.success("Prediction and evaluation completed successfully!")
