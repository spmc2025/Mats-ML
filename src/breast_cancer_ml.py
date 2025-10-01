# File: src/breast_cancer_ml_web.py

import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# =========================
# Streamlit file upload
# =========================
st.title("Breast Cancer Prediction ML")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    # Prepare features and target
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

    # Prepare output folder
    OUTPUT_DIR = "out_files"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save predictions
    df_out = df.loc[y_test.index].copy()
    df_out['predicted'] = y_pred_label
    df_out.drop(columns=['target'], inplace=True)

    df_out.to_csv(os.path.join(OUTPUT_DIR, "All_Patient_Predictions.csv"), index=False)
    df_out[df_out['predicted'] == 'cancer'].to_csv(os.path.join(OUTPUT_DIR, "Cancer_Patient_Predictions.csv"), index=False)
    df_out[df_out['predicted'] == 'no cancer'].to_csv(os.path.join(OUTPUT_DIR, "Non_Cancer_Predictions.csv"), index=False)

    # Save model
    model.save_model(os.path.join(OUTPUT_DIR, "Breast_Cancer_Model.json"))

    # =========================
    # Plots
    # =========================
    plt.figure()
    xgb.plot_importance(model, max_num_features=10, height=0.5, importance_type='weight')
    plt.title("Top 10 Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Feature_Importance.png"))
    plt.close()

    plt.figure()
    pred_counts = pd.Series(y_pred_label).value_counts()
    sns.barplot(x=pred_counts.index, y=pred_counts.values, palette="Set2")
    plt.title("Predicted Cancer vs Non-Cancer")
    plt.xlabel("Prediction")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Predicted_Barchart.png"))
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['cancer','no cancer'], yticklabels=['cancer','no cancer'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Confusion_Matrix.png"))
    plt.close()

    # =========================
    # Evaluation
    # =========================
    accuracy = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, target_names=["cancer","no cancer"])

    st.subheader("Accuracy")
    st.write(f"{accuracy:.4f}")

    st.subheader("Classification Report")
    st.text(cls_report)

    st.success(f"All outputs saved in folder: {OUTPUT_DIR}")
else:
    st.info("Please upload a CSV file to run the model.")
