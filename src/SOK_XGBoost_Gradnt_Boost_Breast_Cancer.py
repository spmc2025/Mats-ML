"""
Breast Cancer Prediction using XGBoost
Now fully CLI-compatible for Azure DevOps pipelines
"""

import argparse
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Command-line arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Breast Cancer Prediction ML Script")
parser.add_argument("--input-csv", required=True, help="Path to input CSV data")
parser.add_argument("--output-csv", required=True, help="Path to save prediction CSV")
parser.add_argument("--output-plots", required=True, help="Folder to save plots")
parser.add_argument("--output-model", required=True, help="Path to save trained XGBoost model")
args = parser.parse_args()

# Ensure output directories exist
os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
os.makedirs(args.output_plots, exist_ok=True)
os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv(args.input_csv)

X = df.drop(['target', 'Patient_ID', 'Patient_Name'], axis=1)
y = df['target'].map({'cancer': 0, 'no cancer': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# ---------------------------
# XGBoost training
# ---------------------------
params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'eta': 0.1,
    'eval_metric': 'logloss',
    'seed': 42
}

model = xgb.train(params, dtrain, num_boost_round=50)

# ---------------------------
# Predictions
# ---------------------------
y_pred_prob = model.predict(dtest)
y_pred = [1 if prob > 0.7 else 0 for prob in y_pred_prob]
y_pred_label = ['cancer' if pred == 0 else 'no cancer' for pred in y_pred]

df_predictions = df.loc[y_test.index].copy()
df_predictions['predicted'] = y_pred_label
df_predictions_no_target = df_predictions.drop(columns=['target'])
df_predictions_no_target.to_csv(args.output_csv, index=False)

# ---------------------------
# Save model
# ---------------------------
model.save_model(args.output_model)

# ---------------------------
# Plots
# ---------------------------
# Feature importance
plt.figure()
xgb.plot_importance(model, max_num_features=10, height=0.5, importance_type='weight')
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig(os.path.join(args.output_plots, "Breast_Cancer_Feature_Importance.png"))
plt.close()

# Bar chart of predictions
plt.figure(figsize=(6, 4))
predicted_counts = pd.Series(y_pred_label).value_counts()
bars = sns.barplot(x=predicted_counts.index, y=predicted_counts.values, palette="Set2")
for bar in bars.patches:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, int(height), ha='center', va='bottom')
plt.title("Predicted Cancer vs Non-Cancer Cases")
plt.xlabel("Prediction")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(os.path.join(args.output_plots, "Breast_Cancer_Barchart.png"))
plt.close()

# Confusion matrix
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["cancer", "no cancer"],
            yticklabels=["cancer", "no cancer"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(args.output_plots, "Breast_Cancer_CM.png"))
plt.close()

# ---------------------------
# Evaluation
# ---------------------------
accuracy = accuracy_score(y_test, y_pred)
cls_report = classification_report(y_test, y_pred, target_names=["cancer", "no cancer"])
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", cls_report)
