import os
import argparse
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# CLI arguments
parser = argparse.ArgumentParser(description="Breast Cancer Prediction using XGBoost")
parser.add_argument("--input-csv", required=True, help="Path to input CSV file")
parser.add_argument("--output-csv", required=True, help="Path to output folder for prediction CSVs")
parser.add_argument("--output-plots", required=True, help="Path to output folder for plots")
parser.add_argument("--output-model", required=True, help="Path to save trained model")
args = parser.parse_args()

# Ensure output directories exist
os.makedirs(args.output_csv, exist_ok=True)
os.makedirs(args.output_plots, exist_ok=True)
os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

# Load dataset
df = pd.read_csv(args.input_csv)

# Prepare features and target
X = df.drop(['target', 'Patient_ID', 'Patient_Name'], axis=1)
y = df['target'].map({'cancer': 0, 'no cancer': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrices
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train model
params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'eta': 0.1,
    'eval_metric': 'logloss',
    'seed': 42
}
model = xgb.train(params, dtrain, num_boost_round=50)

# Predictions
y_pred_prob = model.predict(dtest)
y_pred = [1 if prob > 0.7 else 0 for prob in y_pred_prob]
y_pred_label = ['cancer' if pred == 0 else 'no cancer' for pred in y_pred]

# Prepare output dataframe
df_predictions = df.loc[y_test.index].copy()
df_predictions['predicted'] = y_pred_label
df_predictions_no_target = df_predictions.drop(columns=['target'])

# Save prediction CSVs
df_predictions_no_target.to_csv(os.path.join(args.output_csv, "All_Patient_Predictions_OUT.csv"), index=False)
df_predictions_no_target[df_predictions_no_target['predicted'] == 'cancer'].to_csv(
    os.path.join(args.output_csv, "Cancer_Patient_Predictions_OUT.csv"), index=False)
df_predictions_no_target[df_predictions_no_target['predicted'] == 'no cancer'].to_csv(
    os.path.join(args.output_csv, "Non_Cancer_Patient_Predictions_OUT.csv"), index=False)

# Save model
model.save_model(args.output_model)

# Plots
xgb.plot_importance(model, max_num_features=10, height=0.5, importance_type='weight')
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig(os.path.join(args.output_plots, "Breast_Cancer_Feature_Importance_OUT.png"))
plt.close()

predicted_counts = pd.Series(y_pred_label).value_counts()
plt.figure(figsize=(6, 4))
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

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["cancer", "no cancer"],
            yticklabels=["cancer", "no cancer"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(args.output_plots, "Breast_Cancer_CM_OUT.png"))
plt.close()

# Print metrics
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["cancer", "no cancer"]))
