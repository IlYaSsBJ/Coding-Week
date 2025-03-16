# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "data/processed/dataset.csv"  # Update with your dataset path
df = pd.read_csv(file_path)

# Encode categorical features (if any)
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features (X) and target (y)
X = df.drop(columns=['NObeyesdad'])  # Replace 'NObeyesdad' with your target column
y = df['NObeyesdad']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0)  # Set verbose=0 to suppress output
}

# Evaluate each model
results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Make predictions
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    
    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['NObeyesdad'].classes_, yticklabels=label_encoders['NObeyesdad'].classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Classification Report
    print(f"\nClassification Report - {name}:")
    print(classification_report(y_test, y_pred, target_names=label_encoders['NObeyesdad'].classes_))

# Display results
results_df = pd.DataFrame(results)
print("\nModel Performance Metrics:")
print(results_df)

# Save results to a CSV file (optional)
results_df.to_csv("model_performance_results.csv", index=False)
print("✅ Results saved to 'model_performance_results.csv'")

# Plot ROC curves for each model
plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_pred_proba = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_pred_proba, multi_class='ovr'):.2f}")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# Feature Importance for tree-based models
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        feature_importances.nlargest(10).plot(kind='barh')
        plt.title(f"Feature Importance - {name}")
        plt.show()

# Correlation Heatmap of Features
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Features")
plt.show()

# Distribution of Target Variable
plt.figure(figsize=(8, 6))
sns.countplot(y=y, palette='viridis')
plt.title("Distribution of Target Variable")
plt.xlabel("Count")
plt.ylabel("Target Class")
plt.show()