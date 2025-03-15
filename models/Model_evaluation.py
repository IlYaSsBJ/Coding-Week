import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import gc
import psutil

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split

# === DATA & MODEL LOADING ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Construct paths
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "obesity_model.pkl")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print(f"Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# === DATA PREPARATION ===
categorical_columns = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE",
                       "CAEC", "SCC", "CALC", "MTRANS", "NObeyesdad"]

# Apply Label Encoding
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for inverse transform if needed

# Split features & target variable
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === PREDICTION ===
y_pred = model.predict(X_test)

# === MODEL EVALUATION ===
accuracy = accuracy_score(y_test, y_pred)
print(f" Accuracy: {accuracy:.2f}")

print("\n Classification Report:\n", classification_report(y_test, y_pred))

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# === ROC-AUC SCORE ===
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
roc_auc = roc_auc_score(y_test_binarized, model.predict_proba(X_test), multi_class="ovr")
print(f" ROC-AUC Score: {roc_auc:.4f}")

# === MEMORY OPTIMIZATION ===
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

memory_used = get_memory_usage()
print(f" Memory usage after execution: {memory_used:.2f} MB")

# Clean up memory
variables_to_keep = {"get_memory_usage", "gc", "psutil", "__name__", "__file__", "__builtins__"}
for var in list(globals().keys()):
    if var not in variables_to_keep:
        del globals()[var]

gc.collect()
print(" Memory freed!")
