import joblib
import shap
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === DATA LOADING ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset.csv")

# ✅ Ensure the correct model path
MODEL_PATH = os.path.join(BASE_DIR, "models", "obesity_model.pkl")

# ✅ Check if model file exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print(f"🔹 Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# === DATA PREPARATION ===
df = pd.read_csv(DATA_PATH)

# ✅ List of categorical columns
categorical_columns = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE",
                       "CAEC", "SCC", "CALC", "MTRANS"]

# ✅ Apply Label Encoding to Categorical Columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for inverse transform if needed

# ✅ Split data
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === SHAP EXPLAINER ===
explainer = shap.TreeExplainer(model)

# ✅ X_test is now correctly encoded
shap_values = explainer.shap_values(X_test)

# ✅ SHAP Summary Plot
shap.summary_plot(shap_values, X_test)

print("✅ SHAP analysis completed successfully!")
