import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import gc
import psutil
import os
import pickle
import shap


# === data loading ===
# Get the absolute path of the current script (inside views/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Move up one level to reach the project root
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Construct paths relative to the project root
DATA_PATH = os.path.join(BASE_DIR, "data","processed" , "dataset.csv")
df= pd.read_csv(DATA_PATH)

# === memory reducing ===
def optimize_dataframe(df):
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    return df

df = optimize_dataframe(df)
categorical_columns = ["Gender", "family_history_with_overweight", "NObeyesdad","FAVC","SMOKE","CAEC","SCC","CALC","MTRANS",]

# ==== spliting data ====
# Apply Label Encoding

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders if you need to inverse transform later

# Split into features (X) and target (y)
X = df.drop("NObeyesdad", axis=1)  # Features
y = df["NObeyesdad"]  # Target (Obesity Level)


# Split into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)


# ==== FLAGS FOR SAMPLING METHODS ====
USE_SMOTE = True           # Enable/Disable Oversampling
USE_UNDERSAMPLING = True  # Enable/Disable Undersampling
USE_CLASS_WEIGHTS = False  # Set this to True to use class weighting


# Apply Oversampling (SMOTE)
if USE_SMOTE:
    smote = SMOTE(sampling_strategy="auto", random_state=42)  # 60% oversampling
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("Applied SMOTE Oversampling. New Training Set Size:", X_train.shape)

# Apply Undersampling (RandomUnderSampler)
if USE_UNDERSAMPLING:
    undersample = RandomUnderSampler(sampling_strategy="auto", random_state=42)  # 80% of majority class
    X_train, y_train = undersample.fit_resample(X_train, y_train)
    print("Applied Random UnderSampling. New Training Set Size:", X_train.shape)

# Compute Class Weights (If Selected)
class_weight_dict = None
if USE_CLASS_WEIGHTS:
    class_weights = compute_class_weight(class_weight="balanced",
                                         classes=np.unique(y_train), y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
    print("Applied Class Weights:", class_weight_dict)


# ===== training =====

# Initialize Model with Selected Class Weights
model = RandomForestClassifier(n_estimators=100, random_state=42,
                               class_weight=class_weight_dict if USE_CLASS_WEIGHTS else "balanced")
# Train Model
model.fit(X_train, y_train)


# Save the trained model to a file
with open("obesity_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")


# Train SHAP explainer after training your model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)  # Ensure X_test matches training format


# Save SHAP explainer to a file
with open("shap_explainer.pkl", "wb") as file:
      pickle.dump(explainer, file)

print("SHAP explainer saved successfully as shap_explainer.pkl!")


# === Memmory Optimization ===

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Convertir en Mo

memory_used = get_memory_usage()
print(f" Memory use after execution : {memory_used:.2f} Mo")
variables_a_supprimer = [var for var in globals().keys() if var not in ["get_memory_usage", "gc", "psutil", "pickle", "shap", "__name__", "__file__", "__builtins__"]]


for var in variables_a_supprimer:
    del globals()[var]

gc.collect()
print(" Memory freed !")
