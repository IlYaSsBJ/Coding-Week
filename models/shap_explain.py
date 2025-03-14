import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gc
import psutil
import pandas as pd
import os


# === data loading ===
# Get the absolute path of the current script (inside views/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Move up one level to reach the project root
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Construct paths relative to the project root
DATA_PATH = os.path.join(BASE_DIR, "data","processed" , "dataset.csv")
df= pd.read_csv(DATA_PATH)

MODEL_PATH = "model/obesity_model.pkl"
model = joblib.load(MODEL_PATH)

# === data preparing ====
X = df.drop("NObeyesdad", axis=1)
X_train, X_test, _, _ = train_test_split(X, df["NObeyesdad"], test_size=0.2, random_state=42, stratify=df["NObeyesdad"])

# === Shap Explainer ====
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# === Memmory Optimization ===
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Convertir en Mo

memory_used = get_memory_usage()
print(f" Memory use after execution : {memory_used:.2f} Mo")
variables_a_supprimer = [var for var in globals().keys() if var not in ["get_memory_usage", "gc", "psutil", "__name__", "__file__", "__builtins__"]]

for var in variables_a_supprimer:
    del globals()[var]

gc.collect()
print(" Memory freed !")
