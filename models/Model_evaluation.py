import joblib
from controllers.load-data import load-data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import psutil


# === charging model data ===
df = load-data()
MODEL_PATH = "model/obesity_model.pkl"
model = joblib.load(MODEL_PATH)

X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==== prediction ====
y_pred = model.predict(X_test)

# ==== model evaluation ====
accuracy = accuracy_score(y_test, y_pred)
print(f" Accuracy : {accuracy:.2f}")

print("\n Classification Report:\n", classification_report(y_test, y_pred))

# === confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# ==== ROC-AUC Score ====
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])
roc_auc = roc_auc_score(y_test_binarized, model.predict_proba(X_test), multi_class="ovr")
print(f" ROC-AUC Score: {roc_auc:.4f}")
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
