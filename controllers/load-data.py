import joblib
import pandas as pd
from pathlib import Path

# Détecter automatiquement le dossier du projet
BASE_DIR = Path(__file__).resolve().parent.parent  # Ajuste ici si besoin

#  Définir les chemins dynamiquement
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PATH = BASE_DIR / "model" / "obesity_model.pkl"

def load_data():
    """Charge le dataset et renvoie un DataFrame."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"❌ Erreur : Le dataset '{DATASET_PATH}' est introuvable.")
    return pd.read_csv(DATASET_PATH)

def load_model():
    """Charge le modèle ML."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Erreur : Le modèle '{MODEL_PATH}' est introuvable.")
    return joblib.load(MODEL_PATH)

print("✅ Chargement des fichiers réussi.")
