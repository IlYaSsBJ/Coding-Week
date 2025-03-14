import joblib
import numpy as np
import os

# Get the absolute path of the current script (inside views/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Move up one level to reach the project root
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "obesity_model.pkl")

def test_model_loading():
    """Test du chargement du modèle"""
    model = joblib.load(MODEL_PATH)
    assert model is not None, "⚠ Échec du chargement du modèle"

def test_model_prediction():
    """Test d'une prédiction avec des données simulées"""
    model = joblib.load(MODEL_PATH)
    sample_input = np.array([[25, 1.75, 70, 2, 3, 2, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]])
    prediction = model.predict(sample_input)
    assert prediction in [0, 1, 2, 3, 4, 5, 6], f"⚠ Prédiction invalide : {prediction}"