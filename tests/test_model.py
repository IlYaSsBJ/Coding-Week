import joblib
import numpy as np

MODEL_PATH = "model/obesity_model.pkl"

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