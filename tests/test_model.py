import pickle
import pandas as pd
import numpy as np
import joblib
import os


# Get the absolute path of the current script (inside views/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Move up one level to reach the project root
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "obesity_model.pkl")


# Ensure consistent feature selection
EXPECTED_FEATURES = [
    "Gender", "Age", "Height", "Weight", "family_history_with_overweight",
    "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"
]

# Load the trained model

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"⚠ Model file not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

def test_model_loading():
    """Test if the model loads correctly"""
    assert model is not None, "⚠ Model failed to load!"

def test_model_prediction():
    """Test model prediction with a sample input"""
    sample_input = pd.DataFrame([{
        "Gender": 1, "Age": 30, "Height": 170, "Weight": 75,
        "family_history_with_overweight": 1, "FAVC": 1, "FCVC": 2.5, "NCP": 3,
        "CAEC": 2, "SMOKE": 0, "CH2O": 2, "SCC": 1, "FAF": 1.5, "TUE": 1.5, "CALC": 1, "MTRANS": 2
    }])

    # Ensure only expected features are used
    sample_input = sample_input[EXPECTED_FEATURES]

    # Check that the input matches model expectations
    assert sample_input.shape[1] == len(EXPECTED_FEATURES), f"⚠ Expected {len(EXPECTED_FEATURES)} features, got {sample_input.shape[1]}"

    # Run prediction
    prediction = model.predict(sample_input)
    
    # Check that output is a valid class
    valid_classes = [0, 1, 2, 3, 4, 5, 6]  # Adjust based on your label encoding
    assert prediction[0] in valid_classes, f"⚠ Invalid prediction output: {prediction[0]}"

