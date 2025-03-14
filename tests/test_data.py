import pandas as pd
import os

# Get the absolute path of the current script (inside views/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Move up one level to reach the project root
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed" , "dataset.csv")

def test_dataset_loading():
    """Test du chargement du dataset"""
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "⚠ Le dataset est vide"

def test_data_types():
    """Test que les colonnes numériques sont bien au bon format"""
    df = pd.read_csv(DATA_PATH)
    numerical_columns = ["Age", "Height", "Weight"]
    
    for col in numerical_columns:
        assert df[col].dtype in ["int64", "float64"], f"⚠ Mauvais type de données pour {col}"