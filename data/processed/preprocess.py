import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle

import os

# Get the absolute path of the script's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Move one level up to the root of the project
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

# Correct file path
file_path = os.path.join(PROJECT_DIR, "raw", "ObesityDataSet.csv")

# Debugging: Print the path to verify
print(f"🔍 Chemin utilisé : {file_path}")

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"🚨 Fichier introuvable : {file_path}")

# Charger le dataset
print(f"🔍 Chemin utilisé : {file_path}")
df = pd.read_csv(file_path)

# Aperçu des données
print("📊 Aperçu des données :")
print(df.head())

# Vérification des valeurs manquantes
print("\n🔍 Valeurs manquantes par colonne :")
print(df.isnull().sum())

# Vérifier si certaines valeurs sont codées comme manquantes
print("\n🔍 Valeurs potentiellement manquantes ('?', 'None', '') :")
print(df.isin(["?", "None", ""]).sum())

# Résumé des données
print("\n📈 Résumé des colonnes et valeurs non nulles :")
df.info()

# 🔹 Affichage des outliers avec un boxplot
numerical_cols = df.select_dtypes(include=['number']).columns
df_numerical = df[numerical_cols]

plt.figure(figsize=(14, 6))
sns.boxplot(data=df_numerical)
plt.title("Boxplots des variables numériques", fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# 🔹 Suppression des outliers sur l'âge et le poids
lower_weight, upper_weight = 40, 105
lower_age, upper_age = 10, 26

df_filtered = df[
    (df["Weight"] >= lower_weight) & (df["Weight"] <= upper_weight) &
    (df["Age"] >= lower_age) & (df["Age"] <= upper_age)
]

# 🔹 Sauvegarde des données nettoyées
processed_path = os.path.join(CURRENT_DIR, "data", "processed", "dataset.csv")

# Créer le dossier si inexistant
os.makedirs(os.path.dirname(processed_path), exist_ok=True)

df_filtered.to_csv(processed_path, index=False)
print(f"✅ Données nettoyées sauvegardées dans : {processed_path}")

# 🔹 Encodage de la colonne cible
le = LabelEncoder()
df_filtered["NObeyesdad"] = le.fit_transform(df_filtered["NObeyesdad"])

# 🔹 Sauvegarde de l'encodeur
encoder_path = os.path.join(CURRENT_DIR, "data", "processed", "label_encoder.pkl")

with open(encoder_path, "wb") as file:
    pickle.dump(le, file)

print(f"✅ Label Encoder sauvegardé dans : {encoder_path}")
