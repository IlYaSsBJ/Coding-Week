# missing values :

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(CURRENT_DIR,"raw" , "ObesityDataSet.csv")

# Charger le dataset

df = pd.read_csv(file_path)

# Afficher les premières lignes pour vérifier le chargement des données
print("Aperçu des données :")
print(df.head())

# Vérifier les valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

# Vérifier si certaines valeurs sont codées différemment comme manquantes
print("\nValeurs potentiellement manquantes sous d'autres formes ('?', 'None', '') :")
print(df.isin(["?", "None", ""]).sum())

# Obtenir un résumé des données
print("\nRésumé des colonnes et valeurs non nulles :")
print(df.info())

# results of code missing values = NO MISSING VALUES

# show outliers : 

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle

# Charger les données
file_path = os.path.join(CURRENT_DIR, "ObesityDataSet.csv")

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

import pandas as pd

# Load your dataset
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(CURRENT_DIR,"raw" , "ObesityDataSet.csv")
df = pd.read_csv(file_path)

# Define outlier removal conditions
lower_weight = 40  # Minimum valid weight
upper_weight = 105  # Maximum valid weight
lower_age = 10  # Minimum valid age
upper_age = 26  # Maximum valid age

# Apply both filters (Weight & Age)

df_filtered = df[
    (df["Weight"] >= lower_weight) & (df["Weight"] <= upper_weight) &
    (df["Age"] >= lower_age) & (df["Age"] <= upper_age)
]


# Save the cleaned dataset in the 'data' folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
df_filtered.to_csv(CURRENT_DIR,"raw" , "ObesityDataSet.csv", index=False)

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


#Class distribution 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(CURRENT_DIR, "dataset.csv")
df = pd.read_csv(file_path)

# Check class distribution and display percentages
class_counts = df["NObeyesdad"].value_counts(normalize=True) * 100  # Get percentages
print("Class Distribution:\n", class_counts)

# Plot Class Distribution with percentages displayed
plt.figure(figsize=(10, 5))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
for i, v in enumerate(class_counts.values):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')  # Adding percentages on top of bars
plt.xticks(rotation=45)
plt.xlabel("Obesity Level")
plt.ylabel("Percentage of Samples")
plt.title("Class Distribution of Obesity Levels")
plt.tight_layout()  
plt.show()

# Box plot of all numerical columns
plt.figure(figsize=(10, 5))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Boxplot of Numerical Features")
plt.tight_layout()
plt.show()


########### Boxplot for Age vs. Obesity Level ##########
plt.figure(figsize=(12, 6))
sns.boxplot(x=df["NObeyesdad"], y=df["Age"], palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Age Distribution Across Obesity Levels")
plt.show()

########### Boxplot for Weight vs. Obesity Level ##########
sns.boxplot(x=df["NObeyesdad"], y=df["Weight"], palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Weight Distribution Across Obesity Levels")
plt.show()


########### Boxplot for Weight vs. Obesity Level ##########
plt.figure(figsize=(12, 6))
sns.boxplot(x=df["NObeyesdad"], y=df["Weight"], palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Weight Distribution Across Obesity Levels")
plt.show()

########### Boxplot for Physical Activity Frequency (FAF) vs. Obesity Level ##########
plt.figure(figsize=(12, 6))
sns.boxplot(x=df["NObeyesdad"], y=df["FAF"], palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Physical Activity (FAF) Across Obesity Levels")
plt.show()


########### Print Class Distribution ##########
class_counts = df["NObeyesdad"].value_counts(normalize=True) * 100
print("Class Distribution:\n", class_counts)


#understanding correlation and computing correlation matrix

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(CURRENT_DIR, "dataset.csv")
df = pd.read_csv(file_path)

# Convert all categorical columns to numeric
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

# Convert categorical target variable to numeric for correlation analysis
df["Obesity_Level_Num"] = df["NObeyesdad"].astype("category").cat.codes

# Compute correlation matrix
corr_matrix = df.corr()

# Print class distribution as percentages
print(df['NObeyesdad'].value_counts(normalize=True) * 100)

# Convert categorical features to numeric (using one-hot encoding)
df_encoded = pd.get_dummies(df, drop_first=True)

# Compute correlation matrix of the encoded data
correlation_matrix = df_encoded.corr()

# Plot heatmap of the correlation matrix after encoding
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix ")
plt.show()

print(f"✅ Label Encoder sauvegardé dans : {encoder_path}")

