# missing values :

import pandas as pd

# Charger le dataset
file_path = r"C:\Users\hajar\Downloads\estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition\ObesityDataSet_raw_and_data_sinthetic.csv"
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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
file_path = r"C:\Users\hajar\Downloads\estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition\ObesityDataSet_raw_and_data_sinthetic.csv"
df = pd.read_csv(file_path)

# Sélectionner uniquement les colonnes numériques
numerical_cols = df.select_dtypes(include=['number']).columns
df_numerical = df[numerical_cols]

# Créer le boxplot
plt.figure(figsize=(14, 6))  # Taille du graphe
sns.boxplot(data=df_numerical)

# Personnalisation du graphe
plt.title("Boxplots for Numerical Features", fontsize=14)
plt.xticks(rotation=45)  # Rotation des noms des variables pour une meilleure lisibilité
plt.grid(axis="y", linestyle="--", alpha=0.7)  # Grille légère sur l'axe Y

# Afficher le graphe
plt.show()

# outliers_removal:

import pandas as pd

# Load your dataset
df = pd.read_csv(r"C:\Users\hajar\Downloads\estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition\ObesityDataSet_raw_and_data_sinthetic.csv")  # Ensure you provide the correct file path

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
df_filtered.to_csv(r"C:\Users\hajar\Documents\GitHub\Coding-Week\data\age_and_weight_cleaned.csv", index=False)

print("✅ Cleaned data saved in 'data' folder as 'age_and_weight_cleaned.csv'.")

le = LabelEncoder()
df["NObeyesdad"] = le.fit_transform(df["NObeyesdad"])  # Encoding obesity levels

import pickle

# Save LabelEncoder
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(le, file)