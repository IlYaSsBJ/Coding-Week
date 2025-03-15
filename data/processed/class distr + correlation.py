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