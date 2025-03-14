# 📌 Utiliser une image Python légère
FROM python:3.9-slim

# 📌 Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 📌 Copier les fichiers nécessaires pour l'application
COPY requirements.txt .
COPY view/ view/
COPY model/ model/
COPY controller/ controller/
COPY data/processed/ data/processed/

# 📌 Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 📌 Exposer le port utilisé par Streamlit
EXPOSE 8501

# 📌 Lancer l'application Streamlit
CMD ["streamlit", "run", "view/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
