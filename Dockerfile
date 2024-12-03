# Utiliser une image Python comme base
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier tous les fichiers dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Définir le point d'entrée pour exécuter les deux scripts séquentiellement
CMD ["bash", "-c", "python train_classifier.py && python predict_classification.py"]

