import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os

def ensure_directory_exists(directory):
    """
    Vérifie si un répertoire existe, et le crée s'il n'existe pas.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_model_and_data(model_filename="trained_model.pkl", data_filename="validation_data.csv"):
    # Charger le modèle
    print(f"Chargement du modèle depuis {model_filename}...")
    model = joblib.load(model_filename)
    print("Modèle chargé avec succès.")

    # Charger les données de validation
    print(f"Chargement des données de validation depuis {data_filename}...")
    validation_data = pd.read_csv(data_filename)
    X_val = validation_data.drop(columns=["target"])
    y_val = validation_data["target"]
    print("Données de validation chargées avec succès.")

    return model, X_val, y_val

def predict_and_evaluate(model, X_val, y_val):
    # Détecter automatiquement le nom du modèle
    model_name = type(model).__name__

    # Faire les prédictions
    print("Réalisation des prédictions...")
    y_pred = model.predict(X_val)

    # Rapport de classification
    print("\nRapport de classification :")
    print(classification_report(y_val, y_pred))

    # Matrice de confusion
    plot_confusion_matrix(y_val, y_pred, model_name)

def plot_confusion_matrix(y_test, y_pred, model_name):
    """
    Affiche une matrice de confusion et l'enregistre dans un répertoire 'outputs'.
    """
    # Vérifiez ou créez le répertoire 'outputs'
    output_dir = "./outputs"
    ensure_directory_exists(output_dir)

    # Génération de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Prédictions")
    plt.ylabel("Valeurs actuelles")
    plt.title(f"Matrice de Confusion - {model_name}")

    # Sauvegarde de la matrice
    output_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(output_path)
    plt.close()  # Fermer la figure pour libérer des ressources
    print(f"Matrice de confusion enregistrée sous le nom : {output_path}")

if __name__ == "__main__":
    # Charger le modèle et les données
    model, X_val, y_val = load_model_and_data()

    # Effectuer les prédictions et afficher la matrice de confusion
    predict_and_evaluate(model, X_val, y_val)

