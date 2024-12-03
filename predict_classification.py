import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Fonction pour afficher la matrice de confusion
def plot_confusion_matrix(y_true, y_pred, title="Matrice de Confusion"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités")
    plt.title(title)
    plt.show()

# Charger le modèle entraîné et les données de validation
def main():
    # Charger le modèle
    model_filename = "trained_model.pkl"
    model = joblib.load(model_filename)
    print(f"Modèle chargé depuis {model_filename}")

    # Charger les données de validation
    data_filename = "validation_data.csv"
    validation_data = pd.read_csv(data_filename)
    print(f"Données de validation chargées depuis {data_filename}")

    # Séparer les features et les labels
    X_val = validation_data.drop(columns=["target"]).values
    y_val = validation_data["target"].values

    # Faire des prédictions
    y_pred = model.predict(X_val)

    # Évaluer les performances
    print("\nRapport de classification :")
    print(classification_report(y_val, y_pred))

    # Afficher la matrice de confusion
    plot_confusion_matrix(y_val, y_pred)

if __name__ == "__main__":
    main()
