import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Génération des données
def generate_data(n_samples=10000, n_features=20, n_informative=10, n_classes=3,
                  n_clusters_per_class=2, class_sep=1.0, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        random_state=random_state
    )
    data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    data["target"] = y
    return data


# Division des données
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
    print(f"Taille des ensembles :\n- Entraînement : {len(X_train)}\n- Test : {len(X_test)}\n- Validation : {len(X_val)}")
    return X_train, X_test, X_val, y_train, y_test, y_val


# Entraîner et évaluer un modèle
def train_and_evaluate(X_train, y_train, X_test, y_test, model_choice):
    models = {
        "random_forest": RandomForestClassifier(random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "svm": SVC(random_state=42)
    }

    if model_choice not in models:
        raise ValueError("Choix invalide. Options disponibles : 'random_forest', 'logistic_regression', 'svm'.")

    model = models[model_choice]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    plot_confusion_matrix(y_test, y_pred, model_choice)
    return model


# Matrice de confusion
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Prédictions")
    plt.ylabel("Valeurs actuelles")
    plt.title(f"Matrice de Confusion - {model_name}")
    plt.show()


# Exportation du modèle et des données de validation
def export_model_and_validation_data(model, X_val, y_val, model_filename="trained_model.pkl", data_filename="validation_data.csv"):
    import joblib
    joblib.dump(model, model_filename)
    print(f"Modèle sauvegardé dans le fichier : {model_filename}")

    validation_data = pd.DataFrame(X_val, columns=[f"feature_{i}" for i in range(X_val.shape[1])])
    validation_data["target"] = y_val
    validation_data.to_csv(data_filename, index=False)
    print(f"Données de validation sauvegardées dans le fichier : {data_filename}")


# Fonction principale
def main():
    print("Génération des données...")
    data = generate_data()
    data.to_csv("synthetic_data.csv", index=False)
    print(f"Données générées et sauvegardées dans synthetic_data.csv")

    X = data.drop(columns=["target"]).values
    y = data["target"].values

    print("Division des données...")
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X, y)

    # Fixe le choix par défaut à 1 (un seul modèle)
    choice = "1"
    print(f"Choix par défaut sélectionné : {choice}")

    if choice == "1":
        model_choice = "logistic_regression"  # Modèle par défaut
        print(f"Modèle sélectionné : {model_choice}")
        model = train_and_evaluate(X_train, y_train, X_test, y_test, model_choice)
        export_model_and_validation_data(model, X_val, y_val)
    else:
        print("Choix non supporté.")


if __name__ == "__main__":
    main()
