# Importation des bibliothèques
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

# 1. Génération des données
def generate_data(n_samples=10000, n_features=20, n_informative=10, n_classes=3,
                  n_clusters_per_class=2, class_sep=1.0, random_state=42):
    """
    Génère des données synthétiques avec des paramètres variables.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        random_state=random_state
    )
    # Conversion en DataFrame
    data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    data["target"] = y
    return data

# 2. Division des données
def split_data(X, y):
    """
    Divise les données en trois ensembles : entraînement, test et validation.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
    print(f"Taille des ensembles :\n- Entraînement : {len(X_train)}\n- Test : {len(X_test)}\n- Validation : {len(X_val)}")
    return X_train, X_test, X_val, y_train, y_test, y_val

# 3. Entraînement et évaluation d'un modèle unique
def train_and_evaluate(X_train, y_train, X_test, y_test, model_choice):
    """
    Entraîne et évalue un modèle choisi.
    """
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

    # Affiche la matrice de confusion
    plot_confusion_matrix(y_test, y_pred, model_choice)

    return model

# 4. Entraînement et évaluation de plusieurs modèles
def evaluate_multiple_models(X_train, y_train, X_test, y_test):
    """
    Entraîne et évalue plusieurs modèles.
    """
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(random_state=42)
    }

    best_model = None
    best_score = 0

    for model_name, model in models.items():
        print(f"\nEntraînement et évaluation du modèle : {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Évaluation des performances
        test_accuracy = model.score(X_test, y_test)
        print(f"Précision : {test_accuracy:.2f}")
        print(classification_report(y_test, y_pred))

        # Affiche la matrice de confusion
        plot_confusion_matrix(y_test, y_pred, model_name)

        if test_accuracy > best_score:
            best_score = test_accuracy
            best_model = model

    print(f"\nMeilleur modèle : {type(best_model).__name__} avec une précision de {best_score:.2f}")
    return best_model

# 5. Affichage de la matrice de confusion
def plot_confusion_matrix(y_test, y_pred, model_name):
    """
    Affiche une matrice de confusion.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Prédictions")
    plt.ylabel("Valeurs actuelles")
    plt.title(f"Matrice de Confusion - {model_name}")
    plt.show()

# 6. Exportation du modèle et des données de validation
def export_model_and_validation_data(model, X_val, y_val, model_filename="model_train.pkl", data_filename="validation_data.csv"):
    """
    Exporte le modèle entraîné et les données de validation.
    """
    try:
        print("Début de l'exportation du modèle...")
        joblib.dump(model, model_filename)
        print(f"Modèle sauvegardé sous le nom : {model_filename}")
    except Exception as e:
        print(f"Erreur lors de l'exportation du modèle : {e}")

    try:
        print("Début de l'exportation des données de validation...")
        validation_data = pd.DataFrame(X_val, columns=[f"feature_{i}" for i in range(X_val.shape[1])])
        validation_data["target"] = y_val
        validation_data.to_csv(data_filename, index=False)
        print(f"Données de validation sauvegardées sous le nom : {data_filename}")
    except Exception as e:
        print(f"Erreur lors de l'exportation des données de validation : {e}")

# 7. Fonction principale
def main():
    # Génération et sauvegarde des données
    data = generate_data()
    data.to_csv("synthetic_data.csv", index=False)
    print(f"Données générées et sauvegardées dans synthetic_data.csv")

    # Chargement des données
    X = data.drop(columns=["target"]).values
    y = data["target"].values

    # Division des données
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X, y)

    # Menu interactif
    print("\nVoulez-vous :")
    print("1 - Tester un seul modèle")
    print("2 - Tester plusieurs modèles")
    choice = input("Entrez votre choix (1 ou 2) : ")

    if choice == "1":
        print("\nChoisissez un modèle :")
        print("- random_forest")
        print("- logistic_regression")
        print("- svm")
        model_choice = input("Entrez le nom du modèle : ")
        model = train_and_evaluate(X_train, y_train, X_test, y_test, model_choice)
    elif choice == "2":
        model = evaluate_multiple_models(X_train, y_train, X_test, y_test)
    else:
        print("Choix invalide. Veuillez relancer le programme.")
        return

    # Exporter le modèle et les données de validation
    export_model_and_validation_data(
        model=model,
        X_val=X_val,
        y_val=y_val,
        model_filename="model_train.pkl",
        data_filename="validation_data.csv"
    )

# Exécuter le script principal
if __name__ == "__main__":
    main()

