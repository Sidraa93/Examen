# 1.Génération des données : 

# Importation des bibliothèques

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Fonction pour générer des données avec des paramètres variables

def generate_data(n_samples=10000, n_features=20, n_informative=10, n_classes=3,
                  n_clusters_per_class=2, class_sep=1.0, random_state=42):
    # Générer les données synthétiques
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

# Exécution principale

if __name__ == "__main__":
    data = generate_data()
    print(f"Données générées : {data.shape[0]} échantillons avec {data.shape[1] - 1} features")

# Sauvegarder les données dans un fichier CSV

    data.to_csv("synthetic_data.csv", index=False)
    print("Données générées et sauvegardées dans synthetic_data.csv")


# 2. Classification :

# Division des données

def split_data(X, y):
    """
    Divise les données en trois ensembles : entraînement, test et validation.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
    print(f"Taille des ensembles :\n- Entraînement : {len(X_train)}\n- Test : {len(X_test)}\n- Validation : {len(X_val)}")
    return X_train, X_test, X_val, y_train, y_test, y_val
# Fonction pour entraîner et évaluer un modèle unique
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


# 3. Cross-validation et évaluation
def evaluate_with_cross_validation(X_train, y_train, model, cv=5):
    """
    Effectue une validation croisée pour évaluer les performances du modèle.
    """
    scoring = ['accuracy', 'f1_weighted']
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
    mean_accuracy = np.mean(scores['test_accuracy'])
    print(f"Cross-validation (Accuracy): Moyenne = {mean_accuracy:.2f}")
    return mean_accuracy

# 4. Affichage de la matrice de confusion
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

# 5. Optimisation des hyperparamètres avec Grid Search
def optimize_model_with_grid_search(model, param_grid, X_train, y_train, cv=5):
    """
    Optimise les hyperparamètres d'un modèle avec Grid Search.
    """
    print(f"\nOptimisation des hyperparamètres pour le modèle : {type(model).__name__}")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Meilleurs paramètres trouvés : {grid_search.best_params_}")
    return grid_search.best_estimator_

# 6. Exportation du modèle et des données de validation
def export_model_and_validation_data(model, X_val, y_val, model_filename="model_train.pkl", data_filename="validation_data.csv"):
    import joblib
    import pandas as pd

    # Sauvegarder le modèle
    joblib.dump(model, model_filename)
    print(f"Modèle sauvegardé dans le fichier : {model_filename}")

    # Sauvegarder les données de validation
    validation_data = pd.DataFrame(X_val, columns=[f"feature_{i}" for i in range(X_val.shape[1])])
    validation_data["target"] = y_val
    validation_data.to_csv(data_filename, index=False)
    print(f"Données de validation sauvegardées dans le fichier : {data_filename}")


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
        model,
        X_val,
        y_val,
        model_filename="model_train.pkl",
        data_filename="validation_data.csv"
    )

# Exécuter le script principal
if __name__ == "__main__":
    main()
