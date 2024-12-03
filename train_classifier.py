# Génération des données : 

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

 # Fonction pour diviser les données
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
    print(f"Taille des ensembles :\n- Entraînement : {len(X_train)}\n- Test : {len(X_test)}\n- Validation : {len(X_val)}")
    return X_train, X_test, X_val, y_train, y_test, y_val

# Fonction pour effectuer une cross-validation
def evaluate_with_cross_validation(X_train, y_train, model, cv=5):
    scoring = ['accuracy', 'f1_weighted']
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
    mean_accuracy = np.mean(scores['test_accuracy'])
    print(f"Cross-validation (Accuracy): Moyenne = {mean_accuracy:.2f}")
    return mean_accuracy

# Fonction pour afficher une matrice de confusion
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités terrain")
    plt.title(f"Matrice de Confusion - {model_name}")
    plt.show()

# Fonction pour choisir un modèle
def get_model_by_choice(choice):
    models = {
        "random_forest": RandomForestClassifier(random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "svm": SVC(random_state=42)
    }
    if choice not in models:
        raise ValueError("Choix non valide. Options : 'random_forest', 'logistic_regression', 'svm'.")
    return models[choice]

# Fonction pour entraîner et évaluer un modèle unique
def train_and_evaluate(X_train, y_train, X_test, y_test, model_choice):
    model = get_model_by_choice(model_choice)

    # Cross-validation
    print(f"\nEntraînement et cross-validation pour le modèle : {model_choice}")
    evaluate_with_cross_validation(X_train, y_train, model)

    # Entraîner le modèle
    model.fit(X_train, y_train)
    print(f"Modèle {model_choice} entraîné avec succès.")

    # Prédictions et évaluation
    y_pred = model.predict(X_test)
    print("\nÉvaluation du modèle :")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    plot_confusion_matrix(y_test, y_pred, model_choice)

# Fonction pour entraîner et évaluer plusieurs modèles
def evaluate_multiple_models(X_train, y_train, X_test, y_test):
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"\nEntraînement et évaluation du modèle : {name}")

        # Cross-validation
        mean_accuracy = evaluate_with_cross_validation(X_train, y_train, model)

        # Entraîner et évaluer
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_accuracy = model.score(X_test, y_test)
        print(classification_report(y_test, y_pred))
        plot_confusion_matrix(y_test, y_pred, name)

        # Stocker les résultats
        results[name] = {"Cross-Validation Accuracy": mean_accuracy, "Test Accuracy": test_accuracy}

    print("\nRésumé des performances :")
    for model, scores in results.items():
        print(f"{model}: {scores}")
    return results

# Fonction principale avec le menu interactif
def main():
    # Charger les données
    data = pd.read_csv("synthetic_data.csv")
    X = data.drop(columns=["target"]).values
    y = data["target"].values

    # Diviser les données
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X, y)

    # Menu interactif pour le choix
    print("\nVoulez-vous :")
    print("1 - Tester un seul modèle")
    print("2 - Tester plusieurs modèles en même temps")
    choice = input("Entrez votre choix (1 ou 2) : ")

    if choice == "1":
        # Tester un seul modèle
        print("\nChoisissez un modèle :")
        print("- random_forest")
        print("- logistic_regression")
        print("- svm")
        model_choice = input("Entrez le nom du modèle : ")
        train_and_evaluate(X_train, y_train, X_test, y_test, model_choice)

    elif choice == "2":
        # Tester plusieurs modèles
        evaluate_multiple_models(X_train, y_train, X_test, y_test)

    else:
        print("Choix invalide. Veuillez relancer le programme.")

# Lancer le script
if __name__ == "__main__":
    main()

