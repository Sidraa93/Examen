# Importation des bibliothèques 

from sklearn.datasets import make_classification
import pandas as pd

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
