# Importation des bibliothèques 

import matplotlib.pyplot as plt  # Pour la visualisation
from sklearn.decomposition import PCA # Importer la classe PCA
from sklearn.datasets import make_classification #Générer des données



# Fonction pour générer des données avec des paramètres variables

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
    return X, y


X, y = generate_data()
print(f"Données générées : {X.shape[0]} échantillons")
