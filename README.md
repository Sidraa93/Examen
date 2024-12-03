# **Projet Collaboratif : Pipeline de Classification et Dockerisation**

## **Introduction**

Ce projet est une simulation d'un environnement collaboratif où plusieurs développeurs travaillent ensemble pour construire un pipeline de machine learning complet. L'objectif est de couvrir toutes les étapes d'un projet machine learning, de la génération des données jusqu'à leur déploiement dans un conteneur Docker.  
Chaque membre contribue en développant une fonctionnalité dans une branche dédiée.

---

## **Structure et Objectifs**

### **Objectifs du projet**
1. Générer des données synthétiques pour simuler un problème de classification.
2. Développer, entraîner et optimiser des modèles de classification.
3. Évaluer les performances des modèles à l'aide de techniques avancées comme la validation croisée.
4. Implémenter un script de prédiction basé sur les modèles entraînés.
5. Dockeriser le projet pour faciliter le partage et le déploiement.

### **Architecture des branches**
- **`main` :** Branche principale contenant toute la version du projet.
- **`data_generation` :** Génération des données.
- **`classification` :** Développement des modèles et évaluation.
- **`prediction` :** Chargement des modèles pour effectuer des prédictions.
- **`docker_integration` :** Contient les fichiers Docker et docker-compose.

---

## **Description des Données**

Les données utilisées dans ce projet sont générées grâce à la bibliothèque `scikit-learn`.  
- **Nombre d'échantillons :** 10,000  
- **Nombre de features :** 20  
- **Classes :** 3  
- **Proportions des données :**  
  - **70 %** pour l'entraînement  
  - **20 %** pour le test  
  - **10 %** pour la validation  

Les données sont exportées dans un fichier nommé `synthetic_data.csv` et utilisées pour entraîner différents modèles de classification.

---

## **Démarche Méthodologique**

### **Étapes Suivies :**

1. **Génération et Visualisation des Données :**
   - Création de données synthétiques équilibrées.
   - Analyse des paramètres comme `n_informative`, `class_sep`, et `n_clusters_per_class`.

2. **Entraînement des Modèles :**
   - Modèles testés :
     - Random Forest
     - Régression Logistique
     - SVM
   - Optimisation des hyperparamètres à l'aide de `GridSearchCV`.

3. **Évaluation :**
   - Métriques utilisées : Précision, rappel, F1-score.
   - Matrice de confusion pour visualiser les erreurs.
   - Validation croisée (5-fold) pour évaluer la robustesse des modèles.

4. **Prédiction :**
   - Script `predict_classification.py` pour effectuer des prédictions sur les données de validation.
   - Évaluation des prédictions et affichage des performances.

5. **Dockerisation :**
   - Le projet est dockerisé pour garantir une portabilité totale.  
   - Fichiers : 
     - `Dockerfile` : Crée une image Docker pour le projet.
     - `docker-compose.yml` : Orchestration des conteneurs.

---

## **Résultats**

### **Performance des Modèles :**


### **Visualisation :**
- **Matrice de Confusion :** Permet de visualiser les erreurs de classification.

---

## **Structure du Projet**

### **Organisation des Fichiers :**

```
.
├── notebooks/
│   └── Visualisation_données.ipynb   # Visualisation des données générées
├── train_classifier.py               # Génération, entraînement et évaluation des modèles
├── predict_classification.py         # Prédiction et évaluation sur les données de validation
├── synthetic_data.csv                # Données générées (non suivies par Git)
├── Dockerfile                        # Fichier Docker pour le projet
├── docker-compose.yml                # Configuration des conteneurs Docker
└── README.md                         # Documentation du projet
```

---

## **Utilisation du Projet**

### **1. Lancer le Projet :**

1. **Cloner le projet :**
   ```bash
   git clone git@github.com:<ton_repo>/classification_project.git
   cd classification_project
   ```

2. **Lancer les scripts :**
   - Génération et entraînement :  
     ```bash
     python train_classifier.py
     ```
   - Prédiction :  
     ```bash
     python predict_classification.py
     ```

### **2. Dockerisation :**

1. **Construire l'image Docker :**
   ```bash
   docker build -t classification_project .
   ```

2. **Exécuter le conteneur Docker :**
   ```bash
   docker run -it --rm classification_project
   ```

3. **Avec Docker Compose :**
   - Lancer :  
     ```bash
     docker-compose up
     ```
   - Arrêter :  
     ```bash
     docker-compose down
     ```

---

## **Limites et Améliorations**

### **Limites :**
- Les données utilisées sont synthétiques et ne reflètent pas des cas réels.
- La complexité de certains modèles peut ralentir l'exécution.

### **Améliorations possibles :**
1. Ajouter des modèles supplémentaires (ex. Gradient Boosting, XGBoost).
2. Intégrer une interface utilisateur pour rendre le pipeline plus interactif.
3. Permettre la mise à jour des modèles directement via Docker.

---

## **Contributeurs**

- **Sidraa93** : Analyse des données, implémentation des prédictions.  
- **Dineshan12** : Développement des modèles et optimisation.  
- **Collaborateurs** : Dockerisation et documentation.

---

Ce README revisité est simple, structuré et met en avant l'originalité du projet. Tu peux le personnaliser selon tes besoins ! 🚀
