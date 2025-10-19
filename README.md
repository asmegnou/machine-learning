# Mini-Challenge : Classification d'Images en Couleur

**Auteur :** Assia MEGNOUNIF  
**Licence :** Licence 3 Informatique

## 1. Présentation du challenge

Le projet consiste à résoudre un problème de **classification multi-classes** à partir d’un jeu de données d’images en couleur de taille **32x32 pixels**, réparties en **10 classes**.

Chaque image est codée sous forme d’un vecteur de taille **3072** (3 canaux RGB × 32 × 32 pixels).

**Objectif :**
- Entraîner des modèles de classification sur un **jeu d’apprentissage de 20 000 images**.
- Prédire les classes des **10 000 images de test**.

## 2. Observation des données

- Les données sont **très condensées** et réparties dans un **espace de grande dimension**, ce qui rend leur classification non triviale.
- Visualisation réalisée avec **t-SNE** en 2D et 3D :
  - Les classes sont très **entremêlées**, montrant des frontières complexes.

## 3. Méthodes de classification

### 3.1 K plus proches voisins (KNN)

- k = 3 → **Précision : 29.76%**  
- k = 6 → **Précision : 30.38%**

**Conclusion :**
- Simple mais **peu performant** pour ce jeu de données.  
- **Temps de calcul élevé**.

### 3.2 Régression logistique multivariée

- Sans early stopping → 33.97%
- Avec early stopping → 36.3%

**Remarques :**
- Early stopping aide à éviter le surapprentissage.
- Optimiseur impactant :
  - **SGD** → convergence lente, erreur stagnante ~0.6
  - **Adam** → convergence rapide et stable

### 3.3 Réseau de neurones fully connected (MLP)

| Configuration | Précision (%) | Remarques |
|---------------|---------------|-----------|
| Modèle long, beaucoup d’époques | 11.68 | Surapprentissage ou mauvais hyperparamètres |
| 3000 époques + early stopping | 24.94 | Meilleur mais insuffisant |
| Petit réseau rapide, BatchNorm + Dropout, 30 epochs | 49.13 | Early stopping utile |
| Même architecture, 300 epochs, sans early stopping | 46.32 | Performance moins stable |
| h1=256, h2=64 + early stopping | 49.19 | Bon compromis |
| h1=256, h2=128, h3=64 + early stopping | 48.4 | Légère baisse |

**Conclusion :**
- Dimensions, dropout, early stopping et nombre d’époques impactent les performances.
- BatchNorm + Dropout améliore significativement les résultats.

### 3.4 Réseau de neurones convolutionnel (CNN)

- **Précision obtenue : 53.46%**
- Exploite la **structure spatiale** des images, ce qui explique sa supériorité.

## 4. Conclusion

**Récapitulatif des performances :**

| Méthode | Précision (%) |
|---------|---------------|
| KNN (k=3) | 29.76 |
| KNN (k=6) | 30.38 |
| Régression logistique | 33.97 |
| Régression logistique (ES) | 36.30 |
| MLP simple | 11.68 |
| MLP (3000 epochs + ES) | 24.94 |
| MLP rapide (30 epochs) | 49.13 |
| MLP rapide (300 epochs) | 46.32 |
| MLP ES (256, 64) | 49.19 |
| MLP ES (256,128,64) | 48.40 |
| CNN | 53.46 |

**Conclusion principale :**
- Le **CNN** est la méthode la plus adaptée.
- Les autres méthodes nécessitent un **réglage fin des hyperparamètres** et restent limitées.
