import numpy as np 
import  pickle
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch 
from tqdm import tqdm
#configuration de device (GPU MPS ou CPU ) : J'ai un macos donc j'ai mps 
device = torch.device("mps" if torch.has_mps else "cpu")
print(f"PyTorch utilise : {device}")


# Fixer la graine aléatoire pour la reproductibilité
np.random.seed(42)  # Vous pouvez choisir n'importe quelle valeur pour la graine


with open("dataset_images_train", 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

data = dict['data']

import matplotlib.pyplot as plt 

plt.imshow(data[0].reshape(3,32,32).transpose(1,2,0))
plt.axis("off")
plt.show()

def affiche_image(num) :
    plt.imshow(data[num].reshape(3, 32, 32).transpose(1, 2, 0))
    plt.title(f"Classe : {num}")
    plt.axis("off")
    plt.show()

# Test sur une image spécifique
affiche_image(160)  


data = np.array(dict['data'])  
target = np.array(dict['target']) 



#scaler_tsne = StandardScaler()
#data_normalized = scaler_tsne.fit_transform(data) 

# Sélection des 3000 premières images
n_samples = 3000
data_subset = data[:n_samples]  # Prend les 3000 premières images
targets_subset = target[:n_samples]  # Prend les étiquettes correspondantes

# Normalisation des données (optionnel mais recommandé)
scaler = StandardScaler()
data_subset_normalized = scaler.fit_transform(data_subset)

# Réduction de dimension avec t-SNE
tsne = TSNE(n_components=2, init='random', perplexity=50, learning_rate=200, random_state=42)
data_2d = tsne.fit_transform(data_subset_normalized)

# Affichage des points avec une couleur différente pour chaque classe
plt.figure(figsize=(16, 10))
scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=targets_subset, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, ticks=range(10))
plt.title("Projection t-SNE des 3000 premières images")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig("repartition.png")
plt.show()

#projection 3D 

from mpl_toolkits.mplot3d import Axes3D

# Réduction de dimension avec t-SNE en 3D
tsne = TSNE(n_components=3, init='random', perplexity=50, learning_rate=200, random_state=42)
data_3d = tsne.fit_transform(data_subset_normalized)

# Création du graphique 3D
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot 3D avec une couleur différente pour chaque classe
scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=targets_subset, cmap='tab10', alpha=0.7)

# Ajouter une légende et des axes
ax.set_title("Projection t-SNE 3D des 3000 premières images")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_zlabel("Dimension 3")
fig.colorbar(scatter, ticks=range(10))

# Affichage
plt.savefig("repartition_3D.png")
plt.show()

#ALGO DES K PLUS PROCHE VOISIN :

#j'ai utilisé torch pour accélérer les calculs

# Séparation des données pour l'algorithme des k-plus proches voisins (KNN)
X_train, X_valid, Y_train, Y_valid = train_test_split(
    data, target, test_size=0.2, random_state=42)  # 20% de données de validation

#renormaliser les données

moyene = X_train.mean(axis=0)
standar = X_train.std(axis=0)

X_train=(X_train-moyene) / standar

print(X_train.mean(axis=0))
print(X_train.std(axis=0))


X_valid =(X_valid-moyene)/standar

print(X_valid.mean(axis=0))
print(X_valid.std(axis=0))

print(f"Taille des données d'entraînement : {len(X_train)}")
print(f"Taille des données de validation : {len(X_valid)}")

# Conversion en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.int64).to(device)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(device)
Y_valid_tensor = torch.tensor(Y_valid, dtype=torch.int64).to(device)

# Fonction de distance euclidienne
def euclidean_distance_torch(v1, v2):
    return torch.sqrt(torch.sum((v1 - v2) ** 2, dim=1))

# Fonction pour trouver les k plus proches voisins
def neighbors_torch(x_train, y_label, x_test, k):
    distances = euclidean_distance_torch(x_train, x_test)
    _, nearest_indices = torch.topk(distances, k, largest=False)
    return y_label[nearest_indices]

# Fonction de prédiction
def prediction_torch(neighbors):
    values, counts = torch.unique(neighbors, return_counts=True)
    return values[torch.argmax(counts)].item()

# Fonction d'évaluation
def evaluation_torch(X_train, Y_train, X_valid, Y_valid, k):
    correct_predictions = 0
    total = len(X_valid)

    for i in tqdm(range(X_valid.shape[0]), desc="Évaluation"):
        x_test = X_valid[i]
        nearest_neighbors = neighbors_torch(X_train, Y_train, x_test, k)
        predicted_label = prediction_torch(nearest_neighbors)
        if predicted_label == Y_valid[i].item():
            correct_predictions += 1

    accuracy = correct_predictions / total
    print("Accuracy:", accuracy)
    return accuracy

# Test avec k=5
#evaluation_torch(X_train_tensor, Y_train_tensor, X_valid_tensor, Y_valid_tensor, k=5)


# Choix du meilleur K
#list_accuracy = []
#for k in tqdm(range(1, 10)):
#    list_accuracy.append(evaluation_torch(X_train_tensor, Y_train_tensor, X_valid_tensor, Y_valid_tensor, k))
#
#print(list_accuracy)

##affichage de la courbe de la precision en fonction de k 
#x = range(1,10)
#plt.plot(x,list_accuracy)
#plt.xlabel("k")
#plt.title ("la courbe de précision en fonction de k sur l'ensemnle de validation ")
#plt.ylabel("precision")
#plt.savefig("choose_best_k.png")  # Sauvegarde en format PNG
#plt.show()


#faire le test 
# Chargement des données de test
with open("data_images_test", 'rb') as fo:
    test = pickle.load(fo, encoding='bytes')

data_test = np.array(test['data'])  # Convertir en NumPy array

# Normalisation des données de test
X_test = (data_test - moyene) / standar

# Prédictions avec KNN
results = np.zeros((X_test.shape[0]))

for i in tqdm(range(X_test.shape[0])):
    results[i] = prediction_torch(neighbors_torch(X_train_tensor, Y_train_tensor, torch.tensor(X_test[i], dtype=torch.float32).to(device), 6))

# Sauvegarde des résultats
np.savetxt("predictions_knn.csv", results)

