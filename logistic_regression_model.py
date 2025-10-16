import numpy as np
import pickle
import torch as th
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration de device (GPU MPS ou CPU) : J'ai un macOS donc j'ai MPS
device = th.device("mps" if th.has_mps else "cpu")
print(f"PyTorch utilise : {device}")

# Fixer la graine aléatoire pour la reproductibilité
np.random.seed(0)
th.manual_seed(0)
th.cuda.manual_seed(0)

# Chargement des données
with open("dataset_images_train", 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

data = np.array(dict['data'])  # Accéder aux données avec b'data' pour les clés en bytes
target = np.array(dict['target'])  # Accéder aux étiquettes avec b'target'

# Normalisation des données
mean = data.mean()
std = data.std()
data = (data - mean) / std

# Séparation des données en ensembles d'entraînement et de validation
dim = data.shape[1]
k = 10  # Nombre de classes

indices = np.random.permutation(data.shape[0])
training_idx, valid_idx = indices[:int(data.shape[0] * 0.7)], indices[int(data.shape[0] * 0.7):]
X_train = data[training_idx, :]
Y_train = target[training_idx]

X_valid = data[valid_idx, :]
Y_valid = target[valid_idx]

# Définition du modèle de régression logistique multinomiale
class Reg_log_mult(th.nn.Module):
    def __init__(self, d, k):
        super(Reg_log_mult, self).__init__()
        self.layer = th.nn.Linear(d, k)
        self.layer.reset_parameters()

    def forward(self, x):
        out = self.layer(x)
        return F.softmax(out, 1)

# Initialisation du modèle
model = Reg_log_mult(dim, k).to(device)

# Conversion des données en tenseurs PyTorch
X_train = th.from_numpy(X_train).float().to(device)
y_train = th.from_numpy(Y_train).long().to(device)  # Conversion en long pour CrossEntropyLoss

X_validation = th.from_numpy(X_valid).float().to(device)
y_validation = th.from_numpy(Y_valid).long().to(device)  # Conversion en long pour CrossEntropyLoss

# Hyperparamètres
eta = 0.0003  # Taux d'apprentissage
criterion = th.nn.CrossEntropyLoss()  # Fonction de perte
optimizer = th.optim.Adam(model.parameters(), lr=eta)  # Optimiseur

# Entraînement du modèle
nb_epochs = 10000
pbar = tqdm(range(nb_epochs))

error = []
epochs = []

for i in pbar:
    optimizer.zero_grad()  # Remise à zéro des gradients

    f_train = model(X_train)  # Prédictions sur l'ensemble d'entraînement
    loss = criterion(f_train, y_train)  # Calcul de la perte

    loss.backward()  # Rétropropagation
    optimizer.step()  # Mise à jour des poids

    if i % 10 == 0:
        y_pred_train = th.argmax(f_train, 1)  # Prédictions sur l'ensemble d'entraînement
        error_train = ((y_pred_train != y_train).sum().float()) / y_pred_train.size(0)  # Taux d'erreur

        f_valid = model(X_validation)  # Prédictions sur l'ensemble de validation
        y_pred_valid = th.argmax(f_valid, 1)
        error_valid = ((y_pred_valid != y_validation).sum().float()) / y_pred_valid.size(0)

        error.append(error_valid.item())
        epochs.append(i)

        pbar.set_postfix(iter=i, loss=loss.item(), error_train=error_train.item(), error_valid=error_valid.item())

# Affichage de l'évolution de l'erreur de validation
def plot_validation_error(valid_errors, epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, valid_errors, label="Erreur validation", color="red")
    plt.xlabel("Époques")
    plt.ylabel("Taux d'erreur")
    plt.title("Évolution du taux d'erreur de validation")
    plt.legend()
    plt.grid(True)
    plt.savefig("taux_erreur_ADAM.png")
    plt.show()

plot_validation_error(error, epochs)

# Prédiction et sauvegarde des résultats
def predict_and_save(model, X_test, output_file):
    with th.no_grad():
        f_test = model(X_test)
        y_pred = th.argmax(f_test, 1)

    # Sauvegarder les prédictions
    np.savetxt(output_file, y_pred.cpu().numpy(), fmt='%d')
    print(f"Résultats enregistrés dans {output_file}")

# Charger et convertir les données de test
with open("data_images_test", 'rb') as fo:
    test_dict = pickle.load(fo, encoding='bytes')

X_test = np.array(test_dict['data'])
X_test = (X_test - mean) / std  # Normalisation des données de test
X_test = th.from_numpy(X_test).float().to(device)

# Exécuter la prédiction et sauvegarder les résultats
predict_and_save(model, X_test, "predictions_rmv.csv")