import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import torch as th
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F
import  pickle

#0) Choix d'une graine aléatoire
np.random.seed(0)
th.manual_seed(0)
th.cuda.manual_seed(0)





#1) Chargement des données


with open("dataset_images_train", 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

data = dict['data']

X = np.array(dict['data'])  # Accéder aux données avec b'data' pour les clés en bytes
y = np.array(dict['target'])  # Accéder aux étiquettes avec b'target'

# Normalisation des données
mean = data.mean()
std = data.std()
data = (data - mean) / std

# Séparation des données en ensembles d'entraînement et de validation
d = data.shape[1]
k = 10  # Nombre de classes



#2) fonction qui calcule les prédictions (0, 1 ou 2) à partir des sorties du modèle
def prediction(f):
    return th.argmax(f, 1)

#3) Fonction qui calcule le taux d'erreur en comparant les y prédits avec les y réels
def error_rate(y_pred,y):
    return ((y_pred != y).sum().float())/y_pred.size()[0]

#5) Séparation aléatoire du dataset en ensemble d'apprentissage (70%) et de test (30%)
indices = np.random.permutation(X.shape[0])
training_idx, test_idx = indices[:int(X.shape[0]*0.7)], indices[int(X.shape[0]*0.7):]
X_train = X[training_idx,:]
y_train = y[training_idx]

X_test = X[test_idx,:]
y_test = y[test_idx]





#6) Création du réseau de neurones. Il étend la classe th.nn.Module de la librairie Pytorch
class Neural_network_multi_classif(th.nn.Module):

    # Constructeur qui initialise le modèle
    def __init__(self,d,k,h1,h2):
        super(Neural_network_multi_classif, self).__init__()

        self.layer1 = th.nn.Linear(d, h1)
        self.layer2 = th.nn.Linear(h1, h2)
        self.layer3 = th.nn.Linear(h2, k)

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()

    # Implémentation de la passe forward du modèle
    def forward(self, x):
        phi1 = F.sigmoid(self.layer1(x))
        phi2 = F.sigmoid(self.layer2(phi1))

        return F.softmax(self.layer3(phi2),1)


#7) creation d'un réseau de neurones avec deux couches cachées de taille 200 et 100, et d classes en sortie
nnet = Neural_network_multi_classif(d,k,200,100)

#8) Spécification du materiel utilisé device = "cpu" pour du calcul CPU, device = "cuda:0" pour du calcul sur le device GPU "cuda:0".
device = th.device("mps" if th.has_mps else "cpu")
print(f"PyTorch utilise : {device}")

#9) Chargement du modèle sur le matériel choisi
nnet = nnet.to(device)


#10) Conversion des données en tenseurs Pytorch et envoi sur le device
X_train = th.from_numpy(X_train).float().to(device)
y_train = th.from_numpy(y_train).long().to(device)

X_test = th.from_numpy(X_test).float().to(device)
y_test = th.from_numpy(y_test).long().to(device)


#11) Taux d'apprentissage (learning rate)
eta = 0.01

#12) Définition du critère de Loss. Ici cross entropy pour un modèle de classification multi classe
criterion = th.nn.CrossEntropyLoss()

# optim.SGD Correspond à la descente de gradient standard.
# Il existe d'autres types d'optimizer dans la librairie Pytorch
# Le plus couramment utilisé est optim.Adam
optimizer = optim.Adam(nnet.parameters(), lr=eta)

# tqdm permet d'avoir une barre de progression
nb_epochs = 100000
pbar = tqdm(range(nb_epochs))

for i in pbar:
    # Remise à zéro des gradients
    optimizer.zero_grad()

    f_train = nnet(X_train)

    loss = criterion(f_train,y_train)
    # Calculs des gradients
    loss.backward()

    # Mise à jour des poids du modèle avec l'optimiseur choisi et en fonction des gradients calculés
    optimizer.step()

    if (i % 1000 == 0):

        y_pred_train = prediction(f_train)

        error_train = error_rate(y_pred_train,y_train)
        loss = criterion(f_train,y_train)

        f_test = nnet(X_test)
        y_pred_test = prediction(f_test)

        error_test = error_rate(y_pred_test, y_test)

        pbar.set_postfix(iter=i, loss = loss.item(), error_train=error_train.item(), error_test=error_test.item())







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
predict_and_save(nnet, X_test, "predictions_neural.csv")