import numpy as np
import pickle
import torch as th
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1) Fixer les graines pour la reproductibilité
np.random.seed(0)
th.manual_seed(0)
if th.cuda.is_available():
    th.cuda.manual_seed(0)

# --- 2) Chargement dataset_images_train
with open("dataset_images_train", 'rb') as fo:
    dict_data = pickle.load(fo, encoding='bytes')

# Récupération X, y (compatibilité bytes/str)
X_full = dict_data[b'data'] if b'data' in dict_data else dict_data['data']
y_full = dict_data[b'target'] if b'target' in dict_data else dict_data['target']

X_full = np.array(X_full, dtype=np.float32)
y_full = np.array(y_full, dtype=np.int64)

# --- 3) Normalisation
mean = X_full.mean()
std = X_full.std()
X_full = (X_full - mean) / std

# (Optionnel) **Sous-échantillonnage** pour aller + vite (décommente si besoin)
# X_full = X_full[:5000]   # ex. limiter à 5000
# y_full = y_full[:5000]

# --- 4) Split train/test (70%/30%)
indices = np.random.permutation(len(X_full))
n_train = int(0.7 * len(X_full))
train_idx = indices[:n_train]
test_idx = indices[n_train:]

X_train = X_full[train_idx]
y_train = y_full[train_idx]
X_test = X_full[test_idx]
y_test = y_full[test_idx]

# --- 5) Création Tensors & DataLoader
device = th.device("mps" if th.has_mps else ("cuda" if th.cuda.is_available() else "cpu"))
print("Device utilisé :", device)

X_train_t = th.from_numpy(X_train).to(device)
y_train_t = th.from_numpy(y_train).to(device)
X_test_t = th.from_numpy(X_test).to(device)
y_test_t = th.from_numpy(y_test).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- 6) Définition d'un réseau **léger** + BatchNorm + Dropout
class SmallNet(nn.Module):
    def __init__(self, d, k, h1=256, h2=128 , h3=64 , dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h3, k)  # logits (CrossEntropyLoss fait le softmax)
        )

    def forward(self, x):
        return self.net(x)

# Dimensions d'entrée/sortie
d = X_train.shape[1]  # nb de features
k = 10                # nb classes (d'après ton code)

nnet = SmallNet(d, k).to(device)

# --- 7) Préparation entraînement
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nnet.parameters(), lr=0.01)

# Scheduler OneCycleLR : il fait varier le LR au cours de l'entraînement (super pour converger vite)
# Nombre total de "batches" sur toutes les époques
epochs = 300
steps_per_epoch = len(train_loader)
scheduler = th.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs
)

# AMP : accélération en demi-précision si on a un GPU CUDA récent
scaler = th.cuda.amp.GradScaler() if (device.type == "cuda") else None

# Early stopping
patience = 4
best_loss = float('inf')
epochs_no_improve = 0

train_errors = []
test_errors = []

# --- 8) Boucle d'entraînement
pbar = tqdm(range(epochs))
for epoch in pbar:
    nnet.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        if scaler is not None:
            # Entraînement en précision mixte
            with th.cuda.amp.autocast():
                outputs = nnet(X_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Entraînement classique
            outputs = nnet(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        scheduler.step()  # Mise à jour du LR à chaque batch

    # --- Évaluation fin d'époque
    nnet.eval()
    with th.no_grad():
        # Calcul loss train complet
        train_outputs = nnet(X_train_t)
        train_loss = criterion(train_outputs, y_train_t).item()
        train_pred = train_outputs.argmax(dim=1)
        train_error = (train_pred != y_train_t).float().mean().item()

        # Calcul loss test complet
        test_outputs = nnet(X_test_t)
        test_loss = criterion(test_outputs, y_test_t).item()
        test_pred = test_outputs.argmax(dim=1)
        test_error = (test_pred != y_test_t).float().mean().item()

        train_errors.append(train_error)
        test_errors.append(test_error)

    # Affichage
    pbar.set_description(f"Ep {epoch+1}/{epochs}")
    pbar.set_postfix({
        "train_loss": f"{train_loss:.4f}",
        "test_loss":  f"{test_loss:.4f}",
        "train_err":  f"{train_error*100:.2f}%",
        "test_err":   f"{test_error*100:.2f}%"
    })

    # Early Stopping
    if test_loss < best_loss:
        best_loss = test_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping à l'époque {epoch+1}")
        break


# --- 11) Affichage des courbes d'erreur
plt.figure(figsize=(10, 6))
plt.plot(train_errors, label='Erreur train')
plt.plot(test_errors, label='Erreur test')
plt.xlabel('Époque')
plt.ylabel('Taux d\'erreur')
plt.title('Évolution du taux d\'erreur')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 9) Fonction de prédiction sur le vrai fichier test
def predict_and_save(model, mean, std, output_file="predictions_neural.csv"):
    with open("data_images_test", 'rb') as fo:
        test_dict = pickle.load(fo, encoding='bytes')

    X_test_final = test_dict[b'data'] if b'data' in test_dict else test_dict['data']
    X_test_final = np.array(X_test_final, dtype=np.float32)
    X_test_final = (X_test_final - mean) / std

    X_test_final_t = th.from_numpy(X_test_final).to(device)

    model.eval()
    with th.no_grad():
        y_pred = model(X_test_final_t).argmax(dim=1)
    y_pred_np = y_pred.cpu().numpy()

    np.savetxt(output_file, y_pred_np, fmt='%d')
    print(f"Prédictions sauvegardées dans {output_file}")

# --- 10) Exporter les prédictions
predict_and_save(nnet, mean, std, "predictions_neural.csv")
