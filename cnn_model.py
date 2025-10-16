#parametre 2D ( 3,64, 3)
#             (64,32,3)
#             (32,16,3)
#
#
#linear (16*26*26) modifier dans view

import numpy as np
import pickle
import torch as th
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Fixe les graines
np.random.seed(0)
th.manual_seed(0)

# === Chargement du dataset d'entraînement ===
with open("dataset_images_train", 'rb') as fo:
    data_dict = pickle.load(fo, encoding='bytes')

X = data_dict[b'data'] if b'data' in data_dict else data_dict['data']
y = data_dict[b'target'] if b'target' in data_dict else data_dict['target']

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# === Normalisation ===
mean = X.mean()
std = X.std()
X = (X - mean) / std

# === Reshape en (N, 3, 32, 32) ===
X = X.reshape(-1, 3, 32, 32)

# === Split train/test ===
indices = np.random.permutation(len(X))
n_train = int(0.7 * len(X))
train_idx, test_idx = indices[:n_train], indices[n_train:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# === Device ===
device = th.device("mps" if th.has_mps else ("cuda" if th.cuda.is_available() else "cpu"))
print("Device utilisé :", device)

# === Tensors & DataLoader ===
X_train_t = th.from_numpy(X_train).to(device)
y_train_t = th.from_numpy(y_train).to(device)
X_test_t = th.from_numpy(X_test).to(device)
y_test_t = th.from_numpy(y_test).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# === CNN adapté à des images 32x32 RGB ===
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)    # 32→30
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)   # 30→28
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3)   # 28→26
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(16 * 26 * 26, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (64, 30, 30)
        x = F.relu(self.conv2(x))  # (32, 28, 28)
        x = F.relu(self.conv3(x))  # (16, 26, 26)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

nnet = CNNClassifier().to(device)

# === Entraînement ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nnet.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=50)

train_errors, test_errors = [], []
best_loss = float('inf')
epochs_no_improve = 0
patience = 5
best_model_state = None

pbar = tqdm(range(100))
for epoch in pbar:
    nnet.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = nnet(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

    nnet.eval()
    with th.no_grad():
        train_preds = nnet(X_train_t).argmax(dim=1)
        test_preds = nnet(X_test_t).argmax(dim=1)
        train_error = (train_preds != y_train_t).float().mean().item()
        test_error = (test_preds != y_test_t).float().mean().item()
        train_errors.append(train_error)
        test_errors.append(test_error)

        current_loss = criterion(nnet(X_test_t), y_test_t).item()
        pbar.set_description(f"Epoch {epoch+1}")
        pbar.set_postfix(train=f"{train_error:.2f}", test=f"{test_error:.2f}")

        if current_loss < best_loss:
            best_loss = current_loss
            best_model_state = nnet.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping à l'époque {epoch+1}")
                break

# Recharge le meilleur modèle
if best_model_state:
    nnet.load_state_dict(best_model_state)
    print("Modèle restauré au meilleur état.")

# === Plot des erreurs ===
plt.plot(train_errors, label="Erreur train")
plt.plot(test_errors, label="Erreur test")
plt.xlabel("Époque")
plt.ylabel("Taux d'erreur")
plt.legend()
plt.title("Évolution du taux d'erreur")
plt.grid(True)
plt.show()

# === Prédiction sur le vrai test ===
def predict_and_save(model, mean, std, output_file="predictions_cnn.csv"):
    with open("data_images_test", 'rb') as fo:
        test_dict = pickle.load(fo, encoding='bytes')

    X_test = test_dict[b'data'] if b'data' in test_dict else test_dict['data']
    X_test = np.array(X_test, dtype=np.float32)
    X_test = (X_test - mean) / std
    X_test = X_test.reshape(-1, 3, 32, 32)

    X_test_tensor = th.from_numpy(X_test).to(device)

    model.eval()
    with th.no_grad():
        y_pred = model(X_test_tensor).argmax(dim=1).cpu().numpy()

    np.savetxt(output_file, y_pred, fmt='%d')
    print(f"Prédictions sauvegardées dans {output_file}")

predict_and_save(nnet, mean, std)
