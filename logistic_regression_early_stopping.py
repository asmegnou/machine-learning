import numpy as np
import pickle
import torch as th
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Configuration de device
device = th.device("mps" if th.has_mps else "cpu")
print(f"PyTorch utilise : {device}")

# Fixer la graine aléatoire
np.random.seed(0)
th.manual_seed(0)
th.cuda.manual_seed(0)

# Chargement des données
with open("dataset_images_train", 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

data = np.array(dict['data'])
target = np.array(dict['target'])

# Normalisation des données
mean = data.mean()
std = data.std()
data = (data - mean) / std

# Séparation des données
dim = data.shape[1]
k = 10  # Nombre de classes

indices = np.random.permutation(data.shape[0])
training_idx, valid_idx = indices[:int(data.shape[0] * 0.7)], indices[int(data.shape[0] * 0.7):]
X_train = data[training_idx, :]
Y_train = target[training_idx]
X_valid = data[valid_idx, :]
Y_valid = target[valid_idx]

# Même classe avec une seule couche mais améliorée
class Reg_log_mult(th.nn.Module):
    def __init__(self, d, k):
        super(Reg_log_mult, self).__init__()
        self.layer = th.nn.Linear(d, k)
        # Initialisation améliorée des poids
        th.nn.init.kaiming_normal_(self.layer.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.layer.bias is not None:
            th.nn.init.constant_(self.layer.bias, 0)
        self.dropout = th.nn.Dropout(0.2)  # Ajout de dropout même avec une couche

    def forward(self, x):
        x = self.dropout(x)  # Dropout sur les inputs
        out = self.layer(x)
        return F.log_softmax(out, dim=1)  # Plus stable numériquement que softmax

# Initialisation du modèle
model = Reg_log_mult(dim, k).to(device)

# Conversion des données et création des DataLoaders
batch_size = 256
train_dataset = TensorDataset(th.from_numpy(X_train).float().to(device), 
                             th.from_numpy(Y_train).long().to(device))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

X_validation = th.from_numpy(X_valid).float().to(device)
y_validation = th.from_numpy(Y_valid).long().to(device)

# Hyperparamètres optimisés
optimizer = th.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
criterion = th.nn.CrossEntropyLoss()

# Entraînement avec early stopping
nb_epochs = 1000
best_error = float('inf')
patience = 15
no_improve = 0
min_delta = 0.0001

train_errors = []
valid_errors = []
epochs_list = []

for epoch in tqdm(range(nb_epochs)):
    model.train()
    epoch_train_error = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        preds = th.argmax(outputs, 1)
        epoch_train_error += (preds != batch_y).float().mean().item()
    
    # Erreur moyenne sur l'epoch
    epoch_train_error /= len(train_loader)
    train_errors.append(epoch_train_error)
    
    # Validation
    model.eval()
    with th.no_grad():
        valid_outputs = model(X_validation)
        valid_loss = criterion(valid_outputs, y_validation)
        valid_error = (th.argmax(valid_outputs, 1) != y_validation).float().mean().item()
        valid_errors.append(valid_error)
    
    epochs_list.append(epoch)
    
    # Early stopping et sauvegarde
    if valid_error < (best_error - min_delta):
        best_error = valid_error
        no_improve = 0
        th.save(model.state_dict(), 'best_single_layer_model.pth')
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping à l'epoch {epoch} avec erreur {best_error:.4f}")
            break
    
    scheduler.step(valid_loss)  # Ajustement du LR

    # Affichage dynamique
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Train Error: {epoch_train_error:.4f}, Valid Error: {valid_error:.4f}')

# Charger le meilleur modèle
model.load_state_dict(th.load('best_single_layer_model.pth'))

# Visualisation
plt.figure(figsize=(12, 5))
plt.plot(epochs_list, train_errors, label='Train Error')
plt.plot(epochs_list, valid_errors, label='Validation Error')
plt.axhline(y=best_error, color='r', linestyle='--', label='Best Error')
plt.xlabel('Epochs')
plt.ylabel('Error Rate')
plt.title('Training and Validation Error')
plt.legend()
plt.grid(True)
plt.savefig('error_plot_single_layer.png')
plt.show()

# Prédiction et sauvegarde
def predict_and_save(model, X_test, output_file):
    model.eval()
    with th.no_grad():
        outputs = model(X_test)
        y_pred = th.argmax(outputs, 1)
        np.savetxt(output_file, y_pred.cpu().numpy(), fmt='%d')
    print(f"Résultats enregistrés dans {output_file}")

# Chargement des données de test
with open("data_images_test", 'rb') as fo:
    test_dict = pickle.load(fo, encoding='bytes')

X_test = np.array(test_dict['data'])
X_test = (X_test - mean) / std
X_test = th.from_numpy(X_test).float().to(device)

predict_and_save(model, X_test, "predictions_single_layer.csv")