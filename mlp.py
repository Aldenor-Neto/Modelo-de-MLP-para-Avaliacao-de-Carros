import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import numpy as np  # Adicionado para manipulação de arrays

# Defina o caminho para a pasta onde os arquivos estão localizados
data_dir = 'car_evaluation'

# Carregar o dataset
data_path = os.path.join(data_dir, 'car.data')
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data = pd.read_csv(data_path, names=columns)

# Codificação One-Hot para as características
encoder = OneHotEncoder()
X = encoder.fit_transform(data.iloc[:, :-1]).toarray()

# Codificação One-Hot para a variável alvo
y = pd.get_dummies(data['class']).values
class_labels = list(pd.get_dummies(data['class']).columns)  # Lista de rótulos de classe

# Divisão inicial em conjunto de treino e conjunto temporário (que será dividido em teste e validação)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Divisão do conjunto temporário em conjunto de teste e validação (10% cada)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convertendo os dados para tensores do PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Criando DataLoaders para treino, validação e teste
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Definindo o modelo
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = X_train.shape[1]
hidden_size1 = 64
hidden_size2 = 32
output_size = y_train.shape[1]

model = SimpleNN(input_size, hidden_size1, hidden_size2, output_size)

# Definindo o critério de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Armazenar as perdas e acurácias para cada época
train_losses = []
val_losses = []
test_losses = []
test_accuracies = []  # Lista para armazenar as acurácias de teste de cada época

# Treinamento do modelo
num_epochs = 100
for epoch in range(num_epochs):
    # Modo de treino
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, torch.max(y_batch, 1)[1])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Avaliação no conjunto de validação
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, torch.max(y_batch, 1)[1])
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Avaliação no conjunto de teste
    test_loss = 0
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, torch.max(y_batch, 1)[1])
            test_loss += loss.item()
            test_preds.append(outputs.argmax(dim=1).cpu().numpy())
            test_targets.append(torch.max(y_batch, 1)[1].cpu().numpy())

    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    # Avaliando a acurácia no conjunto de teste
    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    test_accuracy = accuracy_score(test_targets, test_preds)
    test_accuracies.append(test_accuracy)  # Armazena a acurácia de cada época

    print(f'Época [{epoch+1}/{num_epochs}], Perda de Treinamento: {train_loss:.4f}, '
          f'Perda de Validação: {val_loss:.4f}, Perda de Teste: {test_loss:.4f}, '
          f'Acurácia de Teste: {test_accuracy:.4f}')

# Calcular a média das acurácias
mean_accuracy = np.mean(test_accuracies)
print(f'\nAcurácia Média no Conjunto de Teste: {mean_accuracy:.4f}')

# Plotar as curvas de perda
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Perda de Treinamento')
plt.plot(val_losses, label='Perda de Validação')
plt.plot(test_losses, label='Perda de Teste', linestyle='--', color='red')
plt.xlabel('Épocas')
plt.ylabel('Perda (Loss)')
plt.title('Curva de Treinamento')
plt.legend()
plt.savefig('comparacao_tempos_execucao_mlp.png')

# Calcular e plotar a matriz de confusão
conf_matrix = confusion_matrix(test_targets, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=class_labels, yticklabels=class_labels)  # Adiciona rótulos das classes
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão ')
plt.savefig('Matriz_de_confusao_mlp')