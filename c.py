import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Defina o caminho para a pasta onde os arquivos estão localizados
data_dir = 'car_evaluation'

# Carregar o dataset
data_path = os.path.join(data_dir, 'car.data')
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data = pd.read_csv(data_path, names=columns)

# Visualizar as primeiras linhas do dataset para conferir se carregou corretamente
print(data.head())

# Codificação One-Hot para as características
encoder = OneHotEncoder()
X = encoder.fit_transform(data.iloc[:, :-1]).toarray()

# Codificação One-Hot para a variável alvo
y = pd.get_dummies(data['class']).values

# Divisão em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construção do modelo RNA
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

# Compilação do modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Avaliação do modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Acurácia no conjunto de teste: {accuracy*100:.2f}%')
