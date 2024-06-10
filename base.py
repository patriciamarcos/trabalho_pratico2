import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from sklearn.exceptions import ConvergenceWarning
import joblib

# Suprimindo avisos desnecessários
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Carregar os dados (substitua o caminho pelo seu dataset)
df = pd.read_csv('dados_integrados.csv')
df.drop(columns=['Address', 'Latitude'], inplace=True)
df.dropna(subset=['Price'], inplace=True)
X = df.drop(columns=['Price'])
y = df['Price']

# Carregar o scaler treinado
scaler = joblib.load('standard_scaler.pkl')

# Carregar as colunas usadas no scaler
with open('scaler_columns.txt', 'r') as f:
    scaler_columns = [line.strip() for line in f]

# Verificar se as colunas batem
print(f'Colunas usadas no scaler: {scaler_columns}')
print(f'Colunas presentes nos dados: {list(X.columns)}')

# Ajustar a ordem das colunas de X para bater com as do scaler
X = X[scaler_columns]

# Aplicar a normalização aos dados
X_scaled = scaler.transform(X)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Criar o pipeline
pipeline = Pipeline([
    ('mlp', MLPRegressor(max_iter=2000))  # Aumentar o número de iterações para melhorar a convergência
])

# Definir o grid de hiperparâmetros simplificado
param_grid = {
    'mlp__hidden_layer_sizes': [(50,), (100,)],  # Simplificar o número de combinações
    'mlp__activation': ['relu'],
    'mlp__solver': ['adam'],
    'mlp__alpha': [0.0001, 0.001],
    'mlp__learning_rate_init': [0.001, 0.0001]  # Adicionar taxas de aprendizado menores
}

# Configurar o GridSearchCV com KFold
kf = KFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=kf, verbose=2, n_jobs=-1)

# Treinar o modelo
grid_search.fit(X_train, y_train)

# Resultados do melhor modelo
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Prever no conjunto de teste
y_pred = grid_search.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')