import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve


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



#gráficos
# Calcular a matriz de correlação
correlation_matrix = df.corr()

# Plotar a matriz de correlação
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()


# Gerar as curvas de aprendizagem
train_sizes, train_scores, test_scores = learning_curve(
    grid_search.best_estimator_, X_train, y_train, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)

# Calcular as médias e desvios padrão dos scores
train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plotar as curvas de aprendizagem
plt.figure()
plt.title("Curvas de Aprendizagem")
plt.xlabel("Tamanho do Treinamento")
plt.ylabel("Erro Quadrático Médio")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Erro no Treinamento")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Erro na Validação")

plt.legend(loc="best")
plt.show()


# Gerar a curva de validação para um hiperparâmetro específico
param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10]
train_scores, test_scores = validation_curve(
    MLPRegressor(max_iter=2000), X_train, y_train, param_name="alpha",
    param_range=param_range, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)

# Calcular as médias e desvios padrão dos scores
train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plotar a curva de validação
plt.figure()
plt.title("Curva de Validação para Alpha")
plt.xlabel("Alpha")
plt.ylabel("Erro Quadrático Médio")
plt.grid()

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Erro no Treinamento")
plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Erro na Validação")

plt.legend(loc="best")
plt.xscale("log")
plt.show()


# Prever no conjunto de teste
y_pred = grid_search.predict(X_test)

# Plotar Previsões vs Valores Reais
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Valores Reais vs Valores Previstos')
plt.show()