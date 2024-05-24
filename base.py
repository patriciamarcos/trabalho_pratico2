import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Carregar os dados do arquivo CSV (substitua 'dados_integrados.csv' pelo nome do seu arquivo)
data = pd.read_csv('dados_integrados.csv')

# Excluir a coluna que você não quer usar
data.drop(columns=['Address'], inplace=True)  # Substitua 'Address' pelo nome da coluna que deseja excluir

# Remover linhas com valores ausentes na variável alvo (Price)
data.dropna(subset=['Price'], inplace=True)

# Separar as características (features) dos valores alvo (target)
features = data.drop(columns=['Price'])  # Supondo que 'Price' seja a coluna que contém o preço do imóvel
target = data['Price']

# Criar um objeto SimpleImputer
imputer = SimpleImputer(strategy='mean')

# Aplicar a imputação nos dados
features_imputed = imputer.fit_transform(features)

# Verificar se há valores ausentes nas características após a imputação
if np.isnan(features_imputed).any():
    raise ValueError("Existem valores ausentes nas características após a imputação. Verifique seus dados.")

# Verificar se há valores ausentes na variável alvo
if target.isnull().any():
    raise ValueError("Existem valores ausentes na variável alvo. Verifique seus dados.")

# Criar uma instância do modelo de regressão linear
model = LinearRegression()

# Abordagem K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kfold = -cross_val_score(model, features_imputed, target, cv=kfold, scoring='neg_mean_squared_error')
mse_kfold = scores_kfold.mean()

# Abordagem Leave-One-Out Cross-Validation (LOOCV)
loo = LeaveOneOut()
scores_loo = -cross_val_score(model, features_imputed, target, cv=loo, scoring='neg_mean_squared_error')
mse_loo = scores_loo.mean()

# Exibindo as médias dos erros quadráticos médios (MSE)
print("MSE médio (K-Fold):", mse_kfold)
print("MSE médio (LOOCV):", mse_loo)