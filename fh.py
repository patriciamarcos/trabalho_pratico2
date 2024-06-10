import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

#Recolha de dados
dados1 = pd.read_csv('California_Houses.csv')
dados2 = pd.read_csv('Melbourne_housings.csv', low_memory=False)

#Integração de dados
ano_atual =datetime.now().year
dados2['YearBuilt'] = ano_atual - dados2['YearBuilt']
# Combinacao de dados
df_combinado = pd.concat([dados1, dados2], axis=0)


# Eliminar colunas desnecessárias
colunas_eliminar = ['Median_Income','Population','Households','Distance_to_LA','Distance_to_SanDiego','Distance_to_SanJose','Distance_to_SanFrancisco','Suburb','Type','Method','Postcode','SellerG','Date','Car','CouncilArea','Regionname','Propertycount','ParkingArea','BuildingArea']
df_combinado.drop(columns=colunas_eliminar, inplace=True)


#rename colunas identicas mas com nomes diferentes
df_combinado.rename(columns={'Median_House_Value': 'Price', 'Median_Age': 'Age', 'Tot_Rooms': 'Rooms','Tot_Bedrooms': 'Bedroom', 'Distance_to_coast': 'Distance_coast', 'YearBuilt': 'Age', 'Longtitude': 'Longitude', 'Distance': 'Distance_coast'}, inplace=True)

# Agrupar e somar colunas duplicadas
df_combinado = df_combinado.groupby(df_combinado.columns, axis=1).sum()

# Exibir as primeiras linhas do DataFrame combinado
print(df_combinado.head())

# Preencher Colunas Vazias com a Mediana
for column in df_combinado.columns:
    if df_combinado[column].isnull().sum() > 0:
        median_value = df_combinado[column].median()
        df_combinado[column].fillna(median_value, inplace=True)

# Outliers
factor = 1.5
outliers_por_coluna = {}

for coluna in df_combinado.select_dtypes(include=np.number).columns:
    q1 = df_combinado[coluna].quantile(0.25)
    q3 = df_combinado[coluna].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    outliers = df_combinado[coluna][(df_combinado[coluna] < lower_bound) | (df_combinado[coluna] > upper_bound)]
    outliers_por_coluna[coluna] = outliers.count()

# Imprimir a quantidade de outliers por coluna
for coluna, qtd_outliers in outliers_por_coluna.items():
    print(f'Coluna "{coluna}": {qtd_outliers} outliers')

print(df_combinado.head())

# Salvar o DataFrame combinado em um arquivo CSV
df_combinado.to_csv('dados_integrados.csv', index=False)

# Normalização de Dados por StandardScaler
scaler = StandardScaler()
colunas_para_normalizar = df_combinado.columns.difference(['Address', 'Latitude', 'Price'])
df_combinado[colunas_para_normalizar] = scaler.fit_transform(df_combinado[colunas_para_normalizar])

# Salvar o objeto scaler treinado
joblib.dump(scaler, 'standard_scaler.pkl')

# Salvar as colunas usadas no scaler
with open('scaler_columns.txt', 'w') as f:
    for col in colunas_para_normalizar:
        f.write(col + '\n')

#mostrar graficos a normalização
colunas_para_plotar = df_combinado.columns.difference(['Address', 'Latitude'])
for coluna in colunas_para_plotar:
    plt.hist(df_combinado[coluna])
    plt.xlabel('Valores Normalizados')
    plt.ylabel('Frequência')
    plt.title(f'Distribuição dos Valores Normalizados para a coluna {coluna}')
    plt.show()
