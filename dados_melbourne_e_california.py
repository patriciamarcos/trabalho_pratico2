import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from datetime import datetime

#carregar os dados dois arquivos CSV
dados1 = pd.read_csv('C:\\Users\\patri\\Desktop\\ubi\\bsc_iacd\\2023_2024\\2_semestre\\elem IA\\trabalho_pratico1\\dados\\California_Houses.csv')
dados2 = pd.read_csv('C:\\Users\\patri\\Desktop\\ubi\\bsc_iacd\\2023_2024\\2_semestre\\elem IA\\trabalho_pratico1\\dados\\Melbourne_housings.csv', low_memory=False)

df1 = dados1.drop(columns={'Median_Income','Population','Households', 'Distance_to_LA', 'Distance_to_SanDiego', 'Distance_to_SanJose', 'Distance_to_SanFrancisco'})
df2 = dados2.drop(columns={'Suburb', 'Type', 'Method', 'Postcode', 'SellerG','Date', 'Car', 'CouncilArea', 'Regionname', 'Propertycount', 'ParkingArea', 'BuildingArea'})

df1 = df1.rename(columns={'Median_House_Value': 'Price', 'Median_Age': 'Age', 'Tot_Rooms': 'Rooms','Tot_Bedrooms': 'Bedroom', 'Distance_to_coast': 'Distance_coast'})
df2 = df2.rename(columns={'YearBuilt': 'Age', 'Longtitude': 'Longitude', 'Distance': 'Distance_coast'})

ano_atual = datetime.now().year
df2['Age'] = ano_atual - df2['Age']

print("Dados do ficheiro 1:")
print(df1.head())
print("\nDados do ficheiro 2:")
print(df2.head())

#Integração de Dados
df_combinado = pd.concat([df1, df2], axis=0)
print("\nConjunto de dados combinado:")
print(df_combinado.head())
df_combinado.to_csv('dados_integrados.csv', index=False)

#Análise dos dados
# Criar o gráfico de barras para valores em falta
valores_ausentes = df_combinado.isnull().sum()
print(f"Valores em falta em cada coluna:")
print(valores_ausentes)

plt.figure(figsize=(10, 6))
valores_ausentes.plot(kind='bar', color='skyblue')
plt.title('Quantidade de Valores em Falta por Coluna')
plt.xlabel('Coluna')
plt.ylabel('Quantidade de Valores Ausentes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Detetar outliers
def calcular_quantidade_outliers(df_combinado, coluna_a_manter):
    colunas_para_calcular_outliers = df_combinado.columns.difference([coluna_a_manter])
    Q1 = df_combinado[colunas_para_calcular_outliers].quantile(0.25)
    Q3 = df_combinado[colunas_para_calcular_outliers].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = (df_combinado[colunas_para_calcular_outliers] < limite_inferior) | (df_combinado[colunas_para_calcular_outliers] > limite_superior)
    quantidade_outliers_por_coluna = outliers.sum()

    return quantidade_outliers_por_coluna

quantidade_outliers = calcular_quantidade_outliers(df_combinado, 'Address')
print("Quatidade de Outliers:")
print(quantidade_outliers)

colunas_para_plotar = df_combinado.columns.difference(['Address'])
for coluna in colunas_para_plotar:
    plt.figure()
    plt.scatter(df_combinado.index, df_combinado[coluna])
    plt.title(f'Gráfico de Dispersão para {coluna}')
    plt.xlabel('Índice das Linhas')
    plt.ylabel(coluna)
    plt.show()

#Correlações com a variável objetivo(price)
colunas_para_analise = df_combinado.columns.difference(['Address'])

for coluna in colunas_para_analise:
    correlacao = df_combinado[coluna].corr(df_combinado['Price'])
    plt.figure()
    plt.bar(coluna, correlacao)
    plt.title(f'Correlação com {coluna}')
    plt.xlabel('Price')
    plt.ylabel('Correlação')
    plt.ylim(-1, 1)
    plt.grid(axis='y')
    plt.show()