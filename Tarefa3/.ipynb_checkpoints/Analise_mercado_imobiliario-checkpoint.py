# Considerar nº de quartos, tipo de bairro e condição da casa
# nº de quartos = 'Bedroom AbvGr',
# tipo de zonameanto/classificação zoneamento da casa = 'MS Zoning'
# condição da casa = 'Overall Cond'

# Dataset
# 2.930 propriedades em Ames, Iowa
# https://www.kaggle.com/datasets/prevek18/ames-housing-dataset

# streamlit
# https://9vjsmasmhinv6hxbvrrcvb.streamlit.app

# github
# https://github.com/guilhermebreda/AEDI/blob/main/Tarefa3/Analise_mercado_imobiliario.py


import pandas as pd
from scipy.stats import f_oneway
import seaborn as sns
import matplotlib.pyplot as plt
import os

print("Current working directory:", os.getcwd())

file_path = 'Tarefa3/AmesHousing.csv'
housing_data = pd.read_csv(file_path)

print(housing_data.head())
print(housing_data.columns)
print()

# Visualizar as colunas relevantes
print(housing_data[['Bedroom AbvGr', 'MS Zoning', 'Overall Cond', 'SalePrice']].head())


# Verificar dados ausentes
print("Dados ausentes por coluna:")
print(housing_data.isnull().sum())

# Visualizar estatísticas descritivas
print("\nResumo estatístico:")
print(housing_data[['Bedroom AbvGr', 'Overall Cond', 'SalePrice']].describe())

# Garantir que não haja valores ausentes nas colunas analisadas
filtered_data = housing_data[['Bedroom AbvGr', 'MS Zoning', 'Overall Cond', 'SalePrice']].dropna()

# **1. Distribuição do Preço de Venda (SalePrice)**
plt.figure(figsize=(10, 6))
sns.histplot(filtered_data['SalePrice'], kde=True, color='blue', bins=30)
plt.title('Distribuição do Preço de Venda')
plt.xlabel('Preço de Venda')
plt.ylabel('Frequência')
plt.show()

# **2. Distribuição do Número de Quartos (Bedroom AbvGr)**
plt.figure(figsize=(10, 6))
sns.histplot(filtered_data['Bedroom AbvGr'], kde=False, color='green', bins=15)
plt.title('Distribuição do Número de Quartos Acima do Solo')
plt.xlabel('Número de Quartos')
plt.ylabel('Frequência')
plt.show()

# **3. Distribuição da Condição Geral (Overall Cond)**
plt.figure(figsize=(10, 6))
sns.countplot(x='Overall Cond', data=filtered_data, palette='Set2')
plt.title('Distribuição da Condição Geral do Imóvel')
plt.xlabel('Condição Geral')
plt.ylabel('Frequência')
plt.show()

# **4. Preço de Venda por Tipo de Zoneamento**
plt.figure(figsize=(12, 6))
sns.boxplot(x='MS Zoning', y='SalePrice', data=filtered_data, palette='Set3')
plt.title('Preço de Venda por Tipo de Zoneamento')
plt.xlabel('Tipo de Zoneamento')
plt.ylabel('Preço de Venda')
plt.xticks(rotation=45)
plt.show()

# **5. Correlações com SalePrice**
corr_data = filtered_data[['SalePrice', 'Bedroom AbvGr', 'Overall Cond']].corr()
print("\nCorrelação com Preço de Venda:")
print(corr_data)

# Heatmap das correlações
plt.figure(figsize=(8, 6))
sns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlação entre Variáveis Selecionadas')
plt.show()



# Garantir que não haja valores ausentes nas colunas analisadas
filtered_data = housing_data[['Bedroom AbvGr', 'MS Zoning', 'Overall Cond', 'SalePrice']].dropna()

# Função para realizar ANOVA e visualização
def anova_analysis(df, feature, target='SalePrice'):
    groups = df.groupby(feature)[target]
    anova_data = [group for _, group in groups]
    f_stat, p_value = f_oneway(*anova_data)
    
    # Exibir os resultados da ANOVA
    print(f"\nANOVA para {feature}")
    print(f"F-Statistic: {f_stat:.2f}")
    print(f"P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Resultado: Diferenças significativas entre os grupos.")
    else:
        print("Resultado: Não há diferenças significativas entre os grupos.")
    
    # Boxplot para visualizar os dados
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=feature, y=target, data=df)
    plt.title(f"Distribuição de {target} por {feature}")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.xticks(rotation=45)
    plt.show()

# Realizar ANOVA para cada característica
anova_analysis(filtered_data, 'Bedroom AbvGr')  # Número de quartos
anova_analysis(filtered_data, 'MS Zoning')      # Tipo de zoneamento
anova_analysis(filtered_data, 'Overall Cond')   # Condição geral


