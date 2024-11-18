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


