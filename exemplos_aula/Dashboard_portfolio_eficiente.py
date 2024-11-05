#!/usr/bin/env python
# coding: utf-8

# ![MARCADOR.png](attachment:MARCADOR.png)

# # BI - Distribuições de Probabilidade

# ## Bibliotecas

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from PIL import Image
import subprocess


# ### Dashboard

# In[4]:


def load_data(tickers, start_date, end_date):
    data = pd.DataFrame()
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start=start_date, end=end_date)['Close']
    data.reset_index(inplace=True)
    return data

def get_ret_vol_sr(weights, log_ret):
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights * 252)
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sr = ret / vol
    return np.array([ret, vol, sr])

def check_sum(weights):
    return np.sum(weights) - 1

# Função ajustada de alocação de ativos
def alocacao_ativos(dataset, dinheiro_total, seed=0, melhores_pesos=[]):
    dataset = dataset.copy()

    if seed != 0:
        np.random.seed(seed)

    if len(melhores_pesos) > 0:
        pesos = melhores_pesos
    else:
        pesos = np.random.random(len(dataset.columns) - 1)
        pesos = pesos / np.sum(pesos)

    colunas = dataset.columns[1:]

    for i, acao in enumerate(colunas):
        dataset[acao] = (dataset[acao] / dataset[acao][0]) * pesos[i] * dinheiro_total

    dataset['soma valor'] = dataset[colunas].sum(axis=1)
    dataset['taxa retorno'] = dataset['soma valor'].pct_change() * 100

    # Criando um novo DataFrame para 'datas' e 'patrimônio'
    patrimonio_df = pd.DataFrame({'Date': dataset['Date'], 'Patrimônio': dataset['soma valor']})
    
    return dataset, patrimonio_df, dataset['soma valor'].iloc[-1]

# Configuração da página
st.set_page_config(page_title="Teoria do Portfólio Eficiente",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Slider CSS customization for green color
st.markdown(
    """
    <style>
    .stSlider > div > div > div > div > div > div {
        background-color: #4CAF50 !important;  /* Verde para o slider */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Carregar as logos
logo_unb = Image.open("Logo/MARCADOR.png")

# Título e Logos
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image(logo_unb, use_column_width=True)
with col2:
    st.markdown("<h1 style='text-align: center; color: #003366;'>Teoria do Portfólio Eficiente</h1>", 
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #003366;'>Professor João Gabriel de Moraes Souza</h3>", 
                unsafe_allow_html=True)  # Inserir nome do professor
with col3:
    st.image(logo_unb, use_column_width=True)

st.markdown("---")

tickers = st.multiselect('Selecione as ações:', ['ITUB3.SA', 'MGLU3.SA', 'CIEL3.SA', 'PETR3.SA', 
                                                 'CSNA3.SA', '^BVSP'], 
                         default=['ITUB3.SA', '^BVSP'])

start_date = st.date_input('Data de Início', value=pd.to_datetime('2012-01-01'))
end_date = st.date_input('Data de Fim', value=pd.to_datetime('2023-12-31'))

data = load_data(tickers, start_date, end_date)

if st.checkbox('Mostrar Gráfico de Preço'):
    st.subheader('Histórico do Preço das Ações')
    figura = px.line(data, x='Date', y=[data[ticker] for ticker in tickers], 
                     labels={'value': 'Preço', 'variable': 'Ações'})
    st.plotly_chart(figura)

if st.checkbox('Mostrar Histórico de Retorno das Ações'):
    st.subheader('Histórico de Retorno das Ações')
    data.set_index('Date', inplace=True)
    log_retornos = np.log(data / data.shift(1))
    figura_retornos = px.line(log_retornos, labels={'value': 'Retorno Logarítmico', 'variable': 'Ações'})
    st.plotly_chart(figura_retornos)

# Adicionando seletores para o número de portfólios e valor inicial de investimento
num_ports = st.selectbox('Selecione o número de portfólios para a simulação:', 
                         [100, 1000, 10000], index=2)
valor_inicial = st.number_input('Valor Inicial de Investimento:', min_value=1000, 
                                max_value=100000, value=10000, step=1000)    
    
    
if st.checkbox('Realizar Simulação de Portfólio'):
    st.subheader('Simulação de Portfólio')

    data.dropna(inplace=True)
    log_ret = np.log(data / data.shift(1))

    np.random.seed(42)
    all_weights = np.zeros((num_ports, len(data.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for x in range(num_ports):
        weights = np.array(np.random.random(len(data.columns)))
        weights = weights / np.sum(weights)
        
        all_weights[x, :] = weights
        ret_arr[x], vol_arr[x], sharpe_arr[x] = get_ret_vol_sr(weights, log_ret)
    
    max_sr_loc = sharpe_arr.argmax()
    max_sr_ret = ret_arr[max_sr_loc]
    max_sr_vol = vol_arr[max_sr_loc]

    figura_port = px.scatter(x=vol_arr, y=ret_arr, color=sharpe_arr, 
                             color_continuous_scale='viridis', 
                             labels={'x': 'Volatilidade', 'y': 'Retorno'}, title='Simulação de Portfólio')
    figura_port.add_scatter(x=[max_sr_vol], y=[max_sr_ret], marker=dict(color='red', size=20), 
                            name='Melhor Sharpe Ratio')
    st.plotly_chart(figura_port)

    st.write(f'Melhores Pesos: {all_weights[max_sr_loc]}')

    # Carregando os dados e executando a função de alocação de ativos
if st.button('Performance Portfólio'):
    data = load_data(tickers, start_date, end_date)
    dataset, patrimonio_df, soma_valor_final = alocacao_ativos(data, valor_inicial)

    # Plotando a evolução do patrimônio
    figura_valor_port = px.line(patrimonio_df, x='Date', y='Patrimônio', title='Evolução do Patrimônio da Carteira')
    st.plotly_chart(figura_valor_port)

    st.write(f'Valor final do portfólio: R$ {soma_valor_final:.2f}')


# In[ ]:


# Converte o notebook 'Dashboard_Distribuições.ipynb' para um script Python
subprocess.run(["jupyter", "nbconvert", "--to", "script", "Dashboard_Distribuições.ipynb"])

