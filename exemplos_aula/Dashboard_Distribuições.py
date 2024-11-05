#!/usr/bin/env python
# coding: utf-8

# ![MARCADOR.png](attachment:MARCADOR.png)

# # BI - Distribuições de Probabilidade

# ## Bibliotecas

# In[6]:


import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import binom
from scipy.stats import norm
import pandas as pd
from PIL import Image
import subprocess


# ### Dashboard

# In[14]:


# Configuração da página
st.set_page_config(page_title="Análise de Distribuições de Probabilidade",
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
    st.markdown("<h1 style='text-align: center; color: #003366;'>Análise de Distribuições de Probabilidade</h1>", 
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #003366;'>Professor João Gabriel de Moraes Souza</h3>", 
                unsafe_allow_html=True)  # Inserir nome do professor
with col3:
    st.image(logo_unb, use_column_width=True)

st.markdown("---")

# Criação de abas para diferentes distribuições
tab1, tab2, tab3 = st.tabs(["Distribuição Binomial", "Distribuição Poisson", "Distribuição Normal"])

# Aba da Distribuição Binomial
with tab1:
    st.header("Distribuição Binomial")
    st.markdown("### Simulação de Overbooking")

    # Títulos dos sliders em azul
    st.markdown("<h4 style='color: #003366;'>Probabilidade de Comparecimento (p)</h4>", unsafe_allow_html=True)
    p = st.slider("", min_value=0.8, max_value=1.00, value=0.88, step=0.01)  
    st.markdown("<h4 style='color: #003366;'>Número de Assentos Vendidos</h4>", unsafe_allow_html=True)
    seats_sold = st.slider("", min_value=451, max_value=500, value=461, step=1)

    st.markdown("<h4 style='color: #003366;'>Nível de Risco Aceito (%)</h4>", unsafe_allow_html=True)
    risk_level = st.slider("", min_value=0.01, max_value=0.50, value=0.05, step=0.01) 
    
    # Calcular probabilidade
    probability = 1 - binom.cdf(450, seats_sold, p)

    # Gráfico dinâmico usando Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.arange(451, seats_sold + 1), 
                             y=1 - binom.cdf(450, np.arange(451, seats_sold + 1), p),
                             mode='lines', line=dict(color='#003366', width=3)))

    fig.add_hline(y=risk_level, line_dash="dash", line_color="red", line_width=1)  # Dinamicamente ajustado

    fig.update_layout(title="Risco de Overbooking para mais de 450 passageiros",
                      xaxis_title="Assentos Vendidos",
                      yaxis_title="Probabilidade de mais de 450 passageiros aparecerem",
                      xaxis=dict(tickmode='linear', tick0=451, dtick=1),
                      yaxis=dict(range=[0, 1]),
                      plot_bgcolor="white",
                      width=800,
                      height=400)

    st.plotly_chart(fig, use_container_width=True)

    # Gerar tabela de dados com base nos valores selecionados
    seats_sold_range = np.arange(451, seats_sold + 1)
    probabilities = (1 - binom.cdf(450, seats_sold_range, p)) 
    table = pd.DataFrame({'Assentos Vendidos a mais': seats_sold_range, 'Risco de Overbooking (%)': probabilities})

    # Exibir a tabela filtrada no Streamlit
    st.write("### Tabela de Probabilidades para Assentos Vendidos Selecionados")
    st.dataframe(table)  # Exibir a tabela filtrada

    # Determinar o número máximo de assentos vendidos até atingir o nível de risco aceito
    max_seats_within_risk = table[table['Risco de Overbooking (%)'] <= risk_level]['Assentos Vendidos a mais'].max()
    if pd.notna(max_seats_within_risk):
        st.write(f"O número máximo de assentos que podem ser vendidos sem exceder o nível de risco aceito ({round(risk_level, 2)*100}%) é de {int(max_seats_within_risk)} assentos.")
    else:
        st.write(f"Nenhum número de assentos vendidos está abaixo do nível de risco aceito ({round(risk_level, 2)*100}%).")

# Aba da Distribuição Poisson (vazia por enquanto)
with tab2:
    st.header("Distribuição Poisson")
    st.markdown("### Simulação de Número de clientes que chegam em uma loja por dia")
    
    # Entrada do usuário
    st.markdown("<h4 style='color: #003366;'>Taxa média de chegada de clientes (λ)</h4>", unsafe_allow_html=True)
    lambda_value = st.slider(" ", min_value=1, max_value=20, value=5, step=1)
    st.markdown("<h4 style='color: #003366;'>Número de horas simuladas</h4>", unsafe_allow_html=True)
    num_hours = st.slider(" ", min_value=1, max_value=48, value=24, step=1)
    st.markdown("<h4 style='color: #003366;'>Número de simulações Monte Carlo</h4>", unsafe_allow_html=True)
    num_simulations = st.slider(" ", min_value=100, max_value=10000, value=1000, step=100)

    # Simulação Monte Carlo
    np.random.seed(42)  # Para reprodutibilidade
    simulated_data = np.random.poisson(lam=lambda_value, size=(num_simulations, num_hours))

    # Calcular a média de clientes por hora em todas as simulações
    average_customers_per_hour = np.round(np.mean(simulated_data, axis=0), 2)

    # Preparar dados para o Plotly Express
    hourly_data = {
        'Hora do Dia': np.arange(1, num_hours + 1),
        'Número Médio de Clientes': average_customers_per_hour
    }
    df = pd.DataFrame(hourly_data)

    # Gráfico interativo usando Plotly Express
    fig = px.line(df, 
              x='Hora do Dia', 
              y='Número Médio de Clientes', 
              title=(f"Simulação Monte Carlo: Chegada de Clientes por Hora \n"
                     f"(λ = {lambda_value}, {num_hours} horas, {num_simulations} simulações)"),
              labels={'Hora do Dia': 'Hora do Dia', 'Número Médio de Clientes': 'Número Médio de Clientes'},
              markers=True)

    # Mostrar gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Criar um DataFrame para os resultados com labels
    hourly_results = pd.DataFrame({
        'Hora do Dia': [f"Hora {i+1}" for i in range(min(24, num_hours))],
        'Número Médio de Clientes': average_customers_per_hour[:min(24, num_hours)]
    })
    st.write("### Tabela de Quantidade de Clientes Estimada por Dia")
    st.table(hourly_results)

# Aba da Distribuição Normal (vazia por enquanto)
with tab3:
    st.header("Distribuição Normal")
    st.markdown("### Simulação de Número de Caixas Vendidas de Medicamentos")
        # Função para calcular a área entre lb e ub
    def calcular_area(mean, sd, lb, ub):
        area = norm.cdf(ub, mean, sd) - norm.cdf(lb, mean, sd)
        return area

    # Parâmetros ajustáveis
    mean = st.slider("Média (mean)", 50, 150, 100)
    sd = st.slider("Desvio Padrão (sd)", 5, 30, 15)
    lb = st.slider("Limite Inferior (lb)", 50, 100, 80)
    ub = st.slider("Limite Superior (ub)", 100, 150, 120)

    # Cálculo da probabilidade
    area = calcular_area(mean, sd, lb, ub)
    result = round(area * 100, 2)

    # Mensagem do resultado
    st.write(f"P({lb} <= Nº Caixas <= {ub}) = {result}%")

    # Gerar os dados para a distribuição normal
    x = np.linspace(40, 160, 500)
    hx = norm.pdf(x, mean, sd)

    # Gráfico com Plotly
    fig = go.Figure()

    # Adiciona a curva da distribuição normal
    fig.add_trace(go.Scatter(
        x=x, y=hx,
        mode='lines',
        name='Distribuição Normal',
        line=dict(color='black')
    ))

    # Preencher a área entre os limites inferior e superior
    x_fill = np.linspace(lb, ub, 100)
    hx_fill = norm.pdf(x_fill, mean, sd)

    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fill, x_fill[::-1]]),
        y=np.concatenate([hx_fill, np.zeros_like(hx_fill)]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.5)',
        line=dict(color='red'),
        hoverinfo='skip',
        name='Área'
    ))

    # Customizar o layout do gráfico
    fig.update_layout(
        title=f"Distribuição Normal com média={mean} e desvio padrão={sd}",
        xaxis_title="Número de Caixas",
        yaxis_title="Densidade",
        xaxis=dict(range=[40, 160]),
        yaxis=dict(range=[0, 0.03]),
        showlegend=False
    )

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig)


# In[15]:


# Converte o notebook 'Dashboard_Distribuições.ipynb' para um script Python
subprocess.run(["jupyter", "nbconvert", "--to", "script", "Dashboard_Distribuições.ipynb"])

