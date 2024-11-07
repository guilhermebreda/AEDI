# Importando bibliotecas streamlit, numpy, pandas e plotly
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(page_title="Simulação Mensal de Flutuação de Aluguéis", layout="wide")

# Título e Descrição
st.title("Simulação de Monte Carlo Mensal para Flutuação de Aluguéis")
st.write("""
Esta simulação projeta a variação no valor do aluguel mensalmente ao longo do tempo, considerando fatores como 
inflação e demanda do mercado. A simulação usa Monte Carlo para gerar múltiplas trajetórias de evolução dos 
valores de aluguel.
""")

# Parâmetros de Entrada
valor_aluguel_inicial = st.sidebar.number_input("Valor Inicial do Aluguel (R$)", min_value=500, max_value=10000, value=2000, step=100)
meses = st.sidebar.slider("Período de Projeção (meses)", min_value=12, max_value=360, value=60)
simulacoes = st.sidebar.number_input("Número de Simulações", min_value=100, max_value=10000, value=1000, step=100)

# Parâmetros das Distribuições de Inflação e Demanda
st.sidebar.subheader("Parâmetros das Taxas Anuais")
media_inflacao_anual = st.sidebar.slider("Média da Inflação Anual (%)", min_value=0.0, max_value=10.0, value=3.0)
desvio_inflacao_anual = st.sidebar.slider("Desvio Padrão da Inflação (%)", min_value=0.0, max_value=5.0, value=1.0)
media_demanda_anual = st.sidebar.slider("Média da Demanda Anual (%)", min_value=-5.0, max_value=10.0, value=1.0)
desvio_demanda_anual = st.sidebar.slider("Desvio Padrão da Demanda (%)", min_value=0.0, max_value=5.0, value=1.5)

# Convertendo taxas anuais para mensais
media_inflacao_mensal = (1 + media_inflacao_anual / 100) ** (1 / 12) - 1
desvio_inflacao_mensal = desvio_inflacao_anual / 100 / np.sqrt(12)
media_demanda_mensal = (1 + media_demanda_anual / 100) ** (1 / 12) - 1
desvio_demanda_mensal = desvio_demanda_anual / 100 / np.sqrt(12)

# Função de Simulação de Monte Carlo Mensal
def simular_aluguel_mensal(valor_inicial, meses, simulacoes, media_inflacao_mensal, desvio_inflacao_mensal, media_demanda_mensal, desvio_demanda_mensal):
    trajetorias = np.zeros((simulacoes, meses))

    for i in range(simulacoes):
        valor = valor_inicial
        for mes in range(meses):
            # Gerar uma taxa de inflação aleatória para o mês
            taxa_inflacao = np.random.normal(media_inflacao_mensal, desvio_inflacao_mensal)
            # Gerar uma taxa de demanda aleatória para o mês
            taxa_demanda = np.random.normal(media_demanda_mensal, desvio_demanda_mensal)
            # Aplicar essas taxas ao valor do aluguel para o mês atual
            valor *= (1 + taxa_inflacao + taxa_demanda)
            trajetorias[i, mes] = valor
    
    return trajetorias

# Executando a Simulação
trajetorias = simular_aluguel_mensal(
    valor_aluguel_inicial, meses, simulacoes, media_inflacao_mensal, desvio_inflacao_mensal, media_demanda_mensal, desvio_demanda_mensal
)

# Cálculo da média e intervalo de confiança
media_mensal = trajetorias.mean(axis=0)
intervalo_95_mensal = np.percentile(trajetorias, [2.5, 97.5], axis=0)

# Exibindo os Resultados
st.subheader("Evolução Mensal do Valor de Aluguel - Média e Intervalo de Confiança")
st.write(f"**Período Simulado:** {meses} meses")
st.write(f"**Número de Simulações:** {simulacoes}")

# Visualização dos Resultados
fig = go.Figure()

# Adiciona a média dos valores de aluguel para cada mês
fig.add_trace(go.Scatter(
    x=np.arange(1, meses + 1),
    y=media_mensal,
    mode='lines',
    name='Média',
    line=dict(color='blue')
))

# Adiciona a faixa de intervalo de confiança de 95%
fig.add_trace(go.Scatter(
    x=np.concatenate([np.arange(1, meses + 1), np.arange(1, meses + 1)[::-1]]),
    y=np.concatenate([intervalo_95_mensal[0], intervalo_95_mensal[1][::-1]]),
    fill='toself',
    fillcolor='rgba(173, 216, 230, 0.3)',  # Azul claro com transparência
    line=dict(color='rgba(255,255,255,0)'),
    name='Intervalo de 95%'
))

fig.update_layout(
    title="Evolução Mensal do Valor de Aluguel com Intervalo de Confiança de 95%",
    xaxis_title="Mês",
    yaxis_title="Valor do Aluguel (R$)",
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)