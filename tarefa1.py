import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

# Configurações de Título e Descrição
st.title("Simulação de Vida Útil de Dispositivos Eletrônicos")
st.write("""
Este dashboard usa a técnica de Simulação de Monte Carlo com a distribuição Weibull para estimar a vida útil de dispositivos eletrônicos. 
Ajuste os parâmetros da distribuição para ver como eles afetam a previsão de falha.
""")

# Parâmetros da Distribuição Weibull
st.sidebar.header("Parâmetros da Distribuição Weibull")
escala = st.sidebar.slider("Parâmetro de Escala (λ)", min_value=0.5, max_value=5.0, value=3.0, step=0.1)
forma = st.sidebar.slider("Parâmetro de Forma (β)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
n_simulacoes = st.sidebar.number_input("Número de Simulações", min_value=1000, max_value=100000, value=10000, step=1000)

# Simulação de Monte Carlo
def simular_vida_util(escala, forma, n):
    # Gerando tempos de vida com distribuição Weibull
    return weibull_min.rvs(c=forma, scale=escala, size=n)

# Executando a Simulação
tempos_falha = simular_vida_util(escala, forma, n_simulacoes)

# Exibindo Estatísticas
st.subheader("Estatísticas da Simulação")
st.write(f"Vida útil média estimada: {np.mean(tempos_falha):.2f} anos")
st.write(f"Mediana da vida útil: {np.median(tempos_falha):.2f} anos")
st.write(f"Desvio padrão: {np.std(tempos_falha):.2f} anos")

# Probabilidade de Falha Antes de um Ano Específico
ano_critico = st.slider("Ano Crítico para Falha", min_value=1, max_value=10, value=2)
prob_falha_antes = np.mean(tempos_falha < ano_critico)
st.write(f"Probabilidade de falha antes de {ano_critico} anos: {prob_falha_antes * 100:.2f}%")

# Histogramas e Gráficos
st.subheader("Distribuição da Vida Útil")
fig, ax = plt.subplots()
ax.hist(tempos_falha, bins=30, density=True, alpha=0.6, color="skyblue")

# Plotando a função de densidade teórica Weibull
x = np.linspace(0, max(tempos_falha), 100)
ax.plot(x, weibull_min.pdf(x, c=forma, scale=escala), 'r-', lw=2, label="Densidade Weibull Teórica")
ax.set_xlabel("Tempo até Falha (anos)")
ax.set_ylabel("Densidade de Probabilidade")
ax.legend()
st.pyplot(fig)

# Exibindo a Distribuição Acumulada
st.subheader("Função de Distribuição Acumulada")
fig_cdf, ax_cdf = plt.subplots()
ax_cdf.hist(tempos_falha, bins=30, density=True, cumulative=True, alpha=0.6, color="purple")
ax_cdf.set_xlabel("Tempo até Falha (anos)")
ax_cdf.set_ylabel("Probabilidade Acumulada")
st.pyplot(fig_cdf)
