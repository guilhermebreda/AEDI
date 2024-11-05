import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuração da página do Streamlit
st.title("Simulação de Monte Carlo para Valor Futuro de Imóveis")
st.write("""
Este aplicativo usa Simulação de Monte Carlo para projetar o valor futuro de um imóvel com base em 
taxas anuais de valorização, inflação e juros. Ajuste os parâmetros e veja a distribuição dos possíveis 
valores futuros ao final de 10 anos.
""")

# Parâmetros iniciais
valor_inicial = st.sidebar.number_input("Valor Inicial do Imóvel (R$)", min_value=100000, max_value=1000000, value=500000, step=50000)
anos = st.sidebar.slider("Horizonte de Tempo (anos)", min_value=5, max_value=30, value=10)
simulacoes = st.sidebar.number_input("Número de Simulações", min_value=1000, max_value=50000, value=10000, step=1000)

# Parâmetros das distribuições
st.sidebar.subheader("Parâmetros das Taxas Anuais")
media_valorizacao = st.sidebar.slider("Média da Taxa de Valorização (%)", min_value=0.0, max_value=10.0, value=3.0)
desvio_valorizacao = st.sidebar.slider("Desvio da Taxa de Valorização (%)", min_value=0.0, max_value=5.0, value=1.5)
media_inflacao = st.sidebar.slider("Média da Taxa de Inflação (%)", min_value=0.0, max_value=10.0, value=4.0)
desvio_inflacao = st.sidebar.slider("Desvio da Taxa de Inflação (%)", min_value=0.0, max_value=5.0, value=1.0)
media_juros = st.sidebar.slider("Média da Taxa de Juros (%)", min_value=0.0, max_value=15.0, value=6.0)
desvio_juros = st.sidebar.slider("Desvio da Taxa de Juros (%)", min_value=0.0, max_value=5.0, value=2.0)

# Convertendo porcentagens para decimais
media_valorizacao /= 100
desvio_valorizacao /= 100
media_inflacao /= 100
desvio_inflacao /= 100
media_juros /= 100
desvio_juros /= 100

# Função de Simulação de Monte Carlo
def simular_valor_imovel(valor_inicial, anos, simulacoes, media_valorizacao, desvio_valorizacao, media_inflacao, desvio_inflacao, media_juros, desvio_juros):
    valores_futuros = []
    for _ in range(simulacoes):
        valor = valor_inicial
        for _ in range(anos):
            taxa_valorizacao = np.random.normal(media_valorizacao, desvio_valorizacao)
            taxa_inflacao = np.random.normal(media_inflacao, desvio_inflacao)
            taxa_juros = np.random.normal(media_juros, desvio_juros)
            valor *= (1 + taxa_valorizacao + taxa_inflacao - taxa_juros * 0.5)
        valores_futuros.append(valor)
    return valores_futuros

# Executando a Simulação
valores_futuros = simular_valor_imovel(
    valor_inicial, anos, simulacoes, media_valorizacao, desvio_valorizacao, media_inflacao, desvio_inflacao, media_juros, desvio_juros
)

# Análise dos Resultados
valores_futuros = np.array(valores_futuros)
media_futuro = np.mean(valores_futuros)
mediana_futuro = np.median(valores_futuros)
intervalo_95 = np.percentile(valores_futuros, [2.5, 97.5])

# Exibindo os resultados
st.subheader("Estatísticas da Simulação")
st.write(f"**Valor médio projetado após {anos} anos:** R$ {media_futuro:,.2f}")
st.write(f"**Mediana do valor projetado:** R$ {mediana_futuro:,.2f}")
st.write(f"**Intervalo de confiança de 95%:** R$ {intervalo_95[0]:,.2f} a R$ {intervalo_95[1]:,.2f}")

# Visualização dos Resultados
st.subheader("Distribuição do Valor Futuro do Imóvel")
fig, ax = plt.subplots()
ax.hist(valores_futuros, bins=50, color='skyblue', edgecolor='black', density=True)
ax.axvline(media_futuro, color='red', linestyle='dashed', linewidth=1.5, label=f'Média: R$ {media_futuro:,.2f}')
ax.axvline(mediana_futuro, color='green', linestyle='dashed', linewidth=1.5, label=f'Mediana: R$ {mediana_futuro:,.2f}')
ax.set_title("Distribuição do Valor Futuro do Imóvel")
ax.set_xlabel("Valor Futuro (R$)")
ax.set_ylabel("Densidade de Probabilidade")
ax.legend()
st.pyplot(fig)
