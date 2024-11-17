import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from scipy.stats import norm
#import seaborn as sns
import plotly.graph_objects as go
#from PIL import Image



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

# Função para calcular o VaR
def calcular_var(returns, confidence_level):
    returns = np.array(returns)
    z_score = norm.ppf(confidence_level)
    stdev = np.std(returns)
    var = -(returns.mean() + z_score * stdev)
    return var

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


#st.markdown("<h1 style='text-align: center;'>Teoria do Portfólio Eficiente</h1>", 
#            unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Análise de Carteira de Ações</h1>", 
            unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>Aluno: Guilherme Breda Rezende</h3>", 
            unsafe_allow_html=True)  # Inserir nome aluno

st.markdown("<h3 style='text-align: center;'>Professor João Gabriel de Moraes Souza</h3>", 
            unsafe_allow_html=True)  # Inserir nome do professor # color: #003366

st.markdown("---")

tickers = st.multiselect('Selecione as ações:', ['ELET3.SA', 'PETR3.SA', 'TOTS3.SA', 'VALE3.SA', 
                                                 'WEGE3.SA', 'BOVA11.SA'], 
                         default=['ELET3.SA', 'PETR3.SA', 'TOTS3.SA', 'VALE3.SA', 'WEGE3.SA', 'BOVA11.SA'])




start_date = st.date_input('Data de Início', value=pd.to_datetime('2014-11-11'))
end_date = st.date_input('Data de Fim', value=pd.to_datetime('2024-11-11')) #aaaa-mm-dd

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
                         [100, 1000, 10000, 50000, 100000], index=3)
valor_inicial = st.number_input('Valor Inicial de Investimento:', min_value=1000, 
                                max_value=100000, value=35000, step=1000)    

global max_sr_ret, max_sr_vol  # Definindo variáveis globais    
    
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

    # Após a simulação de portfólio e identificação dos melhores pesos
if st.checkbox('Mostrar Pesos do Melhor Portfólio'):
    # Organizando os pesos em um DataFrame para melhor visualização
    ativos = data.columns  # Pegando os nomes dos ativos selecionados
    pesos = all_weights[max_sr_loc]  # Melhores pesos do portfólio
    pesos_percentuais = pesos * 100  # Convertendo para porcentagem

    # Criando um DataFrame com os ativos e seus pesos
    df_pesos = pd.DataFrame({
        'Ativo': ativos,
        'Peso (%)': pesos_percentuais
    })

    # Exibindo o DataFrame com st.table para uma visualização estática
    st.write("### Pesos do Melhor Portfólio")
    st.dataframe(df_pesos.style.format({'Peso (%)': '{:.2f}%'}))

# Carregando os dados e executando a função de alocação de ativos
if st.button('Performance Portfólio'):
    data = load_data(tickers, start_date, end_date)
    dataset, patrimonio_df, soma_valor_final = alocacao_ativos(data, valor_inicial)

    # Plotando a evolução do patrimônio
    figura_valor_port = px.line(patrimonio_df, x='Date', y='Patrimônio', title='Evolução do Patrimônio da Carteira')
    st.plotly_chart(figura_valor_port)

    st.write(f'Valor final do portfólio: R$ {soma_valor_final:.2f}')

# Seção para cálculo e exibição do VaR
if 'max_sr_ret' in globals() and 'max_sr_vol' in globals():
    st.markdown("## Cálculo do Value at Risk (VaR)")

    # Configuração do nível de confiança para o cálculo do VaR
    confidence_level = st.slider('Selecione o Nível de Confiança para o VaR:', min_value=0.80, max_value=0.99, value=0.90)

    # Seletor para escolher entre retorno percentual e valores monetários
    modo_var = st.radio("Escolha a exibição do VaR:", ("Retorno Percentual", "Valores Monetários"))

    # Gerar lista de retornos para calcular o VaR (usando os retornos logarítmicos do portfólio simulado)
    returns = np.random.normal(max_sr_ret * 100, max_sr_vol * 100, 10000)

    # Calcular o VaR no nível de confiança especificado
    var_value = calcular_var(returns, confidence_level)

    # Ajuste para valores monetários se selecionado
    if modo_var == "Valores Monetários":
        var_value = valor_inicial * (var_value / 100)
        returns = returns * (valor_inicial / 100)
        y_label = "Densidade (Valores Monetários)"
        x_label = "Valores Monetários"
    else:
        y_label = "Densidade (Retorno Percentual)"
        x_label = "Retorno Percentual"

    st.write(f'VaR no intervalo de confiança de {confidence_level * 100:.0f}%: {var_value:.2f}')

    # Plotar a distribuição dos retornos com o nível de VaR usando Plotly
    fig = go.Figure()

    # Adicionando o histograma dos retornos
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        histnorm='probability',
        name='Distribuição de Retornos',
        marker=dict(color='skyblue'),
        opacity=0.75
    ))

    # Linha vertical para indicar o VaR
    fig.add_shape(
        type="line",
        x0=var_value,
        y0=0,
        x1=var_value,
        y1=0.1,
        line=dict(color="red", width=2, dash="dash"),
        name=f'VaR {confidence_level*100:.0f}%'
    )

    # Anotação para o VaR
    fig.add_annotation(
        x=var_value,
        y=0.1,
        text=f'VaR {confidence_level*100:.0f}%',
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40
    )

    # Configurações do layout
    fig.update_layout(
        title=f'Distribuição dos Retornos com VaR até {confidence_level * 100:.0f}% de Nível de Confiança',
        xaxis_title=x_label,
        yaxis_title=y_label,
        bargap=0.2,
    )

    st.plotly_chart(fig)
else:
    st.warning("Realize a simulação do portfólio para calcular o VaR.")


