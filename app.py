
################################## IMPORT #######################################
import streamlit as st
from datetime import date
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go
import yfinance as yf
import numpy as np

# Carrega dados históricos do petróleo Brent
indice = "BZ=F"
inicio = "2000-01-01"
dados_acao = yf.download(indice, inicio)
df_cotacoes = pd.DataFrame({indice: dados_acao['Close']})
df_cotacoes.rename(columns={indice: 'Preço'}, inplace=True)

# Cria df para utilizar em visualizações
df_filtrado = df_cotacoes.copy()

# Cria páginas
pagina = st.sidebar.selectbox("Menu", ["Contexto", "Dados históricos + Insigths", "Modelo de previsão"])


################################## PÁGINA CONTEXTO #######################################


if pagina == "Contexto":
    st.title('Preço do petróleo Brent')
    st.header('Contextualização do desafio e abordagens de análise e modelagem')
    st.write("")
    st.markdown("---")

    st.header('Desafio')
    st.write("""
    <div style="text-align: justify">
    Esta aplicação visa explorar os dados históricos de preços do petróleo Brent, fornecendo visualizações e insights sobre diversas influências, como situações geopolíticas, econômicas e a demanda global.<br><br>

    Além disso, o projeto inclui o desenvolvimento de um modelo de machine learning que utiliza análise de séries temporais para prever o preço diário do petróleo contribuindo para tomadas de decisões.<br><br>

    Por fim, o MVP do modelo será implantado em produção usando a ferramenta Streamlit, permitindo acesso fácil e interativo aos resultados e às previsões geradas pelo modelo.
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("---")
    st.header('Dataset')
    st.write("""
    <div style="text-align: justify">
    O conjunto de dados consiste nas colunas de data e preço do petróleo Brent e é possível explorar o histórico de preços desde janeiro de 2000 até parcial de maio de 2024.<br><br>

    Esses dados foram obtidos por meio da API yfinance, utilizando o índice BZ-F como referência. Para mais informações e consulta dos dados, você pode acessar o site: http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view. 
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("---")
    st.header('Modelo de Machine Learning Prophet')
    st.write("""
    <div style="text-align: justify">
     Desenvolvido pelo Facebook com foco em séries temporais, o Prophet é uma ferramenta robusta que oferece a flexibilidade de ajustar vários parâmetros. Além disso, é capaz de lidar de forma eficaz com a sazonalidade e os feriados, tornando-o uma escolha ideal para modelagem e previsão de séries temporais complexas como o preço do petróleo Brent.<br><br>

    No ajuste do modelo foi definido um intervalo de confiança de 95% (interval_width=0.95) para as previsões e a sazonalidade diária foi desconsiderada (daily_seasonality=False). O número de períodos futuros para as previsões foi determinado dinamicamente com base na interação com o usuário. Por fim, apenas os dias úteis foram considerados na frequência dos dados temporais (freq='B').<br><br>
    
    Como métrica de avaliação do modelo, foi adotado o MAPE (Erro Percentual Absoluto Médio), que avalia o quanto as previsões estão próximas dos valores reais em termos percentuais, e o R² (Coeficiente de Determinação), que determina se as variações nos dados são explicadas pelo modelo.
    </div>
    """, unsafe_allow_html=True)

################################## DADOS HISTÓRICOS + INSIGHTS #######################################

elif pagina == "Dados históricos + Insigths":
    st.title('Conhecendo os dados históricos do preço de Petróleo Brent')
    st.write("")
    st.write("<h6 style='font-weight: 300'>Nessa aba você pode explorar, realizar consultas nos dados históricos reais dos preços de petróleo Brent e ter acesso a insights!</h6>", unsafe_allow_html=True)    
    st.write("")

    ## Filtrar período
    st.markdown("---")   
    st.write("### Consultas e visualizações")
    st.write("")
    st.write("###### Selecione o período que  deseja exibir")
    inicio_filtro = st.date_input("Data de Início", min_value=pd.to_datetime(df_cotacoes.index).min(), max_value=pd.to_datetime(df_cotacoes.index).max(), value=pd.to_datetime(df_cotacoes.index).min())
    fim_filtro = st.date_input("Data de Fim", min_value=pd.to_datetime(df_cotacoes.index).min(), max_value=pd.to_datetime(df_cotacoes.index).max(), value=pd.to_datetime(df_cotacoes.index).max())

    df_filtrado = df_cotacoes.loc[inicio_filtro:fim_filtro]
    
    # Cria Big numbers
    # Calculando o preço médio, mínimo e máximo
    preco_medio = df_filtrado['Preço'].mean()
    preco_minimo = df_filtrado['Preço'].min()
    preco_maximo = df_filtrado['Preço'].max()

    # Arredondando os valores dos big numbers
    preco_medio = round(preco_medio, 2)
    preco_minimo = round(preco_minimo, 2)
    preco_maximo = round(preco_maximo, 2)

    # Exibindo os big numbers lado a lado
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Preço Médio", value=preco_medio)
    col2.metric(label="Preço Mínimo", value=preco_minimo)
    col3.metric(label="Preço Máximo", value=preco_maximo)

    # Gráfico de preços históricos
    
    # Tratando coluna de data
    df_filtrado.index = df_filtrado.index.strftime('%Y-%m-%d')

    # Plota gráfico de linha
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtrado.index, y=df_filtrado['Preço'], mode='lines', name='Preço do Petróleo'))

    fig.update_layout(
        title='Preços históricos de petróleo',
        title_font_size=20,
        xaxis_title='Data',
        yaxis_title='Preço',
        template='plotly_white'
    )

    # Exibir o gráfico Plotly
    st.plotly_chart(fig)

    # Exibir a tabela com os dados do preço do petróleo
    
    # Calcular as variações dos preços em relação ao dia anterior
    df_filtrado['Variação Diária'] = df_filtrado['Preço'].diff()

    # Calcular as variações percentuais dos preços em relação ao dia anterior
    df_filtrado['Variação Diária (%)'] = (df_filtrado['Variação Diária'] / df_filtrado['Preço'].shift(1)) * 100

    # Formatar os números das colunas de variação
    df_filtrado['Variação Diária'] = df_filtrado['Variação Diária'].map('{:.2f}'.format)
    df_filtrado['Variação Diária (%)'] = df_filtrado['Variação Diária (%)'].map('{:.2f}%'.format)

    # Exibir a tabela com os dados do preço do petróleo
    st.write("###### Visualize o detalhamento dos dados exibidos no gráfico em tabela com cálculo de variações diárias")
    if st.checkbox('Mostrar/Esconder Tabela'):
        st.write(df_filtrado)

    st.write("")
    st.write("")

    ## Cria consulta de preço em um dia específico
    st.write("###### Descubra o preço do petróleo em um dia específico")

    # Exibir a lista suspensa com as datas 
    selected_date = st.selectbox('Selecione uma data:', df_cotacoes.index.strftime('%Y-%m-%d').tolist())

    # Encontra preço pra data selecionada
    preco_selecionado = df_cotacoes.loc[selected_date, 'Preço']

    # Exibe o preço do petróleo para a data selecionada
    st.write(f'O preço do petróleo em {selected_date} foi de ${preco_selecionado:.2f}')

    st.markdown("---")

    # Análise ano a ano
    st.write("### Insights dos últimos 3 anos")
    st.write("")
    st.write("<h6 style='font-weight: 300'>  O petróleo Brent é um tipo de petróleo de alta qualidade extraído do Mar do Norte e amplamente usado como referência internacional para precificação do petróleo.</h6>", unsafe_allow_html=True)    
    st.write("")
    st.write("<h6 style='font-weight: 300'>  Seu preço é influenciado por diversos fatores, incluindo eles a oferta e demanda global, movimentos geopolíticos, acordos políticos, conjuntura econômica global e flutuações de câmbio, além de eventos como desastres naturais e climáticos.</h6>", unsafe_allow_html=True)    
    st.write("")


    #################### 2021 ######################################################################################################


    st.write("###### 2021")

    import plotly.graph_objs as go

# Filtrar apenas os dados do ano de 2021
    df_2021 = df_cotacoes[df_cotacoes.index.year == 2021]

# Plota gráfico de linha
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_2021.index, y=df_2021['Preço'], mode='lines', name='Preço do Petróleo'))

# Configurar layout do gráfico
    fig.update_layout(
        title='Preços históricos de petróleo em 2021',
        title_font_size=20,
        xaxis_title='Data',
        yaxis_title='Preço',
        template='plotly_white',
        yaxis=dict(range=[20, 130])  # Define o intervalo do eixo y
    )

# Exibir o gráfico Plotly
    st.plotly_chart(fig)

# Calcular a média de preço
    media_preco_2021 = df_2021['Preço'].mean()

# Arredondar o valor da média
    media_preco_2021 = round(media_preco_2021, 2)

# Exibir como big number
    st.metric(label="Preço Médio em 2021", value=media_preco_2021)

    st.write("""
        <h6 style='font-weight: 300; text-align: justify'>
         O ano de 2021 foi caracterizado principalmente pelo período de pandemia globale apresentou os menores preços desde 2004. No entanto, por volta do segundo semestre, ocorreu uma reabertura gradual das atividades industriais, de transporte e consumo, o que resultou em uma tímida recuperação nos preços do petróleo devido à maior demanda. A OPEP+ (Organização dos Países Exportadores de Petróleo) também anunciou ao longo do ano cortes na produção, o que também pode ter afetado os preços.
     </h6>
    """, unsafe_allow_html=True)


    #################### 2022 ######################################################################################################
    st.write("###### 2022")

# Filtrar apenas os dados do ano de 2022
    df_2022 = df_cotacoes[df_cotacoes.index.year == 2022]

# Plota gráfico de linha
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_2022.index, y=df_2022['Preço'], mode='lines', name='Preço do Petróleo'))

# Configurar layout do gráfico
    fig.update_layout(
        title='Preços históricos de petróleo em 2022',
        title_font_size=20,
        xaxis_title='Data',
        yaxis_title='Preço',
        template='plotly_white',
        yaxis=dict(range=[20, 130])  # Define o intervalo do eixo y
    )

# Exibir o gráfico Plotly
    st.plotly_chart(fig)

# Calcular a média de preço
    media_preco_2022 = df_2022['Preço'].mean()

# Arredondar o valor da média
    media_preco_2022 = round(media_preco_2022, 2)

# Exibir como big number
    st.metric(label="Preço Médio em 2022", value=media_preco_2022)

    st.write("""
        <h6 style='font-weight: 300; text-align: justify'>
        O ano de 2022 foi marcado pela recuperação gradual da economia global, impulsionada pela distribuição de vacinas contra a COVID-19 e pela retomada de atividades comerciais e industriais. A demanda por petróleo aumentou significativamente ao longo do ano, especialmente nos setores de transporte e manufatura. No entanto, a oferta global permaneceu relativamente estável, com os países produtores de petróleo mantendo cotas de produção mais baixas em um esforço para manter os preços estáveis.
    </h6>
    """, unsafe_allow_html=True)


    #################### 2023 ######################################################################################################
    st.write("###### 2023")

# Filtrar apenas os dados do ano de 2023
    df_2023 = df_cotacoes[df_cotacoes.index.year == 2023]

# Plota gráfico de linha
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_2023.index, y=df_2023['Preço'], mode='lines', name='Preço do Petróleo'))

# Configurar layout do gráfico
    fig.update_layout(
        title='Preços históricos de petróleo em 2023',
        title_font_size=20,
        xaxis_title='Data',
        yaxis_title='Preço',
        template='plotly_white',
        yaxis=dict(range=[20, 130])  # Define o intervalo do eixo y
    )

# Exibir o gráfico Plotly
    st.plotly_chart(fig)

# Calcular a média de preço
    media_preco_2023 = df_2023['Preço'].mean()

# Arredondar o valor da média
    media_preco_2023 = round(media_preco_2023, 2)

# Exibir como big number
    st.metric(label="Preço Médio em 2023", value=media_preco_2023)

    st.write("""
        <h6 style='font-weight: 300; text-align: justify'>
        O ano de 2023 foi caracterizado por uma recuperação contínua da economia global, com muitos países registrando crescimento robusto no PIB e aumento da atividade industrial. A demanda por petróleo continuou a crescer, superando os níveis pré-pandêmicos em muitas regiões. No entanto, a oferta global também aumentou, com os países produtores de petróleo respondendo ao aumento da demanda com aumentos na produção. Isso ajudou a manter os preços relativamente estáveis ao longo do ano, apesar do aumento da demanda.
    </h6>
    """, unsafe_allow_html=True)



    #################### 2024 ######################################################################################################
    st.markdown("---")

    st.write("")
    st.write("")

    st.write("###### 2024")

    import plotly.graph_objs as go

# Filtrar apenas os dados do ano de 2024
    df_2024 = df_cotacoes[df_cotacoes.index.year == 2024]

# Plota gráfico de linha
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_2024.index, y=df_2024['Preço'], mode='lines', name='Preço do Petróleo'))

# Configurar layout do gráfico
    fig.update_layout(
        title='Preços históricos de petróleo em 2024',
        title_font_size=20,
        xaxis_title='Data',
        yaxis_title='Preço',
        template='plotly_white',
        yaxis=dict(range=[20, 130])  # Define o intervalo do eixo y
    )

# Exibir o gráfico Plotly
    st.plotly_chart(fig)
# Calcular a média de preço
    media_preco_2024 = df_2024['Preço'].mean()

# Arredondar o valor da média
    media_preco_2024 = round(media_preco_2024, 2)

# Exibir como big number
    st.metric(label="Preço Médio em 2024", value=media_preco_2024)


    st.write("""
        <h6 style='font-weight: 300; text-align: justify'>
    A DoE (Departamento de energia em português) projeta que o preço do petróleo Brent feche em torno de US$82,42. A projeção otimista se dá a crença de que os níveis de oferta e demanda se mantenham estáveis gerando uma estabilização no preço. 
        </h6>
    """, unsafe_allow_html=True)

############################ PREVISAO ########################################################################################
elif pagina == "Modelo de previsão":
    st.title('Previsão do preço do Petróleo Brent')
    st.write("")
    st.write("<h6 style='font-weight: 300'>Nessa aba você pode prever o preço do petróleo para os próximos dias!</h6>", unsafe_allow_html=True)    
    st.write("")

    st.markdown("---")

    n_dias = st.slider('Quantos dias você deseja prever?', 1, 7)

    # DF
    df_treino = df_cotacoes[df_cotacoes.index >= "2022-05-01"].reset_index()

    # Renomeando as colunas 
    df_treino = df_treino.rename(columns={'Date': 'ds', 'Preço': 'y'})

    modelo = Prophet(interval_width=0.95, daily_seasonality=False)
    modelo.fit(df_treino)

    futuro = modelo.make_future_dataframe(periods=n_dias, freq='B')

    previsao = modelo.predict(futuro)

    # Criar uma cópia do DataFrame de previsão
    previsao_formatada = previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_dias).copy()

    # Ajustar o formato da coluna de data
    previsao_formatada['ds'] = previsao_formatada['ds'].dt.strftime('%Y-%m-%d')

    # Arredondar as previsões
    previsao_formatada['yhat'] = round(previsao_formatada['yhat'], 2)
    previsao_formatada['yhat_lower'] = round(previsao_formatada['yhat_lower'], 3)
    previsao_formatada['yhat_upper'] = round(previsao_formatada['yhat_upper'], 3)

    # Exibir a tabela de previsões com cabeçalhos personalizados e formato de data ajustado
    st.write("###### Essa é a previsão para os dias que você deseja prever:")
    st.dataframe(previsao_formatada.rename(columns={'ds': 'Data da previsão', 'yhat': 'Previsão do preço', 'yhat_lower': 'Limite de previsão inferior', 'yhat_upper': 'Limite de previsão superior'}))


    ######################## CRIA VISUALIZAÇÕES COM AS PREVISÕES ######################################################
    st.markdown("---")
    ######################## CRIA VISUALIZAÇÕES COM AS PREVISÕES ######################################################
    graph_1 = plot_plotly(modelo, previsao)

    # Adicionar legendas ao gráfico
    layout = dict(
        title='Previsão de preço do petróleo com intervalo de confiança',
        xaxis_title='Data',
        yaxis_title='Preço',
        legend_title='Legenda',
        legend=dict(
            orientation='h',  # Posição da legenda (horizontal)
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    # Atualizar o layout do gráfico
    graph_1.update_layout(layout)

    # Adicionar legendas às séries de dados
    graph_1.data[0].name = 'Previsão'
    graph_1.data[1].name = 'Limite de previsão inferior'
    graph_1.data[2].name = 'Limite de previsão superior'

    # Exibir o gráfico com as legendas
    st.plotly_chart(graph_1)


    graph_2 = plot_components_plotly(modelo, previsao)

    # Adicionar título ao gráfico graph_2
    graph_2.update_layout(title_text="Decomposição da Previsão: Tendência, Sazonalidade e Tendências Irregulares")

    # Exibir o gráfico graph_2
    st.plotly_chart(graph_2)


    ###################### AVALIAÇÃO DO MODELO ########################################################################
    previsao_cols = ['ds', 'yhat']  
    valores_reais_cols = ['ds', 'y']

    previsao_ppt = previsao[previsao_cols]
    valores_reais = df_treino[valores_reais_cols]

    resultados = pd.merge(previsao_ppt, valores_reais, on='ds', how='inner')
    resultados['erro_perc_abs'] = np.abs((resultados['y'] - resultados['yhat']) / resultados['y']) * 100
    mape = np.mean(resultados['erro_perc_abs'])


    # Calcular a média dos valores observados
    mean_observed = df_treino['y'].mean()

    # Calcular a soma total dos quadrados (SS_total)
    df_treino['ss_total'] = (df_treino['y'] - mean_observed) ** 2
    ss_total = df_treino['ss_total'].sum()

    # Calcular a soma dos quadrados dos resíduos (SS_residual)
    previsao['ss_residual'] = (previsao['yhat'] - df_treino['y']) ** 2
    ss_residual = previsao['ss_residual'].sum()

    # Calcular o coeficiente de determinação (R-squared)
    r_squared = 1 - (ss_residual / ss_total)
    st.markdown("---")
    # Exibir o resultado
    #st.write(f"Coeficiente de Determinação (R-squared): {r_squared:.2f}")
    st.write(f"O modelo apresenta um MAPE de {mape:.2f}% e um Coeficiente de Determinação (R-squared) de {r_squared:.2f}%")

    #st.write(resultados)

    # Arredondar os valores na coluna de previsão
    resultados['yhat'] = resultados['yhat'].round(2)

    # Criar um novo DataFrame com as colunas renomeadas e formatadas
    df_mape = resultados[['ds', 'yhat', 'y', 'erro_perc_abs']].copy()

    # Renomear as colunas
    df_mape.columns = ['Data', 'Previsão', 'Realizado', 'Erro percentual absoluto']

    # Formatar a coluna 'Data' como ano-mês-dia
    df_mape['Data'] = df_mape['Data'].dt.strftime('%Y-%m-%d')

    #  Ordenar a tabela pela data mais recente
    df_mape = df_mape.sort_values(by='Data', ascending=False)

    # Exibir a tabela com a opção de expandir/recolher
    if st.checkbox('Selecione para visualizar o detalhamento da Previsão x Realizado diário'):
        st.write(df_mape)


st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

st.markdown("---")
st.write("<h6 style='text-align: center; color: #6E6E6E;'>Desenvolvido por Verônica Urzedo</h6>", unsafe_allow_html=True)
