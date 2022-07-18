import matplotlib
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
showPyplotGlobalUse = False

from gettext import install
from certifi import where

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set()

import tqdm

import warnings
warnings.filterwarnings('ignore')


import missingno as msno

st.image('principal.png', use_column_width = 'always' )

paginas = ['Principal', 'Dados', 'Análise', 'Graficos', 'Creditos']
pagina = st.sidebar.selectbox('Selecione a opção desejada', paginas)

if pagina == 'Principal':
    st.title('Análise de fraude em cartões de crédito')
    st.write('Objetivo')
    st.write('Identificar potenciais operações bancarias fraudulentas utilizando dados das transações com marchine learning para eviar compras indesejadas.')
    st.image('2.jpg')

    st.write('Desenvolvedores')
    st.write('Paula Muniz')
    st.write("[Linkedin](https://www.linkedin.com/in/paula-pereira-muniz/)")
    
    st.write('Rafael Panegassi')
    st.write("[Linkedin](https://www.linkedin.com/in/rafaelpanegassi/)")
    
    st.write('Arildo Júnior')
    st.write("[Linkedin](https://www.linkedin.com/in/arildo-de-azevedo-junior/)")
    
    st.write('André Balbi Aguiar')
    st.write("[Linkedin](https://www.linkedin.com/in/andre-balbi-aguiar/)")
    
    st.write('Repositório')
    st.write("[GIT](https://github.com/Stack-Labs-Dash-Squad/Credit-Card-Fraud-Detection)")

# importando os dados

dados = pd.read_pickle('creditcard.pkl')

# filtros para a tabela

st.sidebar.image('1.jpg', use_column_width = 'always' )

if pagina == 'Dados':
    
    st.sidebar.markdown('Opções')
    
    checkbox_head = st.sidebar.button('Ler dados')

    if checkbox_head:

        st.write('Lendo uma fração dos dados')
        st.write(dados.head(100))

    checkbox_describe = st.sidebar.button('Descrição dos dados')

    if checkbox_describe:
        st.write('Descrição dos dados')
        st.write(dados.describe())

if pagina == 'Análise':
    
    st.sidebar.markdown('Opções')
    
    checkbox_fraud_number = st.sidebar.button('Numero de fraudes')

    if checkbox_fraud_number:
        st.write('Verificando o numero de fraudes e desequilibrio do dados')
        fraud = dados[dados['Class'] == 1]
        valid = dados[dados['Class'] == 0]
        outlierFraction = len(fraud)/float(len(valid))
        st.write(outlierFraction)
        st.write('Fraud Cases: {}'.format(len(dados[dados['Class'] == 1])))
        st.write('Valid Transactions: {}'.format(len(dados[dados['Class'] == 0])))

    checkbox_fraud_amount_describe = st.sidebar.button('Detalhes do valor fraudulentas')

    if checkbox_fraud_amount_describe:
        st.write('Detalhes do valor para Transação Fraudulenta')

        st.write(dados[dados['Class'] == 1].describe())

    checkbox_valid_amount_describe = st.sidebar.button('Detalhes do valor não Fraudulenta')

    if checkbox_valid_amount_describe:
        st.write('Detalhes do valor para Transação não Fraudulenta')

        st.write(dados[dados['Class'] == 0].describe())
    
    checkbox_estatisticas_descritivas = st.sidebar.button('Estatisticas descritivas')

    if checkbox_estatisticas_descritivas:
        st.write('Estatisticas descritivas dos tempos de nao-fraude e fraude')

        st.write(dados.groupby('Class')['Time'].describe())

if pagina == 'Graficos':   
    
    st.sidebar.markdown('Opções')
    
    checkbox_explorando_as_variaveis = st.sidebar.button('Explorando as variavies')

    if checkbox_explorando_as_variaveis:
        st.write('Exploração das variáveis')

        # definindo uma matriz de graficos
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
        bins = 25

        # plotando o tempo por numero de transacoes das fraudes
        figura = (ax1.hist(dados.Time[dados.Class == 1], bins = bins))
        figura = (ax1.set_title('Fraude'))

        # plotando o tempo por numero de transacoes das nao-fraudes
        figura_2 = (ax2.hist(dados.Time[dados.Class == 0], bins = bins))
        figura_2 = (ax2.set_title('Legitimas'))

        # plotando as legendas dos graficos
    
        plt.xlabel('Tempo (s)')
        plt.ylabel('No. Transacoes')
        plt.show()
        st.pyplot()

    checkbox_grafico_de_dispersao = st.sidebar.button('Grafico de dispersão')

    if checkbox_grafico_de_dispersao:
        st.write('Nota-se que as Fraudes se espalham ao longo do dia com dois picos (de manha e a tarde), sendo que no periodo da tarde, as fraudes se diferenciam bastante com relacao as legais')

        # plotando o grafico de dispersao do montante pelo tempo das fraudes e nao-fraudes

        # plotando uma matriz de graficos
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,6))

        # dispersao do montante pelo tempo das fraudes
        figura = (ax1.scatter(dados.Time[dados.Class == 1], dados.Amount[dados.Class == 1]))
        figura = (ax1.set_title('Transações Fradulentas'))

        # dispersao do montante pelo tempo das nao-fraudes
        figura = (ax2.scatter(dados.Time[dados.Class == 0], dados.Amount[dados.Class == 0]))
        figura = (ax2.set_title('Transações Legítimas'))

        # plotando as legendas dos graficos
        plt.xlabel('Time (in Seconds)')
        plt.ylabel('Amount')
        plt.show()
        st.pyplot()

    checkbox_explorando_cada_variavel = st.sidebar.button('Explorando as 28 variaveis')

    if checkbox_explorando_cada_variavel:
        st.write('Explorando cada uma das 28 variaveis')

        # Armazenando as colunas das variaveis explicativas em uma variavel
        num_cols = dados.columns[1:29]

        st.write('Distribuicao das colunas')

        # definindo o tamanho dos graficos
    
        plt.figure(figsize=(15, 80))
    
        # plotando histogramas para cada uma das features
    
        for i, col in tqdm.tqdm_notebook(enumerate(num_cols)):
            plt.subplot(15, 2, i + 1)
            sns.distplot(dados[col][dados.Class == 1], bins=50, color='r') # Fraude --> vermelho
            sns.distplot(dados[col][dados.Class == 0], bins=50) # N Fraude --> azul
            st.pyplot()
        
if pagina == 'Creditos':
    
    st.write('stack-academy')
    st.write("[Page Web](https://stacktecnologias.com.br/)")
    st.write("[Linkedin](https://www.linkedin.com/company/stack-tecnologias/)")
    
    st.write('kaggle')
    st.write("[Page Web](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?page=2)")
    
    st.write('streamlit')
    st.write("[Page Web](https://docs.streamlit.io/)")