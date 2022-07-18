from datetime import datetime,date, timedelta
import math
import pandas as pd
import numpy as np
from io import BytesIO
import joblib
import pickle
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from minio import Minio
#from sqlalchemy.engine import create_engine
from sklearn.model_selection import train_test_split
#from pycaret.classification import *
#from pycaret.datasets import get_data
#from skopt import dummy_minimize, gp_minimize, forest_minimize
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np


data_lake_server = Variable.get("data_lake_server")
data_lake_login = Variable.get("data_lake_login")
data_lake_password = Variable.get("data_lake_password")
science_team = Variable.get("email_science_team")

default_args = {
'start_date': datetime(2022, 7, 10),
}

client = Minio(
        data_lake_server,
        access_key=data_lake_login,
        secret_key=data_lake_password,
        secure=False
    )


def _balanceamento():

    client.fget_object(
        "processing",
        "creditcard.parquet",
        "/tmp/creditcard.parquet"
    )
    df = pd.read_parquet("/tmp/creditcard.parquet")
    qtd_fraud=df['class'].loc[df['class']==1].count()
    qtd_not_fraud=df['class'].loc[df['class']==0].count()
    prop=math.floor((qtd_not_fraud/qtd_fraud)/100)

    df_mod = pd.read_parquet("/tmp/creditcard.parquet")
    fraude = df_mod[df_mod['class'] == 1]
    n_fraude = df_mod[df_mod['class'] == 0]

    # Aumentando (em prop vezes) o numero de observacoes no dataframe fraude
    fraude = pd.concat([fraude] * prop, ignore_index=True)
    # Definindo o numero de nao fraudes de forma aleatoria
    n_fraude = n_fraude.sample(n = fraude.shape[0] * prop, axis=0, random_state = 1)
    # Concatenando os dataframes novos e criando outro dataframe
    df_mod_balanced = pd.concat([fraude,n_fraude])

    df_mod_balanced.to_csv( "/tmp/creditcardbalanced.csv"
                ,index=False
            )

def _train_model():

    df_mod_balanced=pd.read_csv( "/tmp/creditcardbalanced.csv")
    # Definindo as variaveis explicativas e a variavel target
    x = df_mod_balanced.drop(['class'],axis=1)
    y = df_mod_balanced['class']
    model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight={},
                       criterion='entropy', max_depth=9, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0,
                       min_samples_leaf=6, min_samples_split=10,
                       min_weight_fraction_leaf=0.0, n_estimators=230,
                       n_jobs=-1, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
    # treina o modelo com a amostra.
    model.fit(x, y)
    df_final1 = pd.concat((x,y),axis=1)

    score_mean = (cross_val_score(model, x, y, cv=10)).mean()
    if score_mean >= 0.90:
        joblib.dump(model,"/tmp/model.pkl")
        df_total= pd.read_parquet("/tmp/creditcard.parquet")
        x = df_total.drop(['class'],axis=1)
        y = df_total['class']
        model.fit(x, y)
        df_final = pd.concat((x,y),axis=1)
        df_final.to_csv( "/tmp/final.csv",index=False)

    


def load_curated():

    #carrega os dados a partir da área de staging.
    df_ = pd.read_csv("/tmp/final.csv")

    #converte os dados para o formato parquet.    
    df_.to_parquet(
            "/tmp/final.parquet"
            ,index=False
    )

    #carrega os dados para o Data Lake.
    client.fput_object(
        "curated",
        "credit_card_final.parquet",
        "/tmp/final.parquet"
    )
    client.fput_object(
        "curated",
        "model.pkl",
        "/tmp/model.pkl"
    )

with DAG('best_model', schedule_interval='@daily', default_args=default_args) as dag:
    
    
    balancing_task = PythonOperator(
        task_id = "balancing_data",
        python_callable = _balanceamento
    )

    train_task = PythonOperator(
        task_id="train_best_model",
        python_callable=_train_model,
        email_on_failure = True,
        email = science_team
    )
    
    load_curated_task = PythonOperator(
    task_id='load_curated_',
    provide_context=True,
    python_callable=load_curated,
    dag=dag
)



    
balancing_task >>  train_task >>load_curated_task 