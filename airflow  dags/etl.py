from datetime import datetime,date, timedelta
import math
import pandas as pd
import numpy as np
from io import BytesIO
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from minio import Minio
from sqlalchemy.engine import create_engine


DEFAULT_ARGS = {
    'owner': 'Airflow',
    'depends_on_past': False,
    'start_date': datetime(2022, 7, 10),
}

dag = DAG('etl_fraud_detection', 
          default_args=DEFAULT_ARGS,
          schedule_interval="@once"
        )

data_lake_server = Variable.get("data_lake_server")
data_lake_login = Variable.get("data_lake_login")
data_lake_password = Variable.get("data_lake_password")
dags_path = "/opt/airflow/dags"

database_server = Variable.get("database_server")
database_login = Variable.get("database_login")
database_password = Variable.get("database_password")
database_name = Variable.get("database_name")


url_connection = "mysql+pymysql://{}:{}@{}/{}".format(
                 str(database_login)
                ,str(database_password)
                ,str(database_server)
                ,str(database_name)
                )

engine = create_engine(url_connection)

client = Minio(
        endpoint=data_lake_server,
        access_key=data_lake_login,
        secret_key=data_lake_password,
        secure=False
    )

def extract():

    #query para consultar os dados.
    query = """SELECT * FROM creditcard;"""

    df_ = pd.read_sql_query(query,engine)
    
    #persiste os arquivos na área de Staging.
    df_.to_csv( "/tmp/creditcard.csv"
                ,index=False
            )

def transform():
    
    #carrega os dados a partir da área de staging.
    df_ = pd.read_csv("/tmp/creditcard.csv")
    df_mod = df_.copy()
    #aplica as transformações criando as features para o treinamento do modelo
    df_mod['V1_mod'] = df_.v1.map(lambda x: 1 if x < -3 else 0)
    df_mod['V2_mod'] = df_.v2.map(lambda x: 1 if x > 2.5 else 0)
    df_mod['V3_mod'] = df_.v3.map(lambda x: 1 if x < -3.5 else 0)
    df_mod['V4_mod'] = df_.v4.map(lambda x: 1 if x > 2 else 0)
    df_mod['V5_mod'] = df_.v5.map(lambda x: 1 if x < -4.5 else 0)
    df_mod['V6_mod'] = df_.v6.map(lambda x: 1 if x < -2.5 else 0)
    df_mod['V7_mod'] = df_.v7.map(lambda x: 1 if x < -1.5 else 0)
    df_mod['V9_mod'] = df_.v9.map(lambda x: 1 if x < -2 else 0)
    df_mod['V10_mod'] = df_.v10.map(lambda x: 1 if x < -2 else 0)
    df_mod['V11_mod'] = df_.v11.map(lambda x: 1 if x > 2 else 0)
    df_mod['V12_mod'] = df_.v12.map(lambda x: 1 if x < -2.5 else 0)
    df_mod['V14_mod'] = df_.v14.map(lambda x: 1 if x < -2.5 else 0)
    df_mod['V16_mod'] = df_.v16.map(lambda x: 1 if x < -2 else 0)
    df_mod['V17_mod'] = df_.v17.map(lambda x: 1 if (x < -2) | (x > 2) else 0)
    df_mod['V18_mod'] = df_.v18.map(lambda x: 1 if (x < -2) | (x > 2) else 0)
    df_mod['V19_mod'] = df_.v19.map(lambda x: 1 if (x > 1.5) | (x < -1.75) else 0)
    df_mod['V21_mod'] = df_.v21.map(lambda x: 1 if x > 0.3 else 0)
    #persiste os arquivos na área de Staging.
    df_mod.to_csv( "/tmp/creditcardfinal.csv"
                ,index=False
            )
    
def load():

    #carrega os dados a partir da área de staging.
    df_ = pd.read_csv("/tmp/creditcardfinal.csv")

    #converte os dados para o formato parquet.    
    df_.to_parquet(
            "/tmp/creditcard.parquet"
            ,index=False
    )

    #carrega os dados para o Data Lake.
    client.fput_object(
        "processing",
        "creditcard.parquet",
        "/tmp/creditcard.parquet"
    )


extract_task = PythonOperator(
    task_id='extract_data_from_database',
    provide_context=True,
    python_callable=extract,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    provide_context=True,
    python_callable=transform,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_file_to_data_lake',
    provide_context=True,
    python_callable=load,
    dag=dag
)

clean_task = BashOperator(
    task_id="clean_files_on_staging",
    bash_command="rm -f /tmp/*.csv;rm -f /tmp/*.json;rm -f /tmp/*.parquet;",
    dag=dag
)

extract_task >> transform_task >> load_task >> clean_task