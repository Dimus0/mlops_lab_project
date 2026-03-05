from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime
import json
import os

PROJECT_DIR = '/opt/airflow/dags'

def evaluate_model(**kwargs):

    ti = kwargs['ti']
    metrics_path = os.path.join(PROJECT_DIR, "models", "metrics", "metrics.json")

    try:
        with open(metrics_path, 'r', encoding="utf-8") as f:
            metrics = json.load(f)

        f1_test = metrics.get('f1_test', 0)
        ti.xcom_push(key='model_metrics', value=metrics)

        if f1_test > 0.8:
            ti.xcom_push(key='model_approved', value=True)
            return 'register_model'
        else:
            ti.xcom_push(key='model_approved', value=False)
            return 'stop_pipeline'
        
    except Exception as e:
        print(f"Error reading metrics: {e}")
        return 'stop_pipeline'    

with DAG(
    dag_id='ml_training_pipeline',
    start_date=datetime(2025, 1, 1),
    schedule_interval='None',
    catchup=False
) as dag:
    
    check_data = FileSensor(
        task_id='check_data',
        filepath=os.path.join(PROJECT_DIR, "data", "raw", "telco_dataset.csv"),
        poke_interval=10,
    )
    
    stage_prepare = BashOperator(
        task_id='prepare_data',
        bash_command=f'cd {PROJECT_DIR} && dvc repro prepare'
    )

    stage_train = BashOperator(
        task_id='train_model',
        bash_command=f'cd {PROJECT_DIR} && dvc repro train'
    )

    stage_evaluate_model = BashOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model
    )

    stage_register = BashOperator(
        task_id='deploy_model',
        bash_command='echo "Accuracy > 0.85. Registering in MLflow Model Registry..."'
    )

    stage_stop = BashOperator(
        task_id='stop_pipeline',
        bash_command='echo "Accuracy <= 0.85. Pipeline stopped."'
    )

    stage_prepare >> stage_train >> stage_evaluate_model
    stage_evaluate_model >> [stage_register, stage_stop]