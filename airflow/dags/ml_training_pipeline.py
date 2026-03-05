from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime
import json
import os
from mlflow.tracking import MlflowClient

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

def transition_model_to_staging(**kwargs):
    tracking_uri = f"sqlite:///{PROJECT_DIR}/mlflow.db"
    client = MlflowClient(tracking_uri=tracking_uri)

    model_name = "RandomForestClassifier_TelcoChurn"
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"No versions found for model {model_name}")
        return
    
    latest_version = max([int(v.version) for v in versions])
    
    # Змінюємо статус на Staging
    client.transition_model_version_stage(
        name=model_name,
        version=str(latest_version),
        stage="Staging",
        archive_existing_versions=True # Автоматично відправляє попередні Staging-версії в Архів
    )
    print(f"Модель {model_name} (версія {latest_version}) успішно переведена у Staging!")


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

    stage_register = PythonOperator(
        task_id='deploy_model',
        python_callable=transition_model_to_staging
    )

    stage_stop = BashOperator(
        task_id='stop_pipeline',
        bash_command='echo "Accuracy <= 0.85. Pipeline stopped."'
    )

    stage_prepare >> stage_train >> stage_evaluate_model
    stage_evaluate_model >> [stage_register, stage_stop]