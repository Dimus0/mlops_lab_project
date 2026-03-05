from airflow.models import DagBag
import os

def test_dag_import():

    dag_folder = os.path.join(os.path.dirname(__file__), '..', 'dags')
    dag = DagBag(dag_folder, include_examples=False)

    assert len(dag.import_errors) == 0, f"DAG import errors: {dag.import_errors}"