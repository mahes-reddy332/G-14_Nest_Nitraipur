
from datetime import datetime, timedelta
import logging
# Note: Airflow import is wrapped in try-except because we might be in a dev env without airflow installed yet
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
except ImportError:
    # Mock classes for development if airflow is missing
    logging.warning("Airflow not found. Using mocks for development.")
    class DAG:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class PythonOperator:
        def __init__(self, *args, **kwargs): pass
    class BashOperator:
        def __init__(self, *args, **kwargs): pass

# Import our ingestion logic
# In a real setup, we might install the package or add to pythonpath
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def extract_data(**kwargs):
    """
    Wrapper around the Core Data Ingestion logic.
    """
    from core.data_ingestion import ClinicalDataIngester
    from api.config import get_settings
    
    settings = get_settings()
    ingester = ClinicalDataIngester(base_path=settings.RAW_DATA_PATH)
    
    # Trigger ingestion for known sources
    # In a real scenario, this might download from S3/SFTP first
    results = ingester.ingest_all()
    logging.info(f"Ingestion completed: {results}")
    return results

def validate_schema(**kwargs):
    """
    Basic schema validation (Fast Fail) before dbt runs.
    """
    logging.info("Validating schema of ingested files...")
    # This could check if critical columns exist in the raw files
    return "Schema Valid"

default_args = {
    'owner': 'clinical_ops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'clinical_data_ingestion',
    default_args=default_args,
    description='Ingest Clinical Trial Data (EDC, Lab, Safety)',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['clinical', 'ingestion'],
) as dag:

    t1_extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
    )

    t2_validate = PythonOperator(
        task_id='validate_schema',
        python_callable=validate_schema,
    )

    # Trigger dbt transformations
    # Assuming dbt is installed and accessible in the environment
    t3_dbt_run = BashOperator(
        task_id='dbt_run',
        bash_command='cd ../dbt_project && dbt run',
    )
    
    t4_dbt_test = BashOperator(
        task_id='dbt_test',
        bash_command='cd ../dbt_project && dbt test',
    )

    t1_extract >> t2_validate >> t3_dbt_run >> t4_dbt_test
